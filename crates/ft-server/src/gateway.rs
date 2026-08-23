//! V2 control：把请求透明反代到 PyTorch/FreeToken worker。
//!
//! 热路径只做：选 worker → 转发 body → 原样流式回传。
//! 禁止在这里缓冲完整 completion，否则会掐断 CUDA Graph 的流式收益。

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode, Uri},
    response::{IntoResponse, Response},
    routing::{any, get},
    Json, Router,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Clone)]
pub struct Gateway {
    inner: Arc<Inner>,
}

struct Inner {
    workers: Vec<String>,
    client: reqwest::Client,
    rr: AtomicUsize,
}

impl Gateway {
    pub fn new(workers: Vec<String>) -> anyhow::Result<Self> {
        let workers: Vec<String> = workers
            .into_iter()
            .map(|w| w.trim_end_matches('/').to_string())
            .filter(|w| !w.is_empty())
            .collect();
        if workers.is_empty() {
            anyhow::bail!("至少需要一个 --worker");
        }
        let client = reqwest::Client::builder()
            .http1_only()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .build()?;
        Ok(Self {
            inner: Arc::new(Inner { workers, client, rr: AtomicUsize::new(0) }),
        })
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/workers", get(workers))
            .fallback(any(proxy))
            .with_state(self)
    }

    fn pick(&self, headers: &HeaderMap) -> &str {
        // 会话粘滞：同一 x-blend-session / x-session-id 钉同一 worker
        if let Some(key) = headers
            .get("x-blend-session")
            .or_else(|| headers.get("x-session-id"))
            .and_then(|v| v.to_str().ok())
        {
            let i = hash_str(key) % self.inner.workers.len();
            return &self.inner.workers[i];
        }
        let i = self.inner.rr.fetch_add(1, Ordering::Relaxed) % self.inner.workers.len();
        &self.inner.workers[i]
    }
}

fn hash_str(s: &str) -> usize {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h as usize
}

async fn healthz(State(gw): State<Gateway>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ok": true,
        "role": "blend-control",
        "workers": gw.inner.workers,
    }))
}

async fn workers(State(gw): State<Gateway>) -> impl IntoResponse {
    Json(serde_json::json!({ "workers": gw.inner.workers }))
}

async fn proxy(State(gw): State<Gateway>, req: Request) -> Response {
    let method = req.method().clone();
    let headers = req.headers().clone();
    let uri = req.uri().clone();
    let path_q = path_and_query(&uri);
    let worker = gw.pick(&headers).to_string();
    let url = format!("{worker}{path_q}");

    let body = match axum::body::to_bytes(req.into_body(), 16 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, format!("read body: {e}")).into_response();
        }
    };

    let mut wreq = gw.inner.client.request(method, &url);
    for (k, v) in headers.iter() {
        if skip_hop(k.as_str()) {
            continue;
        }
        wreq = wreq.header(k, v);
    }

    let resp = match wreq.body(body).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(%url, "{e}");
            return (StatusCode::BAD_GATEWAY, format!("worker {worker}: {e}")).into_response();
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let mut out = Response::builder().status(status);
    if let Some(hs) = out.headers_mut() {
        for (k, v) in resp.headers().iter() {
            if skip_hop(k.as_str()) {
                continue;
            }
            if let Ok(name) = k.as_str().parse::<header::HeaderName>() {
                if let Ok(val) = HeaderValue::from_bytes(v.as_bytes()) {
                    hs.append(name, val);
                }
            }
        }
        hs.insert("x-blend-worker", HeaderValue::from_str(&worker).unwrap_or_else(|_| HeaderValue::from_static("ok")));
    }
    let stream = resp.bytes_stream();
    let body = Body::from_stream(stream);
    out.body(body).unwrap_or_else(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
    })
}

fn path_and_query(uri: &Uri) -> String {
    uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| uri.path().to_string())
}

fn skip_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host" | "connection" | "keep-alive" | "proxy-connection" | "transfer-encoding" | "upgrade" | "te" | "trailer"
    )
}
