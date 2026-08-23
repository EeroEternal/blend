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
use futures_core::Stream;
use std::pin::Pin;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::task::{Context, Poll};

#[derive(Clone)]
pub struct Gateway {
    inner: Arc<Inner>,
}

struct Inner {
    workers: Vec<String>,
    client: reqwest::Client,
    rr: AtomicUsize,
    inflight: Vec<AtomicUsize>,
    hits: Vec<AtomicUsize>,
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
        let n = workers.len();
        Ok(Self {
            inner: Arc::new(Inner {
                workers,
                client,
                rr: AtomicUsize::new(0),
                inflight: (0..n).map(|_| AtomicUsize::new(0)).collect(),
                hits: (0..n).map(|_| AtomicUsize::new(0)).collect(),
            }),
        })
    }

    pub fn router(self) -> Router {
        Router::new()
            .route("/healthz", get(healthz))
            .route("/v1/workers", get(workers))
            .fallback(any(proxy))
            .with_state(self)
    }

    fn pick_idx(&self, headers: &HeaderMap) -> usize {
        if let Some(key) = headers
            .get("x-blend-session")
            .or_else(|| headers.get("x-session-id"))
            .and_then(|v| v.to_str().ok())
        {
            return hash_str(key) % self.inner.workers.len();
        }
        self.inner.rr.fetch_add(1, Ordering::Relaxed) % self.inner.workers.len()
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
    let list: Vec<_> = gw
        .inner
        .workers
        .iter()
        .enumerate()
        .map(|(i, u)| {
            serde_json::json!({
                "url": u,
                "inflight": gw.inner.inflight[i].load(Ordering::Relaxed),
                "hits": gw.inner.hits[i].load(Ordering::Relaxed),
            })
        })
        .collect();
    Json(serde_json::json!({ "workers": list }))
}

async fn proxy(State(gw): State<Gateway>, req: Request) -> Response {
    let method = req.method().clone();
    let headers = req.headers().clone();
    let uri = req.uri().clone();
    let path_q = path_and_query(&uri);
    let idx = gw.pick_idx(&headers);
    let worker = gw.inner.workers[idx].clone();
    gw.inner.inflight[idx].fetch_add(1, Ordering::Relaxed);
    gw.inner.hits[idx].fetch_add(1, Ordering::Relaxed);
    let url = format!("{worker}{path_q}");
    let _guard = InflightGuard { inner: gw.inner.clone(), idx };

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
    let stream = GuardedStream { inner: resp.bytes_stream(), _guard: _guard };
    let body = Body::from_stream(stream);
    out.body(body).unwrap_or_else(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
    })
}

fn path_and_query(uri: &Uri) -> String {
    uri.path_and_query().map(|p| p.as_str().to_string()).unwrap_or_else(|| uri.path().to_string())
}

struct GuardedStream<S> {
    inner: S,
    _guard: InflightGuard,
}
impl<S: Stream + Unpin> Stream for GuardedStream<S> {
    type Item = S::Item;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

struct InflightGuard {
    inner: Arc<Inner>,
    idx: usize,
}
impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.inner.inflight[self.idx].fetch_sub(1, Ordering::Relaxed);
    }
}

fn skip_hop(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "host" | "connection" | "keep-alive" | "proxy-connection" | "transfer-encoding" | "upgrade" | "te" | "trailer"
    )
}
