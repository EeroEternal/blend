//! OpenAI 兼容 API（/v1/models、/v1/chat/completions）+ 健康检查。

use crate::bridge::{EngineEvent, EngineHandle, ChatJob};
use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_core::Stream;
use ft_core::ChatRequest;
use std::{
    convert::Infallible,
    time::Duration,
};
use tokio::sync::mpsc;

pub fn router(engine: EngineHandle) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn models(
    State(e): State<EngineHandle>,
) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": e.model_name,
            "object": "model",
            "owned_by": "blend",
        }]
    }))
}

async fn chat_completions(
    State(e): State<EngineHandle>,
    Json(req): Json<ChatRequest>,
) -> Result<axum::response::Response, StatusCode> {
    // 拼接所有 message 文本，按空白切词喂给 stub 引擎
    let text: String =
        req.messages.iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join(" ");
    let words: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
    if words.is_empty() {
        return Err(StatusCode::BAD_REQUEST);
    }
    let max_tokens = req.effective_max_tokens(64);
    let (tx, rx) = flume::unbounded();
    e.tx
        .send(ChatJob { prompt_words: words, max_tokens, resp: tx })
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    if req.stream {
        let stream = token_stream(rx);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        // 收齐全部 token 再拼完整回复
        let mut parts = Vec::new();
        while let Ok(ev) = rx.recv_async().await {
            match ev {
                EngineEvent::Token { text, .. } => parts.push(text),
                EngineEvent::Done { .. } => break,
            }
        }
        Ok(Json(serde_json::json!({
            "id": "chatcmpl-blend",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": parts.join(" ")},
                "finish_reason": "stop"
            }],
            "usage": {
                "completion_tokens": parts.len(),
                "total_tokens": parts.len()
            }
        }))
        .into_response())
    }
}

fn token_stream(
    rx: flume::Receiver<EngineEvent>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream_wrap(rx)
}

// 用 async stream 组合器最小实现（避免引入 async-stream 宏依赖）
fn async_stream_wrap(
    rx: flume::Receiver<EngineEvent>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    let (mtx, mrx) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        while let Ok(ev) = rx.recv_async().await {
            let ev = match ev {
                EngineEvent::Token { text, .. } => Event::default().data(text),
                EngineEvent::Done { .. } => Event::default().data("[DONE]"),
            };
            if mtx.send(Ok(ev)).await.is_err() {
                break;
            }
        }
    });
    tokio_stream::wrappers::ReceiverStream::new(mrx)
}
