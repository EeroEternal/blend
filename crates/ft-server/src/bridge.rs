//! server(tokio) ↔ engine(OS 线程) 的消息桥。
//!
//! engine 当前是 StubModel（逐词吐 token），P2 替换为真实 GPU 引擎；
//! 协议不变，server 侧零改动——这就是分层的目的。

use flume::{Receiver, Sender};
use ft_core::SeqId;
use std::time::Duration;

#[derive(Debug)]
pub struct ChatJob {
    pub prompt_words: Vec<String>,
    pub max_tokens: usize,
    pub resp: Sender<EngineEvent>,
}

#[derive(Debug)]
pub enum EngineEvent {
    Token { seq: SeqId, text: String },
    Done { seq: SeqId },
}

/// engine 句柄：tokio 侧只持有 sender。
#[derive(Clone)]
pub struct EngineHandle {
    pub tx: Sender<ChatJob>,
    pub model_name: String,
}

/// 启动 engine 线程（OS 线程，非 tokio 任务）。
pub fn spawn_engine(model_name: impl Into<String>) -> EngineHandle {
    let (tx, rx): (Sender<ChatJob>, Receiver<ChatJob>) = flume::unbounded();
    let model_name = model_name.into();
    std::thread::Builder::new()
        .name("ft-engine".into())
        .spawn(move || engine_loop(&rx))
        .expect("spawn engine thread");
    EngineHandle { tx, model_name }
}

fn engine_loop(rx: &Receiver<ChatJob>) {
    let mut next_seq = 1u64;
    while let Ok(job) = rx.recv() {
        let seq = SeqId(next_seq);
        next_seq += 1;
        // stub 推理：把 prompt 词回显一遍，模拟逐 token 解码
        let out: Vec<String> = job
            .prompt_words
            .iter()
            .take(job.max_tokens)
            .cloned()
            .chain(std::iter::repeat("[done]".into()))
            .take(job.max_tokens)
            .collect();
        for w in out {
            std::thread::sleep(Duration::from_millis(5));
            if job.resp.send(EngineEvent::Token { seq, text: w }).is_err() {
                break;
            }
        }
        let _ = job.resp.send(EngineEvent::Done { seq });
    }
}
