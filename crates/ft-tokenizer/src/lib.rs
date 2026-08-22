//! 分词层。trait + 朴素空白实现；HF tokenizers 在 `hf` feature 下启用。

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
    fn vocab_size(&self) -> usize;
}

/// 朴素空白分词：仅用于测试/冒烟，token = 单词的 FNV 截断。
pub struct WhitespaceTokenizer {
    vocab: std::collections::HashMap<String, u32>,
    inv: Vec<String>,
}

impl WhitespaceTokenizer {
    pub fn new() -> Self {
        Self { vocab: std::collections::HashMap::new(), inv: vec![] }
    }
    fn intern(&mut self, w: &str) -> u32 {
        if let Some(&id) = self.vocab.get(w) {
            return id;
        }
        let id = self.inv.len() as u32;
        self.vocab.insert(w.to_string(), id);
        self.inv.push(w.to_string());
        id
    }
}

impl Default for WhitespaceTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer for WhitespaceTokenizer {
    fn encode(&self, _text: &str) -> Vec<u32> {
        // &self 无法 intern；encode 语义上只查表。测试路径先手动喂词。
        _text.split_whitespace().filter_map(|w| self.vocab.get(w).copied()).collect()
    }
    fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| self.inv.get(t as usize))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ")
    }
    fn vocab_size(&self) -> usize {
        self.inv.len()
    }
}

impl WhitespaceTokenizer {
    /// 测试辅助：注册词表。
    pub fn with_words(words: &[&str]) -> Self {
        let mut t = Self::new();
        for w in words {
            t.intern(w);
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let t = WhitespaceTokenizer::with_words(&["hello", "world"]);
        let ids = t.encode("hello world");
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(t.decode(&ids), "hello world");
        assert_eq!(t.vocab_size(), 2);
    }

    #[test]
    fn unknown_words_dropped() {
        let t = WhitespaceTokenizer::with_words(&["hello"]);
        assert_eq!(t.encode("hello xyz"), vec![0]);
    }
}
