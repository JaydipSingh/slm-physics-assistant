use std::collections::HashMap;
use std::cell::RefCell;
use std::fs;

fn escape_json(s: &str) -> String {
    s.chars().map(|c| match c {
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        c if c.is_control() => format!("\\u{:04x}", c as u32),
        c => c.to_string(),
    }).collect()
}

#[derive(Debug, Clone)]
pub struct BPEModel {
    pub vocab: HashMap<String, i32>,
    pub merges: Vec<(String, String)>,
    pub special_tokens: Vec<String>,
    pub id_to_token: HashMap<i32, String>,
    token_cache: RefCell<HashMap<String, Vec<String>>>,
    next_id: i32,
}

impl BPEModel {
    pub fn new() -> Self {
        let mut m = BPEModel {
            vocab: HashMap::new(),
            merges: Vec::new(),
            special_tokens: vec![
                "<unk>".to_string(), "<pad>".to_string(),
                "<sos>".to_string(), "<eos>".to_string(), "<space>".to_string(),
            ],
            id_to_token: HashMap::new(),
            token_cache: RefCell::new(HashMap::new()),
            next_id: 0,
        };
        m.init_special();
        m
    }

    fn init_special(&mut self) {
        for s in &self.special_tokens {
            if !self.vocab.contains_key(s) {
                self.vocab.insert(s.clone(), self.next_id);
                self.next_id += 1;
            }
        }
        self.rebuild_id_to_token();
    }

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut items: Vec<_> = self.vocab.iter().collect();
        items.sort_by_key(|(_, id)| *id);

        let mut out = String::from("{\n");
        out.push_str("  \"vocab\": [\n");
        for (i, (token, _)) in items.iter().enumerate() {
            if i > 0 { out.push_str(",\n"); }
            out.push_str(&format!("    \"{}\"", escape_json(token)));
        }
        out.push_str("\n  ],\n");

        out.push_str("  \"merges\": [\n");
        for (i, (a, b)) in self.merges.iter().enumerate() {
            if i > 0 { out.push_str(",\n"); }
            out.push_str(&format!("    [\"{}\", \"{}\"]", escape_json(a), escape_json(b)));
        }
        out.push_str("\n  ],\n");

        out.push_str("  \"token_to_id\": {\n");
        for (i, (token, id)) in items.iter().enumerate() {
            if i > 0 { out.push_str(",\n"); }
            out.push_str(&format!("    \"{}\": {}", escape_json(token), id));
        }
        out.push_str("\n  },\n");

        out.push_str("  \"id_to_token\": {\n");
        for (i, (token, id)) in items.iter().enumerate() {
            if i > 0 { out.push_str(",\n"); }
            out.push_str(&format!("    \"{}\": \"{}\"", id, escape_json(token)));
        }
        out.push_str("\n  },\n");

        out.push_str("  \"special_tokens\": [\"<unk>\", \"<pad>\", \"<sos>\", \"<eos>\", \"<space>\"]\n");
        out.push_str("}");

        fs::write(path, out)?;
        println!("Saved to {} | Vocab: {} | Merges: {}", path, self.vocab.len(), self.merges.len());
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<bool> {
        if fs::metadata(path).is_err() {
            println!("No existing model, starting fresh");
            return Ok(false);
        }

        let content = fs::read_to_string(path)?;
        
        // Parse JSON properly using simple state machine
        let mut in_vocab = false;
        let mut in_merges = false;
        let mut merge_buf: Vec<String> = Vec::new();
        
        for line in content.lines() {
            let t = line.trim();
            
            if t.starts_with("\"vocab\"") { in_vocab = true; in_merges = false; continue; }
            if t.starts_with("\"merges\"") { in_vocab = false; in_merges = true; continue; }
            if t.starts_with("\"token_to_id\"") { in_vocab = false; in_merges = false; continue; }
            if t.starts_with("\"id_to_token\"") { in_vocab = false; in_merges = false; continue; }
            if t.starts_with("\"special_tokens\"") { in_vocab = false; in_merges = false; continue; }
            
            // Parse vocab items: "token",
            if in_vocab && t.starts_with("\"") && t.ends_with("\",") {
                let token = extract_quoted(t);
                if !token.is_empty() && !self.vocab.contains_key(&token) {
                    self.vocab.insert(token, self.next_id);
                    self.next_id += 1;
                }
            }
            // Last vocab item: "token"
            else if in_vocab && t.starts_with("\"") && t.ends_with("\"") && !t.ends_with("\",") {
                let token = extract_quoted(t);
                if !token.is_empty() && !self.vocab.contains_key(&token) {
                    self.vocab.insert(token, self.next_id);
                    self.next_id += 1;
                }
            }
            
            // Parse merges: multi-line ["a", "b"],
            else if in_merges {
                if t.starts_with('[') && t.contains('"') {
                    if let Some((a, b)) = parse_inline_merge_pair(t) {
                        self.merges.push((a, b));
                    }
                } else if t == "[" {
                    merge_buf.clear();
                } else if t.starts_with('"') {
                    let token = extract_quoted(t);
                    merge_buf.push(token);
                } else if t.starts_with(']') {
                    if merge_buf.len() == 2 {
                        self.merges.push((merge_buf[0].clone(), merge_buf[1].clone()));
                    }
                    merge_buf.clear();
                }
            }
        }

        println!("Loaded: vocab={} merges={}", self.vocab.len(), self.merges.len());
        Ok(true)
    }

    fn rebuild_id_to_token(&mut self) {
        self.id_to_token = self.vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
    }

    pub fn train(&mut self, corpus_path: &str, target_vocab_size: i32) {
        let start = std::time::Instant::now();
        let text = fs::read_to_string(corpus_path).unwrap_or_default();

        let mut word_freq: HashMap<String, i32> = HashMap::new();
        let mut cur = String::new();
        for c in text.chars() {
            if c == ' ' || c == '\n' || c == '\t' {
                if !cur.is_empty() { *word_freq.entry(cur.clone()).or_insert(0) += 1; cur.clear(); }
            } else { cur.push(c); }
        }
        if !cur.is_empty() { *word_freq.entry(cur).or_insert(0) += 1; }

        println!("Corpus: {} chars | Unique words: {}", text.len(), word_freq.len());

        let mut word_tokens: HashMap<String, Vec<String>> = HashMap::new();
        for (word, _) in &word_freq {
            let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            for c in &chars {
                if !self.vocab.contains_key(c) {
                    self.vocab.insert(c.clone(), self.next_id);
                    self.next_id += 1;
                }
            }
            word_tokens.insert(word.clone(), chars);
        }
        self.init_special();

        let target = target_vocab_size - self.vocab.len() as i32;
        if target <= 0 { println!("Vocab already at target!"); return; }

        println!("Current vocab: {} | Target merges: {}", self.vocab.len(), target);

        for i in 0..target {
            let mut pair_counts: HashMap<(String, String), i32> = HashMap::new();
            for (word, tokens) in &word_tokens {
                let freq = word_freq[word];
                for j in 0..tokens.len().saturating_sub(1) {
                    let pair = (tokens[j].clone(), tokens[j + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += freq;
                }
            }
            if pair_counts.is_empty() { break; }
            let best = pair_counts.iter().max_by_key(|(_, v)| *v).unwrap();
            if *best.1 <= 1 {
                println!("Early stop at {} (best pair only appears once)", i);
                break;
            }
            let (first, second) = best.0.clone();
            let merged = format!("{}{}", first, second);
            for (_, tokens) in word_tokens.iter_mut() {
                let mut new = Vec::new();
                let mut j = 0;
                while j < tokens.len() {
                    if j + 1 < tokens.len() && tokens[j] == first && tokens[j + 1] == second {
                        new.push(merged.clone()); j += 2;
                    } else { new.push(tokens[j].clone()); j += 1; }
                }
                *tokens = new;
            }
            self.merges.push((first.clone(), second.clone()));
            if !self.vocab.contains_key(&merged) {
                self.vocab.insert(merged, self.next_id);
                self.next_id += 1;
            }
            if (i + 1) % 1000 == 0 {
                let sec = start.elapsed().as_secs();
                println!("  Merge {}/{} | Vocab: {} | Time: {}s | Best: {},{} ({}x)",
                    i + 1, target, self.vocab.len(), sec, first, second, best.1);
            }
        }
        let sec = start.elapsed().as_secs();
        println!("\nDone! Vocab={} | Merges={} | Time={}s", self.vocab.len(), self.merges.len(), sec);
        self.rebuild_id_to_token();
    }

    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        // Fast path: check cache first
        if let Some(cached) = self.token_cache.borrow().get(word) {
            return cached.clone();
        }
        // Slow path: full BPE merge loop
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        for (first, second) in &self.merges {
            let merged = format!("{}{}", first, second);
            let mut new = Vec::new();
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == *first && tokens[i + 1] == *second {
                    new.push(merged.clone()); i += 2;
                } else { new.push(tokens[i].clone()); i += 1; }
            }
            tokens = new;
        }
        tokens
    }

    pub fn encode(&self, text: &str) -> Vec<i32> {
        let mut ids = vec![self.vocab["<sos>"]];
        let mut cur = String::new();
        for c in text.chars() {
            if c == ' ' || c == '\n' || c == '\t' {
                if !cur.is_empty() {
                    for t in self.tokenize_word(&cur) {
                        ids.push(*self.vocab.get(&t).unwrap_or(&self.vocab["<unk>"]));
                    }
                    cur.clear();
                }
                ids.push(self.vocab["<space>"]);
            } else { cur.push(c); }
        }
        if !cur.is_empty() {
            for t in self.tokenize_word(&cur) {
                ids.push(*self.vocab.get(&t).unwrap_or(&self.vocab["<unk>"]));
            }
        }
        ids.push(self.vocab["<eos>"]);
        ids
    }

    pub fn decode(&self, ids: &[i32]) -> String {
        let space_id = self.vocab["<space>"];
        let sos_id = self.vocab["<sos>"];
        let eos_id = self.vocab["<eos>"];
        let mut out = String::new();
        for &id in ids {
            if id == sos_id || id == eos_id { continue; }
            if id == space_id { out.push(' '); continue; }
            if let Some(t) = self.id_to_token.get(&id) {
                out.push_str(t);
            } else if let Some((token, _)) = self.vocab.iter().find(|(_, token_id)| **token_id == id) {
                out.push_str(token);
            } else {
                out.push_str("<unk>");
            }
        }
        out
    }

    pub fn vocab_size(&self) -> usize { self.vocab.len() }
    pub fn merge_count(&self) -> usize { self.merges.len() }
}

// Helper: extract content between outermost quotes, handling escapes
fn extract_quoted(s: &str) -> String {
    let trimmed = s.trim();
    if !trimmed.starts_with('"') { return String::new(); }
    
    let mut result = String::new();
    let mut chars = trimmed[1..].chars();
    while let Some(c) = chars.next() {
        if c == '"' { break; } // closing quote
        if c == '\\' {
            if let Some(next) = chars.next() {
                match next {
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    '\\' => result.push('\\'),
                    '"' => result.push('"'),
                    'u' => {
                        // \uXXXX
                        let hex: String = chars.by_ref().take(4).collect();
                        if let Ok(code) = u32::from_str_radix(&hex, 16) {
                            if let Some(ch) = char::from_u32(code) {
                                result.push(ch);
                            }
                        }
                    }
                    _ => result.push(next),
                }
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn parse_inline_merge_pair(line: &str) -> Option<(String, String)> {
    let mut parts: Vec<String> = Vec::new();
    let mut in_quote = false;
    let mut escaped = false;
    let mut current = String::new();

    for c in line.chars() {
        if !in_quote {
            if c == '"' {
                in_quote = true;
                current.clear();
            }
            continue;
        }

        if escaped {
            match c {
                'n' => current.push('\n'),
                'r' => current.push('\r'),
                't' => current.push('\t'),
                '\\' => current.push('\\'),
                '"' => current.push('"'),
                _ => current.push(c),
            }
            escaped = false;
            continue;
        }

        if c == '\\' {
            escaped = true;
            continue;
        }

        if c == '"' {
            in_quote = false;
            parts.push(current.clone());
            if parts.len() == 2 {
                break;
            }
            continue;
        }

        current.push(c);
    }

    if parts.len() == 2 {
        Some((parts[0].clone(), parts[1].clone()))
    } else {
        None
    }
}
pub mod data;

#[cfg(test)]
mod tests;
