use std::collections::{HashMap, HashSet};
use std::fs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).map(|s| s.as_str()).unwrap_or("model.json");
    let output = args.get(2).map(|s| s.as_str()).unwrap_or("model_32k.json");
    let target: usize = args.get(3).map(|s| s.parse().unwrap_or(32000)).unwrap_or(32000);

    println!("Prune: {} -> {} (target={} tokens)", input, output, target);

    let data = fs::read_to_string(input).expect("Cannot read model");
    let model: serde_json::Value = serde_json::from_str(&data).expect("Invalid JSON");

    // Handle BOTH formats: array vocab OR object vocab
    let mut vocab: HashMap<String, i64> = HashMap::new();

    if let Some(vocab_obj) = model["vocab"].as_object() {
        // Format B: {"vocab": {"token": id, ...}}
        for (k, v) in vocab_obj {
            vocab.insert(k.clone(), v.as_i64().unwrap_or(0));
        }
        println!("Detected: object-format vocab");
    } else if let Some(vocab_arr) = model["vocab"].as_array() {
        // Format A: {"vocab": ["token", ...]}
        for (i, v) in vocab_arr.iter().enumerate() {
            if let Some(s) = v.as_str() {
                vocab.insert(s.to_string(), i as i64);
            }
        }
        println!("Detected: array-format vocab");
    }

    let mut merges: Vec<(String, String)> = Vec::new();
    if let Some(merges_arr) = model["merges"].as_array() {
        for m in merges_arr {
            if let Some(pair) = m.as_array() {
                if pair.len() == 2 {
                    let a = pair[0].as_str().unwrap_or("").to_string();
                    let b = pair[1].as_str().unwrap_or("").to_string();
                    merges.push((a, b));
                }
            }
        }
    }

    let special: Vec<String> = model["special_tokens"].as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_else(|| vec![
            "<unk>".to_string(), "<pad>".to_string(),
            "<sos>".to_string(), "<eos>".to_string(), "<space>".to_string(),
        ]);

    let special_count = special.len();

    println!("Loaded: vocab={} merges={} special={}", vocab.len(), merges.len(), special_count);

    if vocab.is_empty() {
        println!("ERROR: Could not parse vocab! Check JSON format.");
        return;
    }

    // Sort vocab by ID (lower = higher frequency)
    let mut sorted: Vec<(String, i64)> = vocab.into_iter().collect();
    sorted.sort_by_key(|(_, id)| *id);

    let target_regular = target - special_count;
    let special_set: HashSet<String> = special.iter().cloned().collect();

    let mut new_vocab = HashMap::new();
    let mut kept = 0;

    for (token, old_id) in sorted {
        if special_set.contains(&token) {
            new_vocab.insert(token, old_id as i32);
        } else if kept < target_regular {
            new_vocab.insert(token, (kept + special_count) as i32);
            kept += 1;
        }
    }

    let kept_tokens: HashSet<String> = new_vocab.keys().cloned().collect();
    let mut new_merges = Vec::new();
    for (a, b) in merges {
        let merged = format!("{}{}", a, b);
        if kept_tokens.contains(&merged) {
            new_merges.push(serde_json::json!([a, b]));
        }
    }

    let max_merges = target_regular.saturating_sub(256);
    if new_merges.len() > max_merges {
        new_merges.truncate(max_merges);
    }

    let mut new_vocab_json = serde_json::Map::new();
    for (k, v) in new_vocab {
        new_vocab_json.insert(k, serde_json::Value::Number(serde_json::Number::from(v)));
    }

    let new_model = serde_json::json!({
        "vocab": new_vocab_json,
        "merges": new_merges,
        "special_tokens": special,
    });

    fs::write(output, serde_json::to_string_pretty(&new_model).unwrap()).expect("Save failed");

    let final_vocab = new_model["vocab"].as_object().unwrap().len();
    let final_merges = new_model["merges"].as_array().unwrap().len();
    println!("Saved: vocab={} merges={} total={}", final_vocab, final_merges, final_vocab + special_count);
}
