use crate::BPEModel;

#[test]
fn test_special_tokens_present() {
    let model = BPEModel::new();
    assert!(model.vocab.contains_key("<unk>"));
    assert!(model.vocab.contains_key("<pad>"));
    assert!(model.vocab.contains_key("<sos>"));
    assert!(model.vocab.contains_key("<eos>"));
    assert!(model.vocab.contains_key("<space>"));
}

#[test]
fn test_tokenize_word_with_merge() {
    let mut model = BPEModel::new();
    model.merges.push(("a".to_string(), "b".to_string()));
    model.vocab.insert("a".to_string(), 100);
    model.vocab.insert("b".to_string(), 101);
    model.vocab.insert("ab".to_string(), 102);

    let tokens = model.tokenize_word("ab");
    assert_eq!(tokens, vec!["ab"]);
}

#[test]
fn test_encode_decode_roundtrip_simple() {
    let mut model = BPEModel::new();
    model.vocab.insert("h".to_string(), 10);
    model.vocab.insert("i".to_string(), 11);

    let ids = model.encode("hi hi");
    let decoded = model.decode(&ids);
    assert_eq!(decoded, "hi hi");
}

#[test]
fn test_save_and_load_model() {
    let mut model = BPEModel::new();
    model.vocab.insert("a".to_string(), 10);
    model.vocab.insert("b".to_string(), 11);
    model.vocab.insert("ab".to_string(), 12);
    model.merges.push(("a".to_string(), "b".to_string()));

    let tmp = std::env::temp_dir().join("subword_tokenizer_test_model.json");
    let path = tmp.to_string_lossy().to_string();

    model.save(&path).expect("save should succeed");

    let mut loaded = BPEModel::new();
    let loaded_ok = loaded.load(&path).expect("load should succeed");
    assert!(loaded_ok);
    assert!(loaded.vocab.contains_key("ab"));
    assert!(loaded
        .merges
        .iter()
        .any(|(a, b)| a == "a" && b == "b"));

    let _ = std::fs::remove_file(tmp);
}
