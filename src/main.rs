use std::env;
use std::path::Path;
use subword_tokenizer::BPEModel;
use subword_tokenizer::data;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage:");
        println!("  cargo run --release -- train <corpus.txt> <vocab_size>");
        println!("  cargo run --release -- expand <corpus.txt> <vocab_size>");
        println!("  cargo run --release -- tokenize <text>");
        println!("  cargo run --release -- decode <ids>");
        println!("  cargo run --release -- count");
        println!("  cargo run --release -- prepare <input.txt> --train <t.bin> --val <v.bin> --test <x.bin> [--split a,b,c] [--eos id]");
        return;
    }

    let mut bpe = BPEModel::new();
    let model_loaded = bpe.load("model.json").is_ok();

    match args[1].as_str() {
        "train" => {
            let corpus = &args[2];
            let size: i32 = args[3].parse().unwrap();
            let mut bpe = BPEModel::new();
            bpe.train(corpus, size);
            bpe.save("model.json").unwrap();
        }
        "expand" => {
            let corpus = &args[2];
            let size: i32 = args[3].parse().unwrap();
            let mut bpe = BPEModel::new();
            let _ = bpe.load("model.json");
            bpe.train(corpus, size);
            bpe.save("model.json").unwrap();
        }
        "tokenize" => {
            let text = &args[2];
            if model_loaded == false { panic!("model.json not found"); }
            let ids = bpe.encode(text);
            println!("IDs: {:?}", ids);
            println!("Count: {}", ids.len());
        }
        "decode" => {
            let ids_str = &args[2];
            if model_loaded == false { panic!("model.json not found"); }
            let ids: Vec<i32> = ids_str.split(',').map(|s| s.trim().parse().unwrap()).collect();
            let text = bpe.decode(&ids);
            println!("Decoded: {}", text);
        }
        "count" => {
            if model_loaded == false { panic!("model.json not found"); }
            println!("Vocab: {} | Merges: {} | Total: {}",
                bpe.vocab_size(), bpe.merge_count(), bpe.vocab_size() + 5);
        }
        "prepare" => {
            if model_loaded == false { panic!("model.json not found"); }
            let input = Path::new(&args[2]);
            let train_str = get_flag(&args, "--train").expect("Missing --train");
            let train = Path::new(&train_str);
            let val_str = get_flag(&args, "--val").expect("Missing --val");
            let val = Path::new(&val_str);
            let test_str = get_flag(&args, "--test").expect("Missing --test");
            let test = Path::new(&test_str);

            let split_str = get_flag(&args, "--split").unwrap_or("0.98,0.01,0.01".to_string());
            let parts: Vec<f64> = split_str.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let eos: i32 = get_flag(&args, "--eos").unwrap_or("3".to_string()).parse().unwrap_or(3);

            let (t, v, te) = data::prepare_dataset(
                |s| bpe.encode(s),
                input, train, val, test,
                (parts[0], parts[1], parts[2]), eos,
            ).unwrap();

            println!("Train: {} tokens", t);
            println!("Val:   {} tokens", v);
            println!("Test:  {} tokens", te);
        }
        _ => println!("Unknown: {}", args[1]),
    }
}

fn get_flag(args: &[String], flag: &str) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        if arg == flag {
            return args.get(i + 1).cloned();
        }
    }
    None
}
