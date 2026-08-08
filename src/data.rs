use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use rand::seq::SliceRandom;

pub fn prepare_dataset<E>(
    encode: E,
    input_path: &Path,
    train_path: &Path,
    val_path: &Path,
    test_path: &Path,
    split_ratios: (f64, f64, f64),
    eos_token: i32,
) -> io::Result<(usize, usize, usize)>
where
    E: Fn(&str) -> Vec<i32>,
{
    let total = split_ratios.0 + split_ratios.1 + split_ratios.2;
    assert!((total - 1.0).abs() < 1e-6, "Split ratios must sum to 1.0");

    println!("Reading input file...");
    let text = std::fs::read_to_string(input_path)?;
    println!("File read: {} bytes", text.len());

    let mut docs: Vec<&str> = text.split("\n\n").collect();
    docs.retain(|d| !d.trim().is_empty());
    println!("Found {} documents", docs.len());

    if docs.is_empty() {
        panic!("No documents found");
    }

    println!("Shuffling documents...");
    let mut rng = rand::rng();
    docs.shuffle(&mut rng);

    println!("Tokenizing {} documents...", docs.len());
    let mut all_ids: Vec<i32> = Vec::new();
    for (i, doc) in docs.iter().enumerate() {
        if i % 1000 == 0 && i > 0 {
            println!("  Tokenized {}/{} docs, {} tokens so far", i, docs.len(), all_ids.len());
        }
        let mut ids = encode(doc.trim());
        all_ids.append(&mut ids);
        all_ids.push(eos_token);
    }
    println!("Total tokens: {}", all_ids.len());

    let n = all_ids.len();
    let n_train = (n as f64 * split_ratios.0) as usize;
    let n_val = (n as f64 * split_ratios.1) as usize;

    println!("Writing train.bin ({} tokens)...", n_train);
    write_i32_as_u16_binary_file(train_path, &all_ids[0..n_train])?;
    println!("Writing val.bin ({} tokens)...", n_val);
    write_i32_as_u16_binary_file(val_path, &all_ids[n_train..n_train + n_val])?;
    println!("Writing test.bin ({} tokens)...", n - n_train - n_val);
    write_i32_as_u16_binary_file(test_path, &all_ids[n_train + n_val..])?;

    println!("Done!");
    Ok((n_train, n_val, n - n_train - n_val))
}

fn write_i32_as_u16_binary_file(path: &Path, ids: &[i32]) -> io::Result<()> {
    let mut file = File::create(path)?;
    for id in ids {
        assert!(*id >= 0 && *id <= 65535, "Token ID {} out of u16 range", id);
        file.write_all(&(*id as u16).to_le_bytes())?;
    }
    Ok(())
}
