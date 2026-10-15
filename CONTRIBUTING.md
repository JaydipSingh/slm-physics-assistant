# Contributing

Thanks for your interest in improving this project. Contributions, bug reports,
and questions are all welcome.

## Reporting bugs and asking questions

- Please open an [issue](https://github.com/JaydipSingh/slm-physics-assistant/issues)
  for bugs, questions, or feature requests.
- For bug reports, include your operating system, Python and Rust versions, the
  command you ran, and the full error output. A minimal example that reproduces
  the problem is extremely helpful.

## Development setup

```bash
# Clone and enter the repository
git clone https://github.com/JaydipSingh/slm-physics-assistant.git
cd slm-physics-assistant

# Python environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Build the Rust tokenizer
cargo build --release
```

## Running the tests

Python tests (pytest):

```bash
python -m pytest tests/ -v
```

Rust tokenizer tests:

```bash
cargo test
```

The Python test suite skips the model tests automatically if optional
dependencies are missing; install `requirements-dev.txt` to run the full suite.

## Making changes

1. Fork the repository and create a feature branch:
   `git checkout -b my-feature`.
2. Make your change, keeping it focused and documented.
3. Add or update tests where it makes sense, and ensure `pytest` and
   `cargo test` pass.
4. Open a pull request describing what you changed and why.

## Code style

- Python: follow PEP 8; keep functions small and readable.
- Rust: run `cargo fmt` before committing.

## Scope

This project is a reproducible, laptop-scale reference implementation for
training domain-specialized small language models. Contributions that improve
reproducibility, documentation, test coverage, or portability to other hardware
and domains are especially welcome.

## License

By contributing, you agree that your contributions will be licensed under the
project's [MIT License](LICENSE).
