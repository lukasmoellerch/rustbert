# rustbert

A pure Rust almost-no-dependency implementation of BERT. Does all of the multiplications itself, no BLAS or anything. Optimized for aarch64 neon, but still slower than torch at the moment.

## Building

You need a nightly rust compiler. Then just run `cargo build --release` and you should be good to go.

Will be slow if not compiled with neon support.

## Running

First, a model has to be converted into a proprietary format. This is done by using the Python script `convert.py`. It takes a BERT model as input and outputs a binary file. This binary file can then be used by the Rust program.

The Rust program takes the binary file as input and outputs timing information at the moment using a hard-coded example. A better API might follow once it's faster.
