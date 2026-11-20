# ft8-ml

Training and evaluation code for machine-learned components of FT8
decoding.

- [`candidate-screening/`](candidate-screening/): gradient-boosted trees
  that rank synchronization candidates before LDPC decoding.
  [FEATURES.md](candidate-screening/FEATURES.md) describes each feature
  column.
- [`neural-llr/`](neural-llr/): PyTorch soft demodulators that estimate
  coded-bit LLRs, with training, proxy-metric evaluation, and ONNX export.

![Candidate screening before LDPC/OSD](candidate-screening/figures/candidate-screening-overview.png)

Datasets, trained models, and decoder integration are not included.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
