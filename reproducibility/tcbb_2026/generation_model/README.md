# Production Generator Artifact

This folder contains the production Transformer checkpoint, vocabulary, tokenizer metadata, run configuration, source-balance table, training log, and training-dataset summary used for the 50,000-sample generation experiment.

The generator is unconditional and autoregressive; it is not a diffusion model and is not conditioned on SMN2. SMN2 information enters downstream docking and ranking. R-BIND/SMMRNA contributed zero rows to this historical training run, and known splicing modifiers were not used as training examples.

The complete ChEMBL/ZINC-derived training corpus is not redistributed here because exact upstream release identifiers, retrieval dates, and redistribution licenses could not be reconstructed with sufficient confidence. The retained hashes and source accounting support provenance auditing, but this remains an end-to-end reproduction limitation.
