# Controlled 50,000-Sample Generator Baseline

This folder contains the configuration and outcome tables for a token 3-gram SMILES baseline run using the same requested sample count and source configuration as the production generation experiment.

The baseline emitted 50,000 records, of which 915 were RDKit-valid, 535 were unique valid molecules, and 163 passed the chemical filters. The production Transformer/SELFIES run produced 49,660 valid strings and 40,477 filter-pass unique molecules. This supports an engineering claim about valid-yield efficiency under these configurations; it is not an external benchmark of RNA binding, SMN2 relevance, or therapeutic performance.
