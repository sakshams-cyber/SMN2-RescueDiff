# Cross-Target 40 x 20 Release

This folder releases the previously missing 800-job non-RNA cross-target campaign: 40 molecules by 20 receptors, with score/status rows, receptor inputs, Vina logs, poses, panel provenance, and the run configuration.

All 800 jobs are recorded as completed real Vina calculations. The table is an engineering-completion and exploratory promiscuity audit. Vina scores from heterogeneous protein receptors and the SMN2 RNA receptor are not calibrated as commensurate binding free energies, so this campaign is not independent evidence of RNA selectivity or safety.

The `run/` folder contains the score table, 800 job logs, and 800 poses. `non_rna_receptor_panel_20.tsv`, `panel_provenance.json`, and `receptors/` define the panel. `config_used.yaml` and `run.log` preserve execution settings and status.
