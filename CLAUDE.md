# CLAUDE.md

Training extensions built on [sd-scripts](https://github.com/kohya-ss/sd-scripts), vendored as a git submodule at
`sd-scripts/` and imported through `boo_sd_scripts._sd_scripts` (sd-scripts is not pip-installable). Each extension
subclasses an sd-scripts trainer (e.g. `MossImageNetworkTrainer`) and overrides its extension seams
(`get_noise_pred_and_target`, `process_batch`, `post_process_loss`, ...). Never edit files inside `sd-scripts/`.

See README.md for the extension list and docs/agents/features.md for the add-an-extension checklist.
`docs/superpowers/` is git-ignored: never commit it.
