# boo-sd-scripts

Training extensions for [sd-scripts](https://github.com/kohya-ss/sd-scripts), vendored as a git submodule.

## Extensions

- [TDM](docs/tdm.md): Trajectory Distribution Matching, distills MOSS-Image into a low-step, guidance-distilled LoRA

## Setup

```bash
git clone --recurse-submodules <this repo>
cd boo-sd-scripts
uv sync --extra cu130   # or cpu / cu124 / cu128 / cu132
```

Set `SD_SCRIPTS_PATH` to use a different sd-scripts checkout instead of the submodule.

## Development

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```
