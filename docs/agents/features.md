# Adding an extension

1. `src/boo_sd_scripts/<ext>/__init__.py` plus the modules. Import `boo_sd_scripts._sd_scripts` before any sd-scripts import.
2. Keep pure math in its own module so it can be tested without a GPU or model.
3. `tests/<ext>/` with unit tests; use `library.moss_image_models.MossImage` with a small config for CPU model tests.
4. `docs/<ext>.md`: flags, an example command, and any known deviation from the source paper.
5. Add the extension to the README list.
6. `uv run ruff check . && uv run ruff format . && uv run pytest` before committing.
