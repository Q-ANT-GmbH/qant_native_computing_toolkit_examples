# Contributing to the Q.ANT Native Computing Toolkit Examples

Thanks for your interest in contributing! This repository collects example
code for the Q.ANT Native Computing Toolkit, and we welcome new examples,
fixes, and improvements from the community.

## Ways to contribute

- **New examples** — a new Python or C++ sample showcasing a workflow,
  pattern, or use case not yet covered.
- **Bug fixes** — corrections to existing examples, build scripts, or docs.
- **Improvements** — clearer explanations, better comments, updated
  dependencies, performance tweaks.

If you're planning a larger addition (e.g. a new example category), please
open an issue first to discuss the idea before investing significant time.

## Getting started

1. Fork the repository and create a feature branch off `latest`.
2. Install the required Q.ANT Native Computing Toolkit interface for the
   language you're working in (see the [README](README.md)).
3. Install [pre-commit](https://pre-commit.com) and set up the hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Adding a new example

New examples from outside Q.ANT go into the **community zoo** under
`community/python/<example_name>/` or `community/cpp/<example_name>/`. The
official `python/` and `cpp/` folders are maintained by Q.ANT. Fixes and
improvements to official examples are still welcome. Community examples
follow the same structure as the official ones:

- **Python examples**
  - Self-contained directory under `python/<example_name>/`.
  - A `requirements.txt` listing all dependencies needed to run it.
  - A `README.md` describing what the example does, prerequisites, and how
    to run it (see `python/digit_recognition/README.md` for reference).
  - A test (e.g. `test_nb.py` for notebooks) that can run in CI without
    requiring NPU hardware, if at all possible.

- **C++ examples**
  - Self-contained directory under `cpp/<example_name>/`.
  - A `CMakeLists.txt` and a `run_example.sh` (or `run_examples.sh`) script
    that builds and runs the example end-to-end.
  - A `README.md` describing prerequisites and usage (see
    `cpp/mul_elementwise/README.md` for reference).
  - Code formatted according to the repository's `.clang-format` style.

Please keep examples focused and minimal, favoring clarity over cleverness,
and only add dependencies needed to demonstrate the concept.

## Community examples

Community examples are written and owned by their authors, **not maintained by
Q.ANT**. See [community/README.md](community/README.md) for the disclaimer.
In addition to the structure above, a community example must:

- Include an `AUTHORS.md` with the author name(s) and GitHub handle(s) (see
  `community/python/small_lm/AUTHORS.md`).
- Start its `README.md` with the community banner (see
  `community/python/small_lm/README.md`).
- Run on the **CPU backend without NPU hardware**, with a test that finishes
  in under ~10 minutes.
- Pin heavy dependencies (e.g. `torch==2.11.0`).
- Not commit large binaries (keep each example under ~10 MB). Download
  datasets and weights from stable, well-known URLs, or generate them.
- Not use Q.ANT logos or suggest that it is an official Q.ANT example.

**How community examples get merged:** the public repository is published
from an internal repository. After review, a maintainer ports your PR into the
internal repository, and the example ships with the next release. You are
credited as co-author of that commit (`Co-authored-by`) and listed in
`AUTHORS.md`. Your PR is then closed with a link to that release, and not
merged directly.

**Removal policy:** We want every example to work for the people trying it, so community
examples are tested in CI. Since you know your example best, fixes are up to you. If an
example stops working (for example a crash, timeout, broken dependency or incompatible
toolkit version), we take it out of the next release. Once you submit a fix, we'll gladly
add it back.

## Code style

- Python code is linted and formatted with [ruff](https://docs.astral.sh/ruff/)
  via pre-commit; notebooks are stripped of output/metadata with `nbstripout`.
- C++ code must conform to the repository's `.clang-format` style
  (`clang-format --dry-run --Werror` is enforced in CI).
- Run `pre-commit run --all-files` before committing to catch formatting
  issues locally.

## Testing

- CI (see `.github/workflows/workflow.yml`) installs the toolkit with the
  CPU backend and runs the tests/scripts of each official example.
- Community examples are discovered automatically by
  `.github/workflows/workflow_community.yml`, so you don't need to edit any
  workflow. CI does not run automatically on pull requests from forks.
  Please run the tests locally before opening your PR.
- Python examples should include a `pytest`-compatible test.
- C++ examples should include a `run_example(s).sh` script that builds and
  runs successfully without manual intervention.

## Submitting your changes

1. Before opening your PR, please run `pre-commit run --all-files`, the
   relevant `pytest` suite(s), and any C++ `run_example(s).sh` scripts
   locally.
2. Commit your changes with a clear, descriptive commit message.
3. Open a pull request against `latest`, describing what the change does and why.
4. A maintainer will review your PR and may suggest changes before merging.

## License

By contributing, you agree that your contributions will be licensed under
the [Apache License 2.0](LICENSE), the same license as this project.

## Questions?

If you're unsure about anything, feel free to open an issue or reach out to
the maintainers before submitting a pull request.
