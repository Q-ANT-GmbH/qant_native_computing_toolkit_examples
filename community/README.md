# Community Examples

This folder contains examples for the Q.ANT Native Computing Toolkit that were **contributed by
the community**. They show what people outside Q.ANT have built with the toolkit.

> [!WARNING]
> **Disclaimer:** The examples in this folder were written by third parties, not by Q.ANT.
> - Q.ANT does **not** maintain, support or endorse them. Their correctness, accuracy and
>   performance have not been verified by Q.ANT.
> - They are provided "AS IS", without warranties or conditions of any kind, under the
>   [Apache License 2.0](../LICENSE).
> - Questions about a community example should go to its authors (see its `AUTHORS.md`).
>   Q.ANT cannot provide support for them.

## Removal policy
We want everyone who tries an example from this repository to have a good experience, so
all community examples are tested in CI. Since the authors know their examples best, we
leave fixes to them. If an example stops working, for example after a toolkit update, we
take it out of the next release so that users only see examples that work. It isn't gone
for good: once a fixed version is submitted, we'll gladly add it back.

## Contributing a community example
See the "Community examples" section in [CONTRIBUTING.md](../CONTRIBUTING.md).
In short:

- Put your example in `community/python/<example_name>/` or `community/cpp/<example_name>/`.
  Use the same structure as the official examples.
- Add an `AUTHORS.md` file and the community banner at the top of your example's `README.md`.
- The example must run on the CPU backend, without NPU hardware, and must come with a test
  (`pytest` for Python, `run_example.sh` for C++).
