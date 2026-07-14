# Contributing to TheUnduster

Thanks for your interest in improving TheUnduster. This project removes dust and
scratches from scanned film, and it gets better when photographers, film
scanners, and developers pitch in. Contributions of all sizes are welcome:
bug reports, documentation fixes, test film scans, and code.

By taking part in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- **Report a bug.** Open an issue with what you did, what you expected, and what
  happened. Include your macOS version and, if you can, a small sample scan that
  reproduces the problem.
- **Suggest a feature.** Open an issue describing the workflow you want. Explain
  the problem before the solution.
- **Improve the docs.** The [user manual](docs/user-manual.md) and this file can
  always be clearer.
- **Help train the detector.** The defect detector is not trained yet — the app
  ships a fixture model, so it won't catch real defects until a model is trained
  on real film. Labelled scans and training work are the biggest open need. See
  [`training/`](training/README.md) and `training/DATA.md`.
- **Write code.** See the setup below.

## Development setup

TheUnduster is a Tauri 2 + Svelte 5 app over a Rust engine, with a separate
Python training pipeline. [Mise](https://mise.jdx.dev/) manages the toolchains;
each part of the repo pins its own versions.

```bash
# App (Node 22, Rust stable)
cd app && npm install && mise exec -- npm run tauri dev
```

See the [README](README.md) for the full repository layout and build commands.

## Before you open a pull request

Run the same checks CI would have run, per area you touched:

```bash
# Rust workspace (from the repo root)
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all --check

# Frontend (from app/)
npm run test
npm run check

# Training (from training/)
uv run pytest
```

Guidelines:

- **One logical change per pull request.** Small, focused changes are easier to
  review and revert.
- **Add a test that fails without your change.** New behavior needs coverage.
- **Write clear commit messages** that explain *why*, not just *what*.
- **Keep it local and private.** TheUnduster processes images on the user's
  machine and never uploads their scans. Please keep it that way: no telemetry,
  no analytics, no remote image processing.

## Reporting security issues

Please do not open a public issue for security problems. See
[SECURITY.md](SECURITY.md) for how to report privately.

## License

By contributing, you agree that your contributions are licensed under the
[GNU General Public License v3.0](LICENSE), the same license as the project.
