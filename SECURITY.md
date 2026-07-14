# Security Policy

## Reporting a vulnerability

Please report security vulnerabilities privately. **Do not open a public issue.**

Use GitHub's private vulnerability reporting for this repository:
**Security → Report a vulnerability** (the "Report a vulnerability" button on the
Security tab). This opens a private channel with the maintainers.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce, or a proof of concept.
- The version or commit you tested.

We will acknowledge your report, work with you to understand and resolve the
issue, and credit you when a fix ships, unless you prefer to stay anonymous.

## Scope

TheUnduster is a local desktop application. It processes images entirely on the
user's machine and does not upload scans or send telemetry. The main areas of
security interest are:

- **Model download and verification.** The healing model is downloaded on first
  use from a pinned revision and verified against a pinned SHA-256 before it is
  used (`app/src-tauri/src/models.rs`). Reports about bypassing that
  verification, or about the download endpoint, are in scope.
- **File handling.** Decoding untrusted image files (`fd-io`) and reading/writing
  per-roll working state (`app/src-tauri/src/roll.rs`, `cache.rs`).
- **Tauri command surface.** The commands the frontend can invoke on the Rust
  backend.

## Supported versions

This project is under active development and does not yet publish tagged
releases. Security fixes are applied to the `main` branch.
