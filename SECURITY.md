# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| latest  | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in **Coleman**, please report it
responsibly.

**Do not open a public GitHub issue.**

Instead, please send an email to
[jacksonpradolima@gmail.com](mailto:jacksonpradolima@gmail.com) with:

- A description of the vulnerability.
- Steps to reproduce the issue.
- Any potential impact or severity assessment.

We will acknowledge receipt within **48 hours** and aim to provide a fix or
mitigation plan within **7 days** of confirmation.

## Disclosure Policy

- We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure).
- Security fixes will be released as patch versions.
- A security advisory will be published on GitHub once the fix is available.

## Scope

This policy applies to the `coleman` Python package and its dependencies as
distributed through this repository. Third-party dependencies are managed via
`uv` and pinned in `uv.lock`.

### New dependencies

The experiment system adds two new direct dependencies:

| Dependency | Purpose | Advisory status |
|-----------|---------|-----------------|
| `pydantic>=2.12.5` | Typed run-spec models & validation | No known advisories |
| `pyyaml>=6.0.3` | YAML config loading (safe_load only) | No known advisories |

All YAML loading uses `yaml.safe_load()` — untrusted YAML with arbitrary
Python constructors is never evaluated.

Thank you for helping keep Coleman safe.
