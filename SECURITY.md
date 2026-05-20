# Security Policy

viva_math is a scientific computing library for Gleam on the BEAM. Security
reports are handled with the same care as numerical correctness reports: clear
reproduction, coordinated disclosure, and minimal public exposure before a fix
is available.

## Supported Versions

Only the latest release in the `1.x` series is currently supported for security
updates.

| Version | Supported |
| ------- | --------- |
| 1.x latest | Yes |
| Older 1.x releases | No |
| < 1.0 | No |

## Reporting a Vulnerability

Do not open a public GitHub issue for security reports.

Send a private email to marlon.barreto2378@gmail.com with `[SECURITY]` in the
subject line. Include:

* A concise description of the vulnerability.
* The affected viva_math version.
* A minimal Gleam reproduction when possible.
* Environment details, including Gleam and OTP versions.
* Any known downstream impact on consumers such as `viva_emotion` or
  `viva_tensor`.

The project will acknowledge valid reports within 72 hours and coordinate a fix
before public disclosure.

## Scope

Security reports in scope include:

* Mathematical vulnerabilities, including numerical instability that can cause
  denial of service through excessive runtime, non-termination, memory pressure,
  or pathological BEAM scheduler behavior.
* Erlang FFI failures that can trigger unexpected panics, crashes, or unsafe
  runtime behavior.
* Supply-chain vulnerabilities, including compromised dependencies or build
  artifacts that affect viva_math users.

Reports out of scope include:

* Version warnings for `gleam_community_maths` by themselves, without an
  exploitable vulnerability in viva_math.
* Deterministic failures for invalid inputs when those failures are explicitly
  documented by the relevant API.

## Disclosure Process

After a fix is available, the project follows coordinated disclosure with a
90-day disclosure window. The maintainers may disclose earlier when users are at
active risk, or later when coordination with affected downstream consumers
requires additional time.

Security advisories should include the affected versions, impact, mitigation,
and upgrade guidance. Public discussion should avoid publishing exploit details
before users have had a reasonable opportunity to upgrade.
