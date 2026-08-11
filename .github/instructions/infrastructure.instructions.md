---
applyTo: "src/ragkit/infrastructure/**"
---

Treat `config.py` as the strict secret-free schema and `bootstrap.py` as the
composition root. Add selections in both places, fail unknown or unavailable
capabilities explicitly, and resolve credentials only while composing adapters.
