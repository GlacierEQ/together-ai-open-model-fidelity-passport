# Open-Model Fidelity Passport

Independent GlacierEQ portfolio implementation aligned to **Together AI** operating themes.

> **Not affiliated.** This repository is not affiliated with, endorsed by, employed by, or deployed at Together AI. No proprietary access, production deployment, customer impact, or company partnership is claimed.

## Purpose

Make an open-model serving variant inspectable enough that a reviewer can prove **what weights are running, how they were transformed, what serving defaults changed, which kernels are present, what behavior moved, what hardware is compatible, and whether the promised cost envelope was actually met**.

## Implemented passport

`OpenModelFidelityPassport` binds:

- base-model identity, exact weights digest, parent lineage, revision and license;
- quantization format, bit width, group size and calibration identity;
- sampler defaults;
- ordered canonical kernel stack;
- minimum GPU memory, supported dtypes and architectures;
- workload-by-workload baseline/variant evaluation scores and maximum allowed drop;
- baseline/variant serving cost and maximum permitted cost ratio.

The normalized record receives a deterministic `passport_digest`. Verification fails closed when:

- weights/lineage evidence is malformed;
- evaluation drop exceeds a workload fidelity budget;
- serving economics exceed the declared cost contract;
- runtime memory, dtype or architecture is incompatible;
- a persisted expected passport digest no longer matches the serving configuration;
- kernel identities are duplicated.

## Run

```bash
python -m pytest -q
python scripts/operate.py
```

Build and install:

```bash
python -m pip install build
python -m build
python -m pip install dist/*.whl
open-model-fidelity-passport
```

Verify a supplied serving variant:

```bash
open-model-fidelity-passport --input variant.json
```

## Proof surface

- `src/open_model_fidelity_passport.py` — passport compiler/verifier
- `src/open_model_fidelity_cli.py` — installable execution surface
- `tests/test_open_model_fidelity_passport.py` — fidelity, cost, lineage, compatibility and drift behavior
- `tests/test_adversarial.py` — fail-closed adversarial coverage
- `.github/workflows/tests.yml` — tests + cold-start + wheel build/install + installed CLI
- `machine/` — existing Helix target/proof/authority surfaces remain preserved

## Current boundary

This is a vendor-neutral serving-passport mechanism. It does not claim Together AI infrastructure access, production deployment, enterprise-scale measurements, or proprietary model metadata. The next depth step is adapters that harvest exact fingerprints and eval receipts from a permitted serving runtime rather than receiving normalized observations as input.
