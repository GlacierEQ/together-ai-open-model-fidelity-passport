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

## Cross-bound runtime receipt adapter

`runtime_receipt_adapter.compile_variant_from_receipts()` compiles a passport input from three independently produced evidence planes rather than one caller-authored variant object:

1. a runtime receipt with lineage, quantization, sampler, kernel stack, compatibility and observed runtime facts;
2. an evaluation receipt with workload outcomes; and
3. an economics receipt with serving-cost evidence.

All three receipts must independently identify the same `variant_id` and exact SHA-256 `weights_digest`. Cross-receipt identity drift fails closed before the passport is built. This prevents a serving configuration from borrowing evaluation or economics evidence produced for another model artifact.

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
- `src/runtime_receipt_adapter.py` — cross-binding of runtime/eval/economics evidence
- `src/open_model_fidelity_cli.py` — installable execution surface
- `tests/test_open_model_fidelity_passport.py` — fidelity, cost, lineage, compatibility and drift behavior
- `tests/test_runtime_receipt_adapter.py` — cross-receipt identity and weights binding
- `tests/test_adversarial.py` — fail-closed adversarial coverage
- `.github/workflows/tests.yml` — tests + cold-start + wheel build/install + installed CLI
- `machine/` — existing Helix target/proof/authority surfaces remain preserved

## Current boundary

The passport can now be compiled from independently produced runtime, evaluation, and economics receipts that are cryptographically bound to the same declared weights digest. It does not yet harvest those receipts directly from a live serving runtime or hash model files itself, and it claims no Together AI infrastructure access, production deployment, enterprise-scale measurements, or proprietary model metadata. The next depth step is a permitted runtime collector that emits these verified receipt shapes from actual serving artifacts and evaluation runs.
