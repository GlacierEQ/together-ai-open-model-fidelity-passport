# DEV_UP_INSTRUCTIONS — implementation record

**Repository:** `GlacierEQ/together-ai-open-model-fidelity-passport`  
**Independent company lens:** Together AI  
**Innovation:** Open-Model Fidelity Passport

## Mission

Make model serving variants machine-verifiable across lineage, transformation, behavior, compatibility and economics.

## Implemented

The generic scaffold has been replaced by a deterministic serving-passport compiler and verifier.

`src/open_model_fidelity_passport.py` now:

- binds exact weights lineage, revision and license identity;
- records quantization/calibration, sampler defaults and canonical kernel stack;
- encodes runtime compatibility requirements;
- evaluates workload-level behavioral deltas against explicit fidelity budgets;
- evaluates serving cost against an explicit cost-ratio contract;
- verifies runtime memory/dtype/architecture compatibility;
- detects serving configuration drift through an expected passport digest;
- rejects malformed lineage and duplicate kernel identities;
- emits a deterministic passport plus SHA-256 decision receipt.

`src/open_model_fidelity_cli.py` and `scripts/operate.py` execute the mechanism directly. The project is packaged with the `open-model-fidelity-passport` console command.

## Verification contract

Behavioral tests cover valid passport generation, workload regression refusal, cost-contract refusal, runtime incompatibility, compatible runtime, passport-drift detection, invalid weights lineage, duplicate kernels, and canonical ordering. Existing adversarial coverage remains active.

CI must pass native tests, cold-start operation, wheel build/install and installed CLI execution before Helix may mint source-bound promotion evidence.

## Truth boundary

No Together AI affiliation, proprietary access, production deployment, customer impact, or company partnership is claimed. Current input is normalized serving evidence; runtime/eval harvesting adapters remain a further end-to-end depth step.
