from __future__ import annotations

import hashlib

from open_model_fidelity_passport import Decision, OpenModelFidelityPassport, OpenModelFidelityPassportRequest


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def variant() -> dict:
    return {
        "variant_id": "llama-open-int4",
        "lineage": {
            "base_model": "org/base-model",
            "weights_digest": sha("weights-v2"),
            "parent_weights_digest": sha("weights-v1"),
            "revision": "r2",
            "license_id": "apache-2.0",
        },
        "quantization": {"format": "gptq", "bits": 4, "group_size": 128, "calibration_digest": sha("calibration")},
        "sampler_defaults": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
        "kernel_stack": [{"name": "attention", "version": "2.1"}, {"name": "gemm", "version": "1.4"}],
        "compatibility": {"min_gpu_memory_gb": 24, "supported_dtypes": ["fp16", "bf16"], "architectures": ["sm80", "sm90"]},
        "evaluations": {
            "reasoning": {"baseline_score": 0.80, "variant_score": 0.79, "max_drop": 0.02},
            "code": {"baseline_score": 0.75, "variant_score": 0.745, "max_drop": 0.01},
        },
        "economics": {"baseline_cost_per_million_tokens": 1.0, "variant_cost_per_million_tokens": 0.7, "max_cost_ratio": 0.8},
    }


def evaluate(v: dict, **extra):
    payload = {"variant": v, **extra}
    return OpenModelFidelityPassport().evaluate(OpenModelFidelityPassportRequest(subject_id="serve-a", payload=payload, budget=1.0))


def test_builds_machine_verifiable_passport() -> None:
    receipt = evaluate(variant())
    assert receipt.decision is Decision.ALLOW
    passport = receipt.metrics["passport"]
    assert passport["schema"] == "glaciereq.open-model-fidelity-passport.v1"
    assert passport["lineage"]["weights_digest"] == sha("weights-v2")
    assert len(passport["passport_digest"]) == 64
    assert passport["economics"]["observed_cost_ratio"] == 0.7


def test_fidelity_regression_fails_closed_by_workload() -> None:
    v = variant()
    v["evaluations"]["code"]["variant_score"] = 0.70
    receipt = evaluate(v)
    assert receipt.decision is Decision.REFUSE
    assert "fidelity_budget_exceeded:code" in receipt.reasons


def test_cost_contract_must_be_earned() -> None:
    v = variant()
    v["economics"]["variant_cost_per_million_tokens"] = 0.95
    receipt = evaluate(v)
    assert receipt.decision is Decision.REFUSE
    assert "cost_contract_exceeded" in receipt.reasons


def test_runtime_compatibility_checks_memory_dtype_and_architecture() -> None:
    receipt = evaluate(variant(), runtime={"gpu_memory_gb": 16, "dtype": "fp32", "architecture": "sm70"})
    assert receipt.decision is Decision.REFUSE
    assert "runtime_gpu_memory_incompatible" in receipt.reasons
    assert "runtime_dtype_incompatible" in receipt.reasons
    assert "runtime_architecture_incompatible" in receipt.reasons


def test_compatible_runtime_is_allowed() -> None:
    receipt = evaluate(variant(), runtime={"gpu_memory_gb": 48, "dtype": "bf16", "architecture": "sm90"})
    assert receipt.decision is Decision.ALLOW


def test_expected_passport_digest_detects_serving_configuration_drift() -> None:
    first = evaluate(variant())
    expected = first.metrics["passport_digest"]
    changed = variant()
    changed["sampler_defaults"]["temperature"] = 0.2
    receipt = evaluate(changed, expected_passport_digest=expected)
    assert receipt.decision is Decision.REFUSE
    assert "passport_digest_mismatch" in receipt.reasons


def test_invalid_weights_lineage_is_refused() -> None:
    v = variant()
    v["lineage"]["weights_digest"] = "not-a-sha"
    receipt = evaluate(v)
    assert receipt.decision is Decision.REFUSE
    assert "weights_digest_invalid" in receipt.reasons


def test_duplicate_kernel_identity_is_refused() -> None:
    v = variant()
    v["kernel_stack"].append({"name": "attention", "version": "2.2"})
    receipt = evaluate(v)
    assert receipt.decision is Decision.REFUSE
    assert "duplicate_kernel:attention" in receipt.reasons


def test_reordering_kernel_input_does_not_change_passport_digest() -> None:
    v1 = variant()
    v2 = variant()
    v2["kernel_stack"] = list(reversed(v2["kernel_stack"]))
    assert evaluate(v1).metrics["passport_digest"] == evaluate(v2).metrics["passport_digest"]
