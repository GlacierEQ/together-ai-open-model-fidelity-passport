from __future__ import annotations

import pytest

from open_model_fidelity_passport import OpenModelFidelityPassport, OpenModelFidelityPassportRequest
from runtime_receipt_adapter import RuntimeReceiptError, compile_variant_from_receipts

DIGEST = "a" * 64


def runtime_receipt():
    return {
        "variant_id": "model-q4",
        "weights_digest": DIGEST,
        "base_model": "example/model",
        "revision": "r1",
        "license_id": "apache-2.0",
        "parent_weights_digest": None,
        "quantization": {"format": "gptq", "bits": 4, "group_size": 128, "calibration_digest": "cal-1"},
        "sampler_defaults": {"temperature": 0.2, "top_p": 0.95, "top_k": 40},
        "kernel_stack": [{"name": "attention", "version": "1"}],
        "compatibility": {"min_gpu_memory_gb": 16, "supported_dtypes": ["fp16"], "architectures": ["sm80"]},
        "runtime": {"gpu_memory_gb": 24, "dtype": "fp16", "architecture": "sm80"},
    }


def evaluation_receipt():
    return {
        "variant_id": "model-q4",
        "weights_digest": DIGEST,
        "evaluations": {"coding": {"baseline_score": 0.90, "variant_score": 0.89, "max_drop": 0.02}},
    }


def economics_receipt():
    return {
        "variant_id": "model-q4",
        "weights_digest": DIGEST,
        "economics": {"baseline_cost_per_million_tokens": 1.0, "variant_cost_per_million_tokens": 0.7, "max_cost_ratio": 0.9},
    }


def test_cross_bound_receipts_compile_and_verify_passport() -> None:
    variant, runtime = compile_variant_from_receipts(
        runtime_receipt(), evaluation_receipt(), economics_receipt()
    )
    receipt = OpenModelFidelityPassport().evaluate(
        OpenModelFidelityPassportRequest(
            subject_id="receipt-bound",
            payload={"variant": variant, "runtime": runtime},
        )
    )
    assert receipt.decision.value == "ALLOW"
    assert receipt.metrics["passport"]["lineage"]["weights_digest"] == DIGEST


def test_adapter_refuses_cross_receipt_identity_drift() -> None:
    bad = evaluation_receipt()
    bad["variant_id"] = "other"
    with pytest.raises(RuntimeReceiptError, match="variant_id_mismatch"):
        compile_variant_from_receipts(runtime_receipt(), bad, economics_receipt())

    bad = economics_receipt()
    bad["weights_digest"] = "b" * 64
    with pytest.raises(RuntimeReceiptError, match="weights_digest_mismatch"):
        compile_variant_from_receipts(runtime_receipt(), evaluation_receipt(), bad)
