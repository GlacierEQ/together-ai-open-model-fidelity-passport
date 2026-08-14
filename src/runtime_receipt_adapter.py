"""Compile fidelity-passport inputs from independently produced runtime receipts."""
from __future__ import annotations

import re
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class RuntimeReceiptError(ValueError):
    pass


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeReceiptError(f"{label}_not_object")
    return value


def _text(value: Any, label: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise RuntimeReceiptError(f"{label}_missing")
    return result


def _digest(value: Any, label: str) -> str:
    result = _text(value, label)
    if not SHA256_RE.fullmatch(result):
        raise RuntimeReceiptError(f"{label}_invalid")
    return result


def compile_variant_from_receipts(
    runtime_receipt: dict[str, Any],
    evaluation_receipt: dict[str, Any],
    economics_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Cross-bind three evidence planes into one passport input.

    Each receipt must independently identify the same variant and exact weights
    digest. A serving/runtime receipt supplies lineage/configuration/compatibility,
    an evaluation receipt supplies workload outcomes, and an economics receipt
    supplies serving cost evidence. Cross-receipt identity drift fails closed.
    """
    runtime_receipt = _object(runtime_receipt, "runtime_receipt")
    evaluation_receipt = _object(evaluation_receipt, "evaluation_receipt")
    economics_receipt = _object(economics_receipt, "economics_receipt")

    receipts = (runtime_receipt, evaluation_receipt, economics_receipt)
    variant_ids = {_text(row.get("variant_id"), "variant_id") for row in receipts}
    if len(variant_ids) != 1:
        raise RuntimeReceiptError("variant_id_mismatch")
    weight_digests = {_digest(row.get("weights_digest"), "weights_digest") for row in receipts}
    if len(weight_digests) != 1:
        raise RuntimeReceiptError("weights_digest_mismatch")

    quantization = _object(runtime_receipt.get("quantization"), "quantization")
    sampler = _object(runtime_receipt.get("sampler_defaults"), "sampler_defaults")
    compatibility = _object(runtime_receipt.get("compatibility"), "compatibility")
    runtime = _object(runtime_receipt.get("runtime"), "runtime")
    kernels = runtime_receipt.get("kernel_stack")
    if not isinstance(kernels, list) or not kernels:
        raise RuntimeReceiptError("kernel_stack_missing")
    evaluations = evaluation_receipt.get("evaluations")
    if not isinstance(evaluations, dict) or not evaluations:
        raise RuntimeReceiptError("evaluations_missing")
    economics = economics_receipt.get("economics")
    if not isinstance(economics, dict):
        raise RuntimeReceiptError("economics_missing")

    variant_id = next(iter(variant_ids))
    weights_digest = next(iter(weight_digests))
    variant = {
        "variant_id": variant_id,
        "lineage": {
            "base_model": _text(runtime_receipt.get("base_model"), "base_model"),
            "weights_digest": weights_digest,
            "parent_weights_digest": runtime_receipt.get("parent_weights_digest"),
            "revision": _text(runtime_receipt.get("revision"), "revision"),
            "license_id": _text(runtime_receipt.get("license_id"), "license_id"),
        },
        "quantization": quantization,
        "sampler_defaults": sampler,
        "kernel_stack": kernels,
        "compatibility": compatibility,
        "evaluations": evaluations,
        "economics": economics,
        "evidence": {
            "source": "cross_bound_runtime_evaluation_economics_receipts",
            "variant_id_bound_across_receipts": True,
            "weights_digest_bound_across_receipts": True,
        },
    }
    return variant, runtime
