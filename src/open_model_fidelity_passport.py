"""Open-Model Fidelity Passport.

Builds and verifies deterministic serving passports that bind model lineage,
weights, quantization, sampler defaults, kernel stack, compatibility constraints,
evaluation deltas and serving economics into one machine-verifiable fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


class Decision(str, Enum):
    ALLOW = "ALLOW"
    REFUSE = "REFUSE"


@dataclass(frozen=True)
class OpenModelFidelityPassportRequest:
    subject_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    budget: float = 1.0
    grant_id: str | None = None
    not_after: float | None = None


@dataclass(frozen=True)
class OpenModelFidelityPassportReceipt:
    decision: Decision
    reasons: tuple[str, ...]
    digest: str
    metrics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"decision": self.decision.value, "reasons": list(self.reasons), "digest": self.digest, "metrics": self.metrics}


class PassportError(ValueError):
    pass


class OpenModelFidelityPassport:
    MIN_BUDGET = 0.0
    VALID_QUANTIZATION_BITS = frozenset({2, 3, 4, 6, 8, 16})

    @staticmethod
    def _id(value: Any, label: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise PassportError(f"{label}_missing")
        return value

    @staticmethod
    def _number(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PassportError(f"{label}_invalid")
        number = float(value)
        if not math.isfinite(number):
            raise PassportError(f"{label}_not_finite")
        if minimum is not None and number < minimum:
            raise PassportError(f"{label}_below_minimum")
        if maximum is not None and number > maximum:
            raise PassportError(f"{label}_above_maximum")
        return number

    @classmethod
    def _lineage(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise PassportError("lineage_missing")
        weights_digest = str(raw.get("weights_digest", "")).strip()
        if not SHA256_RE.fullmatch(weights_digest):
            raise PassportError("weights_digest_invalid")
        parent = raw.get("parent_weights_digest")
        if parent is not None and not SHA256_RE.fullmatch(str(parent).strip()):
            raise PassportError("parent_weights_digest_invalid")
        return {
            "base_model": cls._id(raw.get("base_model"), "base_model"),
            "weights_digest": weights_digest,
            "parent_weights_digest": str(parent).strip() if parent else None,
            "revision": cls._id(raw.get("revision"), "revision"),
            "license_id": cls._id(raw.get("license_id"), "license_id"),
        }

    @classmethod
    def _quantization(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise PassportError("quantization_missing")
        bits = raw.get("bits")
        if isinstance(bits, bool) or not isinstance(bits, int) or bits not in cls.VALID_QUANTIZATION_BITS:
            raise PassportError("quantization_bits_invalid")
        return {
            "format": cls._id(raw.get("format"), "quantization_format"),
            "bits": bits,
            "group_size": int(cls._number(raw.get("group_size", 0), "quantization_group_size", minimum=0)),
            "calibration_digest": cls._id(raw.get("calibration_digest"), "calibration_digest"),
        }

    @classmethod
    def _sampler(cls, raw: Any) -> dict[str, float]:
        if not isinstance(raw, dict):
            raise PassportError("sampler_defaults_missing")
        return {
            "temperature": cls._number(raw.get("temperature"), "temperature", minimum=0),
            "top_p": cls._number(raw.get("top_p"), "top_p", minimum=0, maximum=1),
            "top_k": cls._number(raw.get("top_k"), "top_k", minimum=0),
        }

    @classmethod
    def _kernels(cls, raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list) or not raw:
            raise PassportError("kernel_stack_missing")
        rows: list[dict[str, str]] = []
        seen: set[str] = set()
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise PassportError(f"kernel_{index}_not_object")
            name = cls._id(item.get("name"), f"kernel_{index}_name")
            if name in seen:
                raise PassportError(f"duplicate_kernel:{name}")
            seen.add(name)
            rows.append({"name": name, "version": cls._id(item.get("version"), f"kernel_{index}_version")})
        return sorted(rows, key=lambda row: (row["name"], row["version"]))

    @classmethod
    def _compatibility(cls, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            raise PassportError("compatibility_missing")
        dtypes = raw.get("supported_dtypes")
        if not isinstance(dtypes, list) or not dtypes or any(not str(v).strip() for v in dtypes):
            raise PassportError("supported_dtypes_invalid")
        architectures = raw.get("architectures")
        if not isinstance(architectures, list) or not architectures or any(not str(v).strip() for v in architectures):
            raise PassportError("architectures_invalid")
        return {
            "min_gpu_memory_gb": cls._number(raw.get("min_gpu_memory_gb"), "min_gpu_memory_gb", minimum=0),
            "supported_dtypes": sorted(set(str(v).strip() for v in dtypes)),
            "architectures": sorted(set(str(v).strip() for v in architectures)),
        }

    @classmethod
    def _evals(cls, raw: Any) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(raw, dict) or not raw:
            raise PassportError("evaluations_missing")
        normalized: dict[str, Any] = {}
        violations: list[str] = []
        for workload, row in sorted(raw.items()):
            name = str(workload).strip()
            if not name or not isinstance(row, dict):
                raise PassportError("evaluation_invalid")
            baseline = cls._number(row.get("baseline_score"), f"{name}_baseline_score")
            variant = cls._number(row.get("variant_score"), f"{name}_variant_score")
            max_drop = cls._number(row.get("max_drop"), f"{name}_max_drop", minimum=0)
            drop = baseline - variant
            normalized[name] = {
                "baseline_score": baseline,
                "variant_score": variant,
                "max_drop": max_drop,
                "observed_drop": round(drop, 12),
            }
            if drop > max_drop:
                violations.append(f"fidelity_budget_exceeded:{name}")
        return normalized, violations

    @classmethod
    def build_passport(cls, raw: Any) -> tuple[dict[str, Any], list[str]]:
        if not isinstance(raw, dict):
            raise PassportError("variant_missing")
        evaluations, violations = cls._evals(raw.get("evaluations"))
        economics = raw.get("economics")
        if not isinstance(economics, dict):
            raise PassportError("economics_missing")
        baseline_cost = cls._number(economics.get("baseline_cost_per_million_tokens"), "baseline_cost", minimum=0)
        variant_cost = cls._number(economics.get("variant_cost_per_million_tokens"), "variant_cost", minimum=0)
        max_cost_ratio = cls._number(economics.get("max_cost_ratio", 1.0), "max_cost_ratio", minimum=0)
        cost_ratio = 0.0 if baseline_cost == 0 and variant_cost == 0 else (math.inf if baseline_cost == 0 else variant_cost / baseline_cost)
        if not math.isfinite(cost_ratio) or cost_ratio > max_cost_ratio:
            violations.append("cost_contract_exceeded")
        body = {
            "schema": "glaciereq.open-model-fidelity-passport.v1",
            "variant_id": cls._id(raw.get("variant_id"), "variant_id"),
            "lineage": cls._lineage(raw.get("lineage")),
            "quantization": cls._quantization(raw.get("quantization")),
            "sampler_defaults": cls._sampler(raw.get("sampler_defaults")),
            "kernel_stack": cls._kernels(raw.get("kernel_stack")),
            "compatibility": cls._compatibility(raw.get("compatibility")),
            "evaluations": evaluations,
            "economics": {
                "baseline_cost_per_million_tokens": baseline_cost,
                "variant_cost_per_million_tokens": variant_cost,
                "max_cost_ratio": max_cost_ratio,
                "observed_cost_ratio": round(cost_ratio, 12),
            },
        }
        return {**body, "passport_digest": _digest(body)}, violations

    @staticmethod
    def _runtime_violations(passport: dict[str, Any], runtime: Any) -> list[str]:
        if runtime is None:
            return []
        if not isinstance(runtime, dict):
            raise PassportError("runtime_not_object")
        violations: list[str] = []
        compat = passport["compatibility"]
        memory = runtime.get("gpu_memory_gb")
        if memory is not None and float(memory) < compat["min_gpu_memory_gb"]:
            violations.append("runtime_gpu_memory_incompatible")
        dtype = runtime.get("dtype")
        if dtype is not None and str(dtype) not in compat["supported_dtypes"]:
            violations.append("runtime_dtype_incompatible")
        architecture = runtime.get("architecture")
        if architecture is not None and str(architecture) not in compat["architectures"]:
            violations.append("runtime_architecture_incompatible")
        return violations

    def evaluate(self, req: OpenModelFidelityPassportRequest) -> OpenModelFidelityPassportReceipt:
        reasons: list[str] = []
        if not str(req.subject_id or "").strip():
            reasons.append("subject_id_missing")
        if isinstance(req.budget, bool) or not isinstance(req.budget, (int, float)) or not math.isfinite(float(req.budget)) or float(req.budget) <= self.MIN_BUDGET:
            reasons.append("budget_non_positive_or_invalid")
        payload = req.payload if isinstance(req.payload, dict) else {}
        if not isinstance(req.payload, dict):
            reasons.append("payload_not_object")
        passport: dict[str, Any] | None = None
        try:
            passport, violations = self.build_passport(payload.get("variant"))
            reasons.extend(violations)
            reasons.extend(self._runtime_violations(passport, payload.get("runtime")))
            expected = payload.get("expected_passport_digest")
            if expected is not None and str(expected).strip() != passport["passport_digest"]:
                reasons.append("passport_digest_mismatch")
        except (PassportError, TypeError, ValueError) as exc:
            reasons.append(str(exc))
        decision = Decision.REFUSE if reasons else Decision.ALLOW
        metrics = {"passport": passport, "passport_digest": passport.get("passport_digest") if passport else None}
        body = {"subject_id": req.subject_id, "decision": decision.value, "reasons": reasons, "metrics": metrics}
        return OpenModelFidelityPassportReceipt(decision, tuple(reasons or ["fidelity_passport_verified"]), _digest(body), metrics)


Mechanism = OpenModelFidelityPassport
