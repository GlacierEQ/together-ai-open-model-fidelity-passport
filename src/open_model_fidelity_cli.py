from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from open_model_fidelity_passport import Decision, OpenModelFidelityPassport, OpenModelFidelityPassportRequest


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def demo_payload() -> dict:
    return {
        "variant": {
            "variant_id": "open-model-int4",
            "lineage": {"base_model": "org/base-model", "weights_digest": sha("weights-v2"), "parent_weights_digest": sha("weights-v1"), "revision": "r2", "license_id": "apache-2.0"},
            "quantization": {"format": "gptq", "bits": 4, "group_size": 128, "calibration_digest": sha("calibration")},
            "sampler_defaults": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
            "kernel_stack": [{"name": "attention", "version": "2.1"}, {"name": "gemm", "version": "1.4"}],
            "compatibility": {"min_gpu_memory_gb": 24, "supported_dtypes": ["bf16", "fp16"], "architectures": ["sm80", "sm90"]},
            "evaluations": {
                "reasoning": {"baseline_score": 0.80, "variant_score": 0.79, "max_drop": 0.02},
                "code": {"baseline_score": 0.75, "variant_score": 0.745, "max_drop": 0.01},
            },
            "economics": {"baseline_cost_per_million_tokens": 1.0, "variant_cost_per_million_tokens": 0.7, "max_cost_ratio": 0.8},
        },
        "runtime": {"gpu_memory_gb": 48, "dtype": "bf16", "architecture": "sm90"},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and verify an open-model serving fidelity passport")
    parser.add_argument("--input", type=Path, help="JSON payload; defaults to a deterministic demo")
    parser.add_argument("--subject", default="fidelity-demo")
    parser.add_argument("--budget", type=float, default=1.0)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text()) if args.input else demo_payload()
    receipt = OpenModelFidelityPassport().evaluate(OpenModelFidelityPassportRequest(args.subject, payload, args.budget))
    print(json.dumps(receipt.as_dict(), indent=2, sort_keys=True))
    return 0 if receipt.decision is Decision.ALLOW else 2


if __name__ == "__main__":
    raise SystemExit(main())
