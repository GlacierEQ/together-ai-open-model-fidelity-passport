# Issue contract — Open-Model Fidelity Passport

## Problem
maintaining broad model choice and training-to-serving integration while proving behavioral fidelity and cost advantage at enterprise scale

## Desired outcome
A bounded, open, testable implementation of **Open-Model Fidelity Passport** that demonstrates Publish machine-verifiable serving fingerprints: weights lineage, quantization, sampler defaults, kernel stack, eval deltas, and compatibility constraints for every deployed model variant.

## Non-goals
- Together AI affiliation or proprietary integration
- Portfolio-wide scale/performance claims
- UI marketing site

## Acceptance
1. Mechanism module implements allow + refuse with structured receipts
2. pytest behavioral suite green
3. operate.py cold-start produces JSON receipt
4. Non-affiliation disclaimer preserved
