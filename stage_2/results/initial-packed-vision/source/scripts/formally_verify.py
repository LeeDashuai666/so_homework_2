#!/usr/bin/env python3
"""Bounded Z3 model of feature leases; includes an unsafe negative control."""

import itertools
import json
from pathlib import Path
import sys

import torch
import z3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vlmbench.cache import FeatureCache, Payload


def model(protect, steps=8, entries=3):
    solver = z3.Solver()
    # alive describes the actual payload, including a live borrowed reference;
    # refs are inference leases. Free represents eviction/replacement/release
    # of the backing allocation, not a harmless dictionary unlink.
    alive = [[z3.Bool(f"alive_{t}_{i}") for i in range(entries)] for t in range(steps + 1)]
    refs = [[z3.Int(f"refs_{t}_{i}") for i in range(entries)] for t in range(steps + 1)]
    operations = [z3.Int(f"op_{t}") for t in range(steps)]
    targets = [z3.Int(f"target_{t}") for t in range(steps)]
    for i in range(entries):
        solver.add(alive[0][i], refs[0][i] == 0)
    for t in range(steps):
        op, target = operations[t], targets[t]
        solver.add(op >= 0, op <= 4, target >= 0, target < entries)
        for i in range(entries):
            selected = target == i
            borrow = z3.And(selected, op == 0)
            release = z3.And(selected, op == 1)
            free = z3.And(selected, op == 2)
            refill = z3.And(selected, op == 3)
            solver.add(z3.Implies(borrow, z3.And(alive[t][i], refs[t][i] < 2)))
            solver.add(z3.Implies(release, refs[t][i] > 0))
            solver.add(z3.Implies(refill, z3.And(z3.Not(alive[t][i]), refs[t][i] == 0)))
            if protect:
                solver.add(z3.Implies(free, refs[t][i] == 0))
            solver.add(refs[t + 1][i] == z3.If(borrow, refs[t][i] + 1, z3.If(release, refs[t][i] - 1, refs[t][i])))
            solver.add(alive[t + 1][i] == z3.If(free, False, z3.If(refill, True, alive[t][i])))
    violations = [z3.And(refs[t][i] > 0, z3.Not(alive[t][i])) for t in range(steps + 1) for i in range(entries)]
    solver.add(z3.Or(violations))
    result = solver.check()
    witness = None
    if result == z3.sat:
        assignment = solver.model()
        witness = [{"step": t, "operation": assignment.eval(operations[t]).as_long(),
                    "target": assignment.eval(targets[t]).as_long(),
                    "alive": [z3.is_true(assignment.eval(alive[t][i])) for i in range(entries)],
                    "refs": [assignment.eval(refs[t][i]).as_long() for i in range(entries)]}
                   for t in range(steps)]
        witness.append({"step": steps, "alive": [z3.is_true(assignment.eval(alive[steps][i])) for i in range(entries)],
                        "refs": [assignment.eval(refs[steps][i]).as_long() for i in range(entries)]})
    return str(result), solver.to_smt2(), witness


def main():
    output = ROOT / "results/verification"
    output.mkdir(parents=True, exist_ok=True)
    safe, safe_smt, _ = model(True)
    unsafe, unsafe_smt, witness = model(False)
    (output / "leases-safe.smt2").write_text(safe_smt)
    (output / "leases-unsafe.smt2").write_text(unsafe_smt)
    # Exhaust the real implementation's victim-selection predicate, for both
    # policies, not only the abstract transition model.
    checked = 0
    for policy in ["lru", "value"]:
        for pins in itertools.product(range(3), repeat=3):
            cache = FeatureCache(1024, 1024, policy, device="cpu")
            for key, count in zip(["a", "b", "c"], pins):
                cache.touch(key)
                cache.gpu[key] = Payload(torch.ones(4), (torch.ones(4),))
                cache.pins[key] = count
            victim = cache.victim(cache.gpu)
            assert victim is None if all(pins) else victim is not None and cache.pins[victim] == 0
            checked += 1
    assert safe == "unsat" and unsafe == "sat"
    result = {"z3_version": z3.get_version_string(), "entries": 3, "steps": 8, "lease_bound_per_entry": 2,
              "property": "No live inference lease refers to a freed visual feature",
              "safe_with_guard": safe, "negative_control_without_guard": unsafe,
              "real_victim_selector_cases": checked,
              "negative_control_witness": witness,
              "operation_codes": {"0": "borrow", "1": "release", "2": "free", "3": "refill", "4": "idle"},
              "scope": "Bounded abstract ownership model plus exhaustive real selector checks; not a proof of Python/CUDA execution or neural numerical equivalence",
              "passed": True}
    (output / "formal.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
