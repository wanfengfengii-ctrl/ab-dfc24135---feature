"""Unit tests for the audit-plan domain solver.

Optimality and lexicographic tie-breaking are cross-checked against an
exhaustive enumeration in randomized tests; infeasibility must never
produce a fabricated plan.
"""

from __future__ import annotations

import itertools
import random

import pytest

from app.audit import (
    MAX_TARGET,
    MIN_TARGET,
    AuditPlanError,
    build_audit_plan,
)


def _brute(n, risks, target, zones):
    best = None
    for comb in itertools.combinations(range(n), target):
        if any(b == a + 1 for a, b in zip(comb, comb[1:])):
            continue
        if any(sum(1 for c in comb if s <= c <= e) < q for s, e, q in zones):
            continue
        total = sum(risks[c] for c in comb)
        key = (total, tuple(-x for x in comb))
        if best is None or key > best[0]:
            best = (key, comb)
    return None if best is None else (sum(risks[c] for c in best[1]), best[1])


def _make_zones(raw):
    return [{"start": s, "end": e, "quota": q} for s, e, q in raw]


def test_selects_exact_target_without_adjacency():
    plan = build_audit_plan(
        7,
        {"target": 3, "risks": [10, 90, 20, 80, 30, 70, 40], "zones": _make_zones([(0, 6, 1)])},
    )
    sel = plan["selected"]
    assert len(sel) == 3
    assert all(b != a + 1 for a, b in zip(sel, sel[1:]))
    assert plan["risk_total"] == sum(plan["per_chunk"][i]["risk"] for i in sel)


def test_zone_quota_is_met():
    plan = build_audit_plan(
        8,
        {
            "target": 3,
            "risks": [100, 0, 0, 0, 0, 0, 0, 100],
            "zones": _make_zones([(2, 4, 2)]),
        },
    )
    sel = plan["selected"]
    assert sum(1 for i in sel if 2 <= i <= 4) >= 2
    zone = plan["zones"][0]
    assert zone["quota_met"] is True
    assert zone["selected_count"] >= 2


def test_lexicographic_tie_break_prefers_smaller_indices():
    # All scores equal: the lex-smallest non-adjacent triple of 6 wins.
    plan = build_audit_plan(
        6,
        {"target": 3, "risks": [5, 5, 5, 5, 5, 5], "zones": _make_zones([(0, 5, 1)])},
    )
    assert plan["selected"] == [0, 2, 4]


def test_risk_max_beats_lexicographic_order():
    # [1,3,5] sums to 90*3 but adjacent-free combos; the optimizer must
    # prefer the high scores even though [0,2,4] is lex-smaller.
    plan = build_audit_plan(
        6,
        {"target": 3, "risks": [1, 90, 1, 90, 1, 90], "zones": _make_zones([(0, 5, 1)])},
    )
    assert plan["selected"] == [1, 3, 5]
    assert plan["risk_total"] == 270


def test_infeasible_adjacent_zones_returns_blocking_conditions():
    # [0,2] q=2 forces {0,2}; [3,5] q=2 forces {3,5}: 2 and 3 collide.
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            10,
            {
                "target": 5,
                "risks": list(range(10)),
                "zones": _make_zones([(0, 2, 2), (3, 5, 2)]),
            },
        )
    assert exc.value.status_code == 409
    types = {b["type"] for b in exc.value.extra["blocking"]}
    assert "zone_boundary_conflict" in types


def test_target_over_global_capacity_is_blocking():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            5,
            {"target": 4, "risks": [1] * 5, "zones": _make_zones([(0, 4, 1)])},
        )
    assert exc.value.status_code == 409
    assert any(b["type"] == "global_capacity" for b in exc.value.extra["blocking"])


def test_quota_sum_over_target_is_blocking():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            12,
            {
                "target": 3,
                "risks": [1] * 12,
                "zones": _make_zones([(0, 3, 2), (6, 9, 2)]),
            },
        )
    assert exc.value.status_code == 409
    assert any(b["type"] == "quota_sum" for b in exc.value.extra["blocking"])


def test_zone_capacity_is_blocking():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            10,
            {"target": 4, "risks": [1] * 10, "zones": _make_zones([(0, 2, 3)])},
        )
    assert exc.value.status_code == 409
    assert any(b["type"] == "zone_capacity" for b in exc.value.extra["blocking"])


# ---------------------------------------------------------------------------
# located validation rejections (all 400)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"target": MIN_TARGET - 1, "risks": [1] * 4, "zones": _make_zones([(0, 1, 1)])},
        {"target": MAX_TARGET + 1, "risks": [1] * 4, "zones": _make_zones([(0, 1, 1)])},
        {"target": 5, "risks": [1] * 4, "zones": _make_zones([(0, 1, 1)])},
        {"target": 3, "risks": [1] * 3, "zones": _make_zones([(0, 1, 1)])},  # wrong count
        {"target": 3, "risks": [1, 101, 1, 1], "zones": _make_zones([(0, 1, 1)])},
        {"target": 3, "risks": [-1, 1, 1, 1], "zones": _make_zones([(0, 1, 1)])},
        {"target": 3, "risks": [1] * 4, "zones": []},  # no zones
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(0, i, 1) for i in range(5)])},
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(3, 1, 1)])},  # start > end
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(0, 4, 1)])},  # out of range
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(0, 2, 0)])},  # quota < 1
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(0, 1, 3)])},  # quota > length
        {"target": 3, "risks": [1] * 4, "zones": _make_zones([(0, 2, 1), (1, 3, 1)])},  # overlap
    ],
)
def test_invalid_requests_are_rejected_with_location(body):
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(4, body)
    assert exc.value.status_code == 400


def test_overlapping_zones_message_names_both():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            6,
            {
                "target": 3,
                "risks": [1] * 6,
                "zones": _make_zones([(0, 3, 1), (3, 5, 1)]),
            },
        )
    assert exc.value.status_code == 400
    assert "overlap" in exc.value.error


def test_wrong_risk_count_message_is_specific():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            5,
            {"target": 2, "risks": [1, 2, 3], "zones": _make_zones([(0, 1, 1)])},
        )
    assert "exactly 5" in exc.value.error and "got 3" in exc.value.error


def test_non_integer_values_rejected():
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            4,
            {"target": 2.5, "risks": [1] * 4, "zones": _make_zones([(0, 1, 1)])},
        )
    assert exc.value.status_code == 400
    with pytest.raises(AuditPlanError) as exc:
        build_audit_plan(
            4,
            {"target": True, "risks": [1] * 4, "zones": _make_zones([(0, 1, 1)])},
        )
    assert exc.value.status_code == 400


# ---------------------------------------------------------------------------
# randomized cross-check against exhaustive enumeration
# ---------------------------------------------------------------------------


def test_matches_brute_force_on_random_cases():
    rng = random.Random(20260924)
    feasible = 0
    for _ in range(1500):
        n = rng.randint(2, 11)
        risks = [rng.randint(0, 100) for _ in range(n)]
        m = rng.randint(1, min(4, n))
        starts = sorted(rng.sample(range(n), m))
        zones = []
        for j, st in enumerate(starts):
            limit = (starts[j + 1] - 1) if j + 1 < m else n - 1
            en = rng.randint(st, limit)
            zones.append((st, en, rng.randint(1, en - st + 1)))
        target = rng.randint(MIN_TARGET, min(MAX_TARGET, n))
        body = {
            "target": target,
            "risks": risks,
            "zones": [{"start": s, "end": e, "quota": q} for s, e, q in zones],
        }
        expected = _brute(n, risks, target, zones)
        try:
            plan = build_audit_plan(n, body)
        except AuditPlanError as exc:
            assert expected is None, f"solver said infeasible, brute found {expected}: {body}"
            assert exc.status_code == 409
            continue
        assert expected is not None, f"solver fabricated a plan for {body}"
        assert plan["risk_total"] == expected[0]
        assert tuple(plan["selected"]) == expected[1]
        feasible += 1
    assert feasible > 100
