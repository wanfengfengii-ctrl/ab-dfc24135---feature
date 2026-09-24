"""Domain tests for the sampling audit-plan optimizer."""

from __future__ import annotations

import itertools

import pytest

from app.audit import (
    MAX_TARGET,
    AuditValidationError,
    build_audit_plan,
)


def _brute(n, target, scores, ranges):
    """Reference optimizer: enumerate every combination."""
    best = None
    for combo in itertools.combinations(range(n), target):
        if any(b - a < 2 for a, b in zip(combo, combo[1:])):
            continue
        if not all(
            sum(1 for b in combo if s <= b <= e) >= q for s, e, q in ranges
        ):
            continue
        val = sum(scores[b] for b in combo)
        key = (-val, combo)
        if best is None or key < best[0]:
            best = (key, list(combo), val)
    return None if best is None else (best[1], best[2])


def _ranges(raw):
    return [(r["start"], r["end"], r["quota"]) for r in raw]


def test_picks_highest_risk_non_adjacent():
    # [10,90,10,90,10,80], target 2 -> 90@1 and 90@3 (non-adjacent), sum 180
    p = build_audit_plan(
        6, 2, [10, 90, 10, 90, 10, 80], [{"start": 0, "end": 5, "quota": 0}]
    )
    assert p.solvable is True
    assert p.blocks == [1, 3]
    assert p.risk_sum == 180


def test_tie_break_is_lexicographically_smallest_sequence():
    p = build_audit_plan(8, 3, [5] * 8, [{"start": 0, "end": 7, "quota": 0}])
    assert p.solvable is True
    # equal sums everywhere -> the smallest possible ascending sequence
    assert p.blocks == [0, 2, 4]
    assert p.risk_sum == 15


def test_quota_forces_blocks_inside_range():
    # best free choice would be the 100s at #0 and #7; quota on [4,5] forces
    # one of #4/#5 into the plan, dropping #7... #7 is still compatible with #5
    scores = [100, 0, 0, 0, 90, 80, 0, 100]
    p = build_audit_plan(8, 3, scores, [{"start": 4, "end": 5, "quota": 1}])
    assert p.solvable is True
    assert p.blocks == [0, 4, 7]
    assert p.risk_sum == 290
    hit = next(r for r in p.to_dict()["ranges"] if r["start"] == 4)
    assert hit["selected"] == 1 and hit["quota"] == 1


def test_quota_accounting_per_block():
    scores = list(range(10))
    p = build_audit_plan(
        10,
        3,
        scores,
        [
            {"start": 0, "end": 2, "quota": 1},
            {"start": 7, "end": 9, "quota": 1},
        ],
    )
    assert p.solvable is True
    ranges = {(r["start"], r["end"]): r for r in p.to_dict()["ranges"]}
    assert ranges[(0, 2)]["selected"] >= 1
    assert ranges[(7, 9)]["selected"] >= 1
    assert sum(r["selected"] for r in ranges.values()) <= 3


@pytest.mark.parametrize(
    "n,target,ranges",
    [
        # forced blocks #0 and #3 leave no legal third pick in 4.. but n=5:
        # #4 adjacent to #3, #1/#2 adjacent to #0/#3
        (5, 3, [{"start": 0, "end": 0, "quota": 1},
                {"start": 3, "end": 3, "quota": 1}]),
    ],
)
def test_infeasible_that_only_the_dp_detects(n, target, ranges):
    p = build_audit_plan(n, target, [1] * n, ranges)
    assert p.solvable is False
    assert p.blocks == []
    assert "no feasible plan" in p.block_reason


def test_infeasible_target_exceeds_non_adjacent_capacity():
    # 2 blocks can provide at most 1 non-adjacent selection, target is 2
    p = build_audit_plan(2, 2, [1, 2], [{"start": 0, "end": 1, "quota": 0}])
    assert p.solvable is False
    assert "at most 1" in p.block_reason
    # 3 blocks with target 2 IS feasible ({0,2}); sanity guard
    ok = build_audit_plan(3, 2, [1, 2, 3], [{"start": 0, "end": 2, "quota": 0}])
    assert ok.solvable is True and ok.blocks == [0, 2]


def test_infeasible_quotas_exceed_target():
    p = build_audit_plan(
        10,
        3,
        [5] * 10,
        [
            {"start": 0, "end": 2, "quota": 2},
            {"start": 5, "end": 7, "quota": 2},
        ],
    )
    assert p.solvable is False
    assert "quotas require at least 4" in p.block_reason


def test_infeasible_quota_within_tight_range():
    # a 3-block range can hold at most 2 non-adjacent selections
    p = build_audit_plan(8, 3, [1] * 8, [{"start": 2, "end": 4, "quota": 3}])
    assert p.solvable is False
    assert "[2,4]" in p.block_reason


def test_dp_never_fabricates_a_plan():
    p = build_audit_plan(
        6, 3, [1] * 6, [{"start": 0, "end": 5, "quota": 3}]
    )
    # 6 blocks hold exactly 3 non-adjacent selections (0,2,4) -> feasible
    assert p.solvable is True and p.blocks == [0, 2, 4]


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        (dict(chunk_count=8, target=1, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "between 2 and 16"),
        (dict(chunk_count=8, target=MAX_TARGET + 1, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "between 2 and 16"),
        (dict(chunk_count=8, target=9, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "must not exceed"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 7,
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "must match"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 7 + [101],
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "risk_scores[7]"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 7 + [-1],
              raw_ranges=[{"start": 0, "end": 7, "quota": 0}]), "risk_scores[7]"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 8,
              raw_ranges=[]), "1-4"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 8, "quota": 0}]), "out of bounds"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 8,
              raw_ranges=[{"start": 3, "end": 2, "quota": 0}]), "end 2"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 2, "quota": 4}]), "exceeds interval"),
        (dict(chunk_count=8, target=3, risk_scores=[0] * 8,
              raw_ranges=[{"start": 0, "end": 2, "quota": -1}]), "quota"),
    ],
)
def test_validation_errors(kwargs, fragment):
    with pytest.raises(AuditValidationError) as exc:
        build_audit_plan(**kwargs)
    assert fragment in str(exc.value)


def test_overlapping_ranges_rejected_with_location():
    with pytest.raises(AuditValidationError) as exc:
        build_audit_plan(
            8, 2, [0] * 8,
            [{"start": 0, "end": 3, "quota": 1}, {"start": 3, "end": 5, "quota": 1}],
        )
    assert "must not overlap" in str(exc.value)


def test_five_ranges_rejected():
    with pytest.raises(AuditValidationError):
        build_audit_plan(
            12,
            2,
            [0] * 12,
            [{"start": i, "end": i, "quota": 0} for i in range(5)],
        )


def test_wrong_types_and_bool_rejected():
    with pytest.raises(AuditValidationError):
        build_audit_plan(8, True, [0] * 8, [{"start": 0, "end": 7, "quota": 0}])
    with pytest.raises(AuditValidationError):
        build_audit_plan(
            8, 2, [0] * 7 + ["9"], [{"start": 0, "end": 7, "quota": 0}]
        )
    with pytest.raises(AuditValidationError):
        build_audit_plan(
            8, 2, [0] * 8, [{"start": "0", "end": 7, "quota": 0}]
        )
    with pytest.raises(AuditValidationError):
        build_audit_plan(8, 2, [0] * 8, [{"start": 0, "end": 7}])


def test_optimizer_matches_brute_force():
    import random

    random.seed(2026)
    cases = 0
    for _ in range(300):
        n = random.randint(2, 12)
        target = random.randint(2, min(6, n))
        scores = [random.randint(0, 100) for _ in range(n)]
        rc = random.randint(1, 3)
        if n + 1 < 2 * rc:
            continue
        cuts = sorted(random.sample(range(n + 1), 2 * rc))
        raw = []
        ok = True
        for j in range(rc):
            s, e = cuts[2 * j], cuts[2 * j + 1] - 1
            if s > e:
                ok = False
                break
            raw.append({"start": s, "end": e, "quota": random.randint(0, (e - s) // 2 + 1)})
        if not ok:
            continue
        cases += 1
        p = build_audit_plan(n, target, scores, raw)
        expected = _brute(n, target, scores, _ranges(raw))
        if expected is None:
            assert p.solvable is False, (n, target, scores, raw, p.blocks)
        else:
            assert p.solvable is True
            assert (p.blocks, p.risk_sum) == expected
    assert cases > 200
