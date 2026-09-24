"""Sampling audit-plan domain logic.

Given a sealed package of ``n`` blocks, a quality controller supplies:

* ``target``       – how many blocks the plan must contain (2..16, <= n);
* ``risk_scores``  – one 0..100 score per actual block (exactly n values);
* ``ranges``       – 1..4 mutually non-overlapping contiguous *focus
  intervals* of block indices, each with its own minimum sample quota.

A feasible plan must:

* contain exactly ``target`` blocks;
* never select two adjacent blocks (|i-j| >= 2);
* put at least ``quota`` selected blocks inside every focus interval.

Among **all** feasible plans we first maximize the total risk score; ties are
broken by the lexicographically smallest ascending sequence of block numbers
(compare the first differing block). When no plan exists the *blocking
condition* is reported explicitly instead of fabricating a plan.

This module is pure: it neither touches storage nor speaks HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_TARGET = 2
MAX_TARGET = 16
MIN_RISK = 0
MAX_RISK = 100
MAX_RANGES = 4

_NEG = -10**9


class AuditValidationError(ValueError):
    """The audit request itself is malformed (mapped to HTTP 400)."""


@dataclass(frozen=True)
class FocusRange:
    start: int
    end: int  # inclusive
    quota: int
    rank: int  # position in the request, 0-based

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass
class AuditPlan:
    target: int
    blocks: list[int]
    risk_sum: int
    ranges: list[FocusRange]
    quota_hits: dict[int, int]
    solvable: bool
    block_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "blocks": self.blocks,
            "risk_sum": self.risk_sum,
            "solvable": self.solvable,
            "block_reason": self.block_reason,
            "ranges": [
                {
                    "start": r.start,
                    "end": r.end,
                    "quota": r.quota,
                    "selected": self.quota_hits.get(r.rank, 0),
                }
                for r in self.ranges
            ],
        }


def _as_int(value, name: str) -> int:
    # bool is a subclass of int: reject it explicitly.
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditValidationError(f"{name} must be an integer")
    return value


def normalize_ranges(raw) -> list[FocusRange]:
    """Validate raw range dicts and return them in request order.

    Exactly 1..4 ranges are required. Indices must be canonical integers,
    0 <= start <= end, quotas >= 0, and ranges must be mutually
    non-overlapping (touching end-to-end, e.g. [0,2],[3,5], is allowed).
    """
    if not isinstance(raw, list):
        raise AuditValidationError("ranges must be a list of 1-4 focus intervals")
    if not 1 <= len(raw) <= MAX_RANGES:
        raise AuditValidationError("ranges must contain 1-4 focus intervals")

    ranges: list[FocusRange] = []
    for pos, item in enumerate(raw):
        where = f"ranges[{pos}]"
        if not isinstance(item, dict):
            raise AuditValidationError(f"{where} must be an object")
        if not {"start", "end", "quota"} <= set(item):
            raise AuditValidationError(
                f"{where} requires start, end and quota fields"
            )
        start = _as_int(item.get("start"), f"{where}.start")
        end = _as_int(item.get("end"), f"{where}.end")
        quota = _as_int(item.get("quota"), f"{where}.quota")
        if start < 0:
            raise AuditValidationError(f"{where}.start must be >= 0")
        if end < start:
            raise AuditValidationError(
                f"{where}: end {end} must be >= start {start}"
            )
        if quota < 0:
            raise AuditValidationError(f"{where}.quota must be >= 0")
        ranges.append(FocusRange(start=start, end=end, quota=quota, rank=pos))

    ordered = sorted(ranges, key=lambda r: (r.start, r.end))
    for prev, cur in zip(ordered, ordered[1:]):
        if cur.start <= prev.end:
            raise AuditValidationError(
                "focus ranges must not overlap: "
                f"[{prev.start},{prev.end}] overlaps [{cur.start},{cur.end}]"
            )
    return ranges


def validate_request(
    chunk_count: int, target, risk_scores, raw_ranges
) -> tuple[int, int, list[int], list[FocusRange]]:
    """Validate an audit request against the actual block count.

    Returns ``(n, target, scores, ranges)``.
    """
    n = _as_int(chunk_count, "chunk_count")
    target = _as_int(target, "target")
    if not MIN_TARGET <= target <= MAX_TARGET:
        raise AuditValidationError(
            f"target must be between {MIN_TARGET} and {MAX_TARGET}"
        )
    if target > n:
        raise AuditValidationError(
            f"target {target} must not exceed total block count {n}"
        )

    if not isinstance(risk_scores, list):
        raise AuditValidationError("risk_scores must be a list")
    if len(risk_scores) != n:
        raise AuditValidationError(
            f"risk_scores length {len(risk_scores)} must match actual block count {n}"
        )
    scores: list[int] = []
    for i, value in enumerate(risk_scores):
        score = _as_int(value, f"risk_scores[{i}]")
        if not MIN_RISK <= score <= MAX_RISK:
            raise AuditValidationError(
                f"risk_scores[{i}] must be between {MIN_RISK} and {MAX_RISK}"
            )
        scores.append(score)

    ranges = normalize_ranges(raw_ranges)
    for r in ranges:
        if r.end >= n:
            raise AuditValidationError(
                f"ranges[{r.rank}] end {r.end} is out of bounds for {n} blocks"
            )
        if r.quota > r.length:
            raise AuditValidationError(
                f"ranges[{r.rank}] quota {r.quota} exceeds interval length {r.length}"
            )
    return n, target, scores, ranges


def _independent_capacity(lo: int, hi: int) -> int:
    """Maximum number of pairwise non-adjacent indices in [lo, hi]."""
    width = hi - lo + 1
    return (width + 1) // 2 if width > 0 else 0


def precheck(n: int, target: int, ranges: list[FocusRange]) -> str | None:
    """Return a human-readable blocking reason, or None.

    These are cheap necessary conditions that produce a precise diagnosis;
    the dynamic program remains the authoritative feasibility check.
    """
    global_cap = _independent_capacity(0, n - 1)
    if global_cap < target:
        return (
            f"no feasible plan: {n} blocks can provide at most {global_cap} "
            f"non-adjacent selections but target is {target}"
        )

    positive = [r for r in ranges if r.quota > 0]
    quota_sum = sum(r.quota for r in positive)
    if quota_sum > target:
        return (
            "no feasible plan: focus quotas require at least "
            f"{quota_sum} blocks but target is only {target}"
        )

    for r in positive:
        cap = _independent_capacity(r.start, r.end)
        if r.quota > cap:
            return (
                f"no feasible plan: range [{r.start},{r.end}] can hold at most "
                f"{cap} non-adjacent blocks but quota is {r.quota}"
            )

    if positive:
        # Blocks serving different ranges must also be pairwise non-adjacent,
        # including across interval borders. Served ranges may be separated by
        # gaps; selections from two ranges interact only when the gap is < 2.
        # Measure the minimal span covering all ranges that demand blocks.
        span_first = min(r.start for r in positive)
        span_last = max(r.end for r in positive)
        span_cap = _independent_capacity(span_first, span_last)
        if quota_sum > span_cap:
            return (
                "no feasible plan: the required "
                f"{quota_sum} selections across ranges spanning "
                f"[{span_first},{span_last}] cannot stay pairwise non-adjacent "
                f"(span capacity {span_cap})"
            )

    return None


def solve(
    n: int, target: int, scores: list[int], ranges: list[FocusRange]
) -> AuditPlan:
    """Compute the optimal plan, or report why none exists.

    Dynamic program over blocks in ascending index order.

    State after processing blocks 0..i:

      free[k][s]   – best risk sum with block i NOT selected
      locked[k][s] – best risk sum with block i SELECTED

    where k is the number selected and s is a saturated per-range count
    vector (count_r = min(selected blocks inside range r, quota_r)). Only
    "block i free" states may take block i+1, which enforces non-adjacency.

    Ties with equal risk sums are broken afterwards by backwards greedy
    reconstruction that prefers SKIPPING the highest-indexed block whenever
    the optimum remains reachable: among ascending block sequences that
    choice makes the first differing index smallest, i.e. lexicographically
    minimal.
    """
    reason = precheck(n, target, ranges)
    if reason is not None:
        return AuditPlan(target, [], 0, ranges, {}, False, reason)

    quotas = tuple(r.quota for r in ranges)
    dims = tuple(q + 1 for q in quotas)
    state_count = 1
    for d in dims:
        state_count *= d
    goal = _encode_state(quotas, dims)

    # Range owning each block (-1 = outside every focus range). Ranges are
    # mutually non-overlapping, so at most one range owns a block.
    owner = [-1] * n
    for r in ranges:
        for i in range(r.start, r.end + 1):
            owner[i] = r.rank

    def fresh():
        return [_NEG] * state_count

    # Layer "-1" (no block processed): nothing selected and nothing locked.
    free = [fresh() for _ in range(target + 1)]
    locked = [fresh() for _ in range(target + 1)]
    free[0][0] = 0
    # history[i] = (free, locked) AFTER blocks 0..i were decided.
    history: list[tuple[list[list[int]], list[list[int]]]] = []

    for i in range(n):
        own = owner[i]
        w = scores[i]
        new_free = [fresh() for _ in range(target + 1)]
        new_locked = [fresh() for _ in range(target + 1)]
        for k in range(target + 1):
            frow, lrow = free[k], locked[k]
            skip_row = new_free[k]
            for s in range(state_count):
                sf = frow[s]
                sl = lrow[s]
                # skip i: both predecessor kinds land in "i not selected".
                best = sf if sf >= sl else sl
                if best != _NEG:
                    skip_row[s] = best
            # take i: only from free (i-1 not selected).
            if k < target:
                take_row = new_locked[k + 1]
                for s, val in enumerate(frow):
                    if val == _NEG:
                        continue
                    ns = _add_owner(s, dims, own)
                    cand = val + w
                    if cand > take_row[ns]:
                        take_row[ns] = cand
        free, locked = new_free, new_locked
        history.append((free, locked))

    final_score = max(free[target][goal], locked[target][goal])
    if final_score == _NEG:
        return AuditPlan(
            target,
            [],
            0,
            ranges,
            {},
            False,
            "no feasible plan: no non-adjacent selection of "
            f"{target} blocks satisfies every focus quota",
        )

    # --- backwards reconstruction: lexicographically smallest optimum -----
    # While walking from the last block downwards, `required` is the set of
    # saturated-count encodings that blocks 0..i must still attain, given the
    # suffix already fixed (several encodings can remain compatible once a
    # range is saturated by an extra in-range block).
    required = {goal}
    chosen: list[int] = []
    suffix_score = 0
    blocked = False  # True => block i+1 was taken, so block i is forbidden
    k = target
    for i in range(n - 1, -1, -1):
        f_layer = history[i][0]
        prefix_score = final_score - suffix_score

        # Option A: skip i. The state over 0..i-1 is unchanged; layer free at
        # i guarantees block i itself is unselected (also satisfying blocked).
        can_skip = any(
            f_layer[k][s] == prefix_score for s in required
        )
        if can_skip:
            blocked = False
            continue

        # Option B: take i. Only legal when block i+1 was not taken; the
        # predecessor state over 0..i-1 must leave block i-1 unselected.
        own = owner[i]
        predecessor_states: set[int] = set()
        if not blocked and k > 0:
            pf = history[i - 1][0] if i > 0 else _base_free(target, state_count)
            for s in required:
                for c in _predecessor_states(s, own, dims):
                    if pf[k - 1][c] + scores[i] == prefix_score:
                        predecessor_states.add(c)

        if not predecessor_states:
            # pragma: no cover - DP invariant, defended explicitly.
            raise RuntimeError("audit DP reconstruction failed")
        chosen.append(i)
        suffix_score += scores[i]
        required = predecessor_states
        k -= 1
        blocked = True

    blocks = list(reversed(chosen))
    if k != 0 or required != {0} or suffix_score != final_score:
        raise RuntimeError("audit DP reconstruction mismatch")  # pragma: no cover

    hits = {r.rank: 0 for r in ranges}
    for b in blocks:
        own = owner[b]
        if own >= 0:
            hits[own] += 1
    return AuditPlan(target, blocks, final_score, ranges, hits, True)


# A constant "layer -1": free[0][0] = 0, everything else unreachable.
_BASE: dict[tuple[int, int], list[list[int]]] = {}


def _base_free(k_max: int, state_count: int) -> list[list[int]]:
    key = (k_max, state_count)
    cached = _BASE.get(key)
    if cached is None:
        cached = [[_NEG] * state_count for _ in range(k_max + 1)]
        cached[0][0] = 0
        _BASE[key] = cached
    return cached


def _encode_state(counts: tuple[int, ...], dims: tuple[int, ...]) -> int:
    """Mixed-radix encoding of saturated per-range counts."""
    code = 0
    for c, d in zip(counts, dims):
        code = code * d + c
    return code


def _add_owner(encoded: int, dims: tuple[int, ...], owner_rank: int) -> int:
    """Encode the state after selecting a block owned by ``owner_rank``."""
    if owner_rank < 0:
        return encoded
    # Decode, bump (saturated), re-encode. States are tiny; clarity first.
    counts = _decode_state(encoded, dims)
    counts[owner_rank] = min(counts[owner_rank] + 1, dims[owner_rank] - 1)
    return _encode_state(tuple(counts), dims)


def _decode_state(encoded: int, dims: tuple[int, ...]) -> list[int]:
    counts = [0] * len(dims)
    for j in range(len(dims) - 1, -1, -1):
        counts[j] = encoded % dims[j]
        encoded //= dims[j]
    return counts


def _predecessor_states(
    encoded: int, owner_rank: int, dims: tuple[int, ...]
) -> set[int]:
    """Saturated states c over earlier blocks such that selecting a block
    owned by ``owner_rank`` reaches state ``encoded``.

    A block outside every range leaves the state unchanged. A block inside
    range r bumps its saturated count; when the target state is already at
    the quota, the predecessor may be at quota-1 (this block fills it) or
    already at quota (this block is an extra in-range selection).
    """
    if owner_rank < 0:
        return {encoded}
    counts = _decode_state(encoded, dims)
    results = set()
    cur = counts[owner_rank]
    candidates = [cur - 1]
    if cur == dims[owner_rank] - 1:
        candidates.append(cur)  # already saturated: extra in-range block
    for value in candidates:
        if value < 0:
            continue
        prev = list(counts)
        prev[owner_rank] = value
        results.add(_encode_state(tuple(prev), dims))
    return results


def build_audit_plan(
    chunk_count: int, target, risk_scores, raw_ranges
) -> AuditPlan:
    """Validate the request, then solve. Entry point for the storage layer."""
    n, target, scores, ranges = validate_request(
        chunk_count, target, risk_scores, raw_ranges
    )
    return solve(n, target, scores, ranges)
