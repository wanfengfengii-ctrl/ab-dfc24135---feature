"""Sampling audit-plan domain logic for sealed cryo-EM packages.

Given one 0..100 risk score per chunk, a target sample size and 1-4
mutually disjoint priority zones (each with a minimum quota), the plan
must select exactly ``target`` chunks so that:

* no two selected chunks are adjacent;
* every zone contributes at least its quota;
* the total risk score is maximal among all feasible selections;
* ties are broken by the lexicographically smallest ascending sequence
  of chunk indices.

Optimization is a dynamic program over the chunk line (the real chunk
count is at most 128 for an 8 MiB package), so feasibility and
optimality are exact - no combination enumeration, no fabricated plans.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MIN_TARGET = 2
MAX_TARGET = 16

NEG = -(10 ** 18)


class AuditPlanError(Exception):
    """Invalid request (4xx) or an infeasible plan (409)."""

    def __init__(self, status_code: int, error: str, **extra: object) -> None:
        super().__init__(error)
        self.status_code = status_code
        self.error = error
        self.extra = extra


@dataclass(frozen=True)
class Zone:
    start: int
    end: int
    quota: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    @property
    def capacity(self) -> int:
        """Maximum mutually non-adjacent picks inside the zone."""
        return (self.length + 1) // 2


def _as_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditPlanError(400, f"{where} must be an integer")
    return value


def _validate(chunk_count: int, body: object) -> tuple[int, list[int], list[Zone]]:
    if not isinstance(body, dict):
        raise AuditPlanError(400, "request body must be a JSON object")
    if "target" not in body:
        raise AuditPlanError(400, "target is required")
    if "risks" not in body:
        raise AuditPlanError(400, "risks is required")
    if "zones" not in body:
        raise AuditPlanError(400, "zones is required")

    target = _as_int(body["target"], "target")
    if not MIN_TARGET <= target <= MAX_TARGET:
        raise AuditPlanError(
            400, f"target must be between {MIN_TARGET} and {MAX_TARGET}"
        )
    if target > chunk_count:
        raise AuditPlanError(
            400, f"target {target} exceeds total chunk count {chunk_count}"
        )

    risks_raw = body["risks"]
    if not isinstance(risks_raw, list):
        raise AuditPlanError(400, "risks must be an array")
    if len(risks_raw) != chunk_count:
        raise AuditPlanError(
            400,
            f"risks must contain exactly {chunk_count} scores, one per chunk "
            f"(got {len(risks_raw)})",
        )
    risks: list[int] = []
    for i, value in enumerate(risks_raw):
        score = _as_int(value, f"risks[{i}]")
        if not 0 <= score <= 100:
            raise AuditPlanError(
                400, f"risks[{i}] must be between 0 and 100 (got {score})"
            )
        risks.append(score)

    zones_raw = body["zones"]
    if not isinstance(zones_raw, list):
        raise AuditPlanError(400, "zones must be an array")
    if not 1 <= len(zones_raw) <= 4:
        raise AuditPlanError(400, "zones must contain between 1 and 4 entries")

    zones: list[Zone] = []
    for zi, raw in enumerate(zones_raw):
        if not isinstance(raw, dict):
            raise AuditPlanError(400, f"zones[{zi}] must be an object")
        for field in ("start", "end", "quota"):
            if field not in raw:
                raise AuditPlanError(400, f"zones[{zi}].{field} is required")
        start = _as_int(raw["start"], f"zones[{zi}].start")
        end = _as_int(raw["end"], f"zones[{zi}].end")
        quota = _as_int(raw["quota"], f"zones[{zi}].quota")
        if not 0 <= start <= end < chunk_count:
            raise AuditPlanError(
                400,
                f"zones[{zi}] is out of range: require 0 <= start <= end < "
                f"{chunk_count} (got [{start}, {end}])",
            )
        if quota < 1:
            raise AuditPlanError(
                400, f"zones[{zi}].quota must be at least 1 (got {quota})"
            )
        if quota > end - start + 1:
            raise AuditPlanError(
                400,
                f"zones[{zi}].quota {quota} exceeds the zone length "
                f"{end - start + 1}",
            )
        zones.append(Zone(start=start, end=end, quota=quota))

    zones.sort(key=lambda z: (z.start, z.end))
    for prev, nxt in zip(zones, zones[1:]):
        if nxt.start <= prev.end:
            raise AuditPlanError(
                400,
                f"zones overlap: [{prev.start}, {prev.end}] and "
                f"[{nxt.start}, {nxt.end}]",
            )
    return target, risks, zones


def _zone_map(n: int, zones: list[Zone]) -> list[int]:
    """zone_of[i] = sorted zone index owning chunk i, or -1."""
    zone_of = [-1] * n
    for zi, z in enumerate(zones):
        for i in range(z.start, z.end + 1):
            zone_of[i] = zi
    return zone_of


def _optimizer(
    n: int,
    risks: list[int],
    target: int,
    zones: list[Zone],
    zone_of: list[int],
    exact: bool,
):
    """Return a memoized F(i, k, vec, last) -> best achievable risk sum.

    k is the number of picks so far, vec the per-zone capped pick counts
    and last says chunk i-1 was picked.  ``exact`` requires exactly
    ``target`` picks at the end; otherwise at most ``target`` (used to
    attribute infeasibility).
    """
    m = len(zones)
    zero = (0,) * m
    memo: dict[tuple, int] = {}

    def f(i: int, k: int, vec: tuple[int, ...], last: bool) -> int:
        if k > target:
            return NEG
        free_positions = n - i - (1 if last else 0)
        # Not enough slots left to ever reach the target count.
        if exact and target - k > (free_positions + 1) // 2:
            return NEG
        key = (i, k, vec, last)
        cached = memo.get(key)
        if cached is not None:
            return cached

        if i == n:
            quotas_met = all(vec[j] >= zones[j].quota for j in range(m))
            done = k == target if exact else k <= target
            value = 0 if quotas_met and done else NEG
            memo[key] = value
            return value

        # Skip chunk i; afterwards the previous chunk is no longer adjacent.
        best = f(i + 1, k, vec, False)

        if not last:
            zi = zone_of[i]
            if zi >= 0 and vec[zi] < zones[zi].quota:
                nv = list(vec)
                nv[zi] += 1
                nv_tuple: tuple[int, ...] = tuple(nv)
            else:
                nv_tuple = vec
            tail = f(i + 1, k + 1, nv_tuple, True)
            # Keep NEG contagious: adding risk must not un-block a dead tail.
            take = NEG if tail <= NEG // 2 else risks[i] + tail
            if take > best:
                best = take

        memo[key] = best
        return best

    return f, zero, memo


def _blocking_conditions(
    n: int, target: int, zones: list[Zone]
) -> Optional[list[dict]]:
    """Structural, always-applicable reasons for infeasibility (or None)."""
    blocking: list[dict] = []

    capacity = (n + 1) // 2
    if target > capacity:
        blocking.append(
            {
                "type": "global_capacity",
                "message": (
                    f"non-adjacency allows at most {capacity} of {n} chunks, "
                    f"but target is {target}"
                ),
                "max_selectable": capacity,
                "target": target,
            }
        )

    quota_sum = sum(z.quota for z in zones)
    if quota_sum > target:
        blocking.append(
            {
                "type": "quota_sum",
                "message": (
                    f"zone quotas require at least {quota_sum} chunks, "
                    f"exceeding target {target}"
                ),
                "quota_sum": quota_sum,
                "target": target,
            }
        )

    for zi, z in enumerate(zones):
        if z.quota > z.capacity:
            blocking.append(
                {
                    "type": "zone_capacity",
                    "message": (
                        f"zone[{zi}] [{z.start}, {z.end}] holds at most "
                        f"{z.capacity} non-adjacent chunks, quota is {z.quota}"
                    ),
                    "zone_index": zi,
                    "start": z.start,
                    "end": z.end,
                    "quota": z.quota,
                    "capacity": z.capacity,
                }
            )

    return blocking or None


def _boundary_conflicts(zones: list[Zone]) -> list[dict]:
    """Adjacent zone pairs whose quotas force selecting touching blocks."""
    conflicts: list[dict] = []

    def forces_endpoints(z: Zone) -> bool:
        # An odd-length zone saturated to its capacity forces both
        # endpoints: only the pattern s, s+2, ..., e reaches the capacity.
        return z.length % 2 == 1 and z.quota == z.capacity

    for zi, z in enumerate(zones[:-1]):
        nxt = zones[zi + 1]
        if nxt.start == z.end + 1 and forces_endpoints(z) and forces_endpoints(nxt):
            conflicts.append(
                {
                    "type": "zone_boundary_conflict",
                    "message": (
                        f"zones [{z.start}, {z.end}] and [{nxt.start}, {nxt.end}] "
                        f"force adjacent chunks {z.end} and {nxt.start}"
                    ),
                    "left": {"start": z.start, "end": z.end, "quota": z.quota},
                    "right": {"start": nxt.start, "end": nxt.end, "quota": nxt.quota},
                }
            )
    return conflicts


def build_audit_plan(chunk_count: int, body: object) -> dict:
    """Validate, solve and serialize an audit plan or raise AuditPlanError."""
    target, risks, zones = _validate(chunk_count, body)
    n = chunk_count
    zone_of = _zone_map(n, zones)

    structural = _blocking_conditions(n, target, zones)
    if structural is not None:
        raise AuditPlanError(
            409,
            "no feasible audit plan under the given conditions",
            feasible=False,
            blocking=structural,
        )

    f, zero, _memo = _optimizer(n, risks, target, zones, zone_of, exact=True)
    optimum = f(0, 0, zero, False)
    if optimum <= NEG:
        # Attribute the failure with a quotas-only feasibility pass.
        g, _zero2, _memo2 = _optimizer(n, risks, target, zones, zone_of, exact=False)
        blocking: list[dict]
        if g(0, 0, zero, False) <= NEG:
            blocking = _boundary_conflicts(zones)
            blocking.append(
                {
                    "type": "no_feasible_combination",
                    "message": (
                        "mandatory zone selections cannot all be made without "
                        "selecting adjacent chunks"
                    ),
                }
            )
        else:
            blocking = [
                {
                    "type": "target_unreachable",
                    "message": (
                        f"zone quotas can be met, but no non-adjacent selection "
                        f"reaches exactly {target} chunks"
                    ),
                    "target": target,
                }
            ]
        raise AuditPlanError(
            409,
            "no feasible audit plan under the given conditions",
            feasible=False,
            blocking=blocking,
        )

    # Reconstruct the lexicographically smallest optimum: at each chunk a
    # pick is preferred over a skip whenever it stays on an optimal path,
    # because taking the smaller index first minimizes the sequence.
    selected: list[int] = []
    i, k, vec, last = 0, 0, zero, False
    remaining = optimum
    while i < n:
        take_tail = NEG
        take_vec: Optional[tuple[int, ...]] = None
        if not last and k < target:
            zi = zone_of[i]
            if zi >= 0 and vec[zi] < zones[zi].quota:
                nv = list(vec)
                nv[zi] += 1
                take_vec = tuple(nv)
            else:
                take_vec = vec
            take_tail = f(i + 1, k + 1, take_vec, True)

        if (
            take_vec is not None
            and take_tail > NEG // 2
            and risks[i] + take_tail == remaining
        ):
            selected.append(i)
            k += 1
            vec = take_vec
            last = True
            remaining = take_tail
        else:
            last = False
            remaining = f(i + 1, k, vec, False)
        i += 1

    selected_set = set(selected)
    zone_results = []
    for zi, z in enumerate(zones):
        picked = [i for i in selected if z.start <= i <= z.end]
        zone_results.append(
            {
                "start": z.start,
                "end": z.end,
                "quota": z.quota,
                "selected": picked,
                "selected_count": len(picked),
                "quota_met": len(picked) >= z.quota,
            }
        )

    per_chunk = [
        {
            "index": i,
            "risk": risks[i],
            "zone": zone_of[i] if zone_of[i] >= 0 else None,
            "selected": i in selected_set,
        }
        for i in range(n)
    ]

    return {
        "feasible": True,
        "chunks": n,
        "target": target,
        "selected": selected,
        "risk_total": sum(risks[i] for i in selected),
        "zones": zone_results,
        "per_chunk": per_chunk,
    }
