"""HTTP smoke test for a running sealing-desk instance.

Exercises the whole contract over real HTTP (stdlib only):
  health + SPA, validation, out-of-order PUT, idempotent retransmission,
  409 conflicts that never mutate state, missing-range listing, atomic seal,
  identical receipt on repeated seal, and sealed-session immutability.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")
CHUNK = 65536


def call(method: str, path: str, body: bytes | None = None, headers=None):
    req = urllib.request.Request(
        BASE + path,
        data=body,
        method=method,
        headers=headers or {},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return exc.code, {"raw": raw.decode(errors="replace")}


def put(session, offset, data, total=None, sha=None):
    headers = {"X-Chunk-Offset": str(offset), "Content-Type": "application/octet-stream"}
    if total is not None:
        headers["X-Total-Size"] = str(total)
    if sha is not None:
        headers["X-Content-SHA256"] = sha
    return call("PUT", f"/api/uploads/{session}/chunks", data, headers)


def wait_healthy(timeout=30.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            status, _ = call("GET", "/health")
            if status == 200:
                return
        except Exception as exc:  # connection refused while starting
            last = exc
        time.sleep(0.5)
    raise SystemExit(f"service never became healthy: {last}")


def check(cond, label):
    if not cond:
        raise SystemExit(f"SMOKE FAIL: {label}")
    print(f"  ok - {label}")


def main():
    wait_healthy()
    suffix = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
    s = f"SMOKE{suffix}"

    print("[health + static]")
    status, body = call("GET", "/health")
    check(status == 200 and body.get("status") == "ok", "GET /health -> 200 ok")
    req = urllib.request.Request(BASE + "/")
    with urllib.request.urlopen(req, timeout=10) as resp:
        index = resp.read().decode()
    check(resp.status == 200 and 'id="root"' in index, "SPA index.html is served")

    print("[validation]")
    status, body = put("bad-id!", 0, b"x", 1, "a" * 64)
    check(status == 400, f"invalid session rejected (400), got {status}")
    bad_digest = put(s + "B", 0, b"x", 1, "Z" * 64)
    check(bad_digest[0] == 400, "uppercase digest rejected (400)")

    # 3-chunk file, last chunk short
    blob = os.urandom(2 * CHUNK + 123)
    digest = hashlib.sha256(blob).hexdigest()
    c0, c1, c2 = blob[:CHUNK], blob[CHUNK:2 * CHUNK], blob[2 * CHUNK:]

    print("[out of order arrival]")
    status, body = put(s, CHUNK, c1, len(blob), digest)
    check(status == 200 and body["confirmed_chunks"] == [1], "chunk #1 first -> 200")
    status, body = put(s, 2 * CHUNK, c2)
    check(status == 200 and set(body["confirmed_chunks"]) == {1, 2}, "chunk #2 -> 200")

    print("[located rejections before completion]")
    status, body = put(s, 1, b"x")
    check(status == 400 and "not aligned" in body["error"], "unaligned offset -> 400")
    status, body = put(s, 3 * CHUNK, b"x")
    check(status == 400 and "beyond" in body["error"], "out-of-bounds offset -> 400")
    status, body = put(s, 0, c0[:-1])
    check(status == 400 and "chunk length" in body["error"], "wrong chunk length -> 400")

    print("[seal with missing blocks]")
    status, body = call("POST", f"/api/uploads/{s}/seal")
    check(status == 409 and body["missing_ranges"] == [[0, 0]],
          f"missing ranges reported: {body}")

    print("[idempotent retransmission then completion]")
    status, body = put(s, CHUNK, c1, len(blob), digest)
    check(status == 200 and body["duplicate"] is True, "identical retransmit -> 200 duplicate")
    status, body = put(s, 0, c0)
    check(status == 200 and body["confirmed_chunks"] == [0, 1, 2], "chunk #0 completes upload")

    print("[409 conflict must not overwrite]")
    evil = bytearray(c1)
    evil[0] ^= 0xFF
    status, body = put(s, CHUNK, bytes(evil))
    check(status == 409, f"different bytes at same offset -> 409, got {status} {body}")
    status, body = put(s, CHUNK, c1)
    check(status == 200 and body["duplicate"] is True,
          "original chunk still intact and idempotent after 409")
    status, body = put(s, CHUNK, c1, len(blob) + 1, digest)
    check(status == 409, "changed total_size -> 409")
    status, body = put(s, CHUNK, c1, len(blob), "a" * 64)
    check(status == 409, "changed sha256 -> 409")

    print("[atomic seal + stable receipt]")
    status, receipt = call("POST", f"/api/uploads/{s}/seal")
    check(status == 200 and receipt["sha256"] == digest and receipt["chunks"] == 3,
          f"seal succeeds with correct digest: {receipt}")
    status, again = call("POST", f"/api/uploads/{s}/seal")
    check(status == 200 and again == receipt, "repeat seal returns the SAME receipt")

    print("[sealed immutability]")
    wrong0 = bytes(CHUNK)  # all-zero chunk, differs from random c0
    status, _ = put(s, 0, wrong0)
    check(status == 409, "different chunk after seal -> 409")
    status, body = put(s, 0, c0)
    check(status == 200 and body["sealed"] is True, "identical PUT after seal stays 200")

    print("[digest mismatch never seals]")
    bad = b"q" * 50
    sb = s + "D"
    status, _ = put(sb, 0, bad, len(bad), "a" * 64)
    check(status == 200, "wrong-declared digest upload accepted chunk-wise")
    status, body = call("POST", f"/api/uploads/{sb}/seal")
    check(status == 409 and "digest" in body["error"], "digest mismatch -> 409, no receipt")
    status, body = call("GET", f"/api/uploads/{sb}")
    check(status == 200 and body["sealed"] is False and body["receipt"] is None,
          "no receipt exists after digest mismatch")

    audit_smoke(s, suffix)

    print(f"\nSMOKE OK against {BASE} (sessions {s}, {sb})")


def post_json(path, payload):
    data = json.dumps(payload).encode()
    return call("POST", path, data, {"Content-Type": "application/json"})


def audit_smoke(sealed_session, suffix):
    """Audit-plan contract over HTTP on an already sealed 3-chunk session."""
    sa = f"AUD{suffix}"

    print("[audit: sealed prerequisite]")
    # A fresh sealed 5-chunk session gives room for non-adjacent samples.
    blob = bytes((i * 7) % 251 for i in range(4 * CHUNK + 17))
    digest = hashlib.sha256(blob).hexdigest()
    for i in range(5):
        part = blob[i * CHUNK:(i + 1) * CHUNK]
        st, _ = put(sa, i * CHUNK, part, len(blob), digest)
        assert st == 200, st
    st, receipt = call("POST", f"/api/uploads/{sa}/seal")
    check(st == 200 and receipt["chunks"] == 5, "5-chunk session sealed for audit")

    # An unsealed session must never produce a plan.
    su = f"AUDU{suffix}"
    put(su, 0, b"z" * 10, 10, hashlib.sha256(b"z" * 10).hexdigest())
    st, body = post_json(
        f"/api/uploads/{su}/audit-plan",
        {"target": 2, "risks": [1, 2], "zones": [{"start": 0, "end": 1, "quota": 1}]},
    )
    check(st == 409 and body.get("sealed") is False,
          "audit-plan on unsealed session -> 409, no plan")
    st, body = call("GET", f"/api/uploads/{su}")
    check(body.get("audit_plan") is None, "unsealed session keeps audit_plan=null")

    # Unknown session -> 404.
    st, _ = post_json(
        f"/api/uploads/NOPE{suffix}/audit-plan",
        {"target": 2, "risks": [1, 2], "zones": [{"start": 0, "end": 1, "quota": 1}]},
    )
    check(st == 404, "audit-plan on unknown session -> 404")

    print("[audit: located validation rejections]")
    good_zones = [{"start": 0, "end": 4, "quota": 1}]
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 1, "risks": [1] * 5, "zones": good_zones},
    )
    check(st == 400 and "between 2 and 16" in body["error"], "target < 2 -> 400")
    st, _ = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 17, "risks": [1] * 5, "zones": good_zones},
    )
    check(st == 400, "target > 16 -> 400")
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 6, "risks": [1] * 5, "zones": good_zones},
    )
    check(st == 400 and "exceeds total" in body["error"], "target > chunks -> 400")
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": [1] * 4, "zones": good_zones},
    )
    check(st == 400 and "exactly 5" in body["error"], "risk count mismatch -> 400")
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": [1, 101, 1, 1, 1], "zones": good_zones},
    )
    check(st == 400 and "between 0 and 100" in body["error"], "risk > 100 -> 400")
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": [1] * 5, "zones": [{"start": 3, "end": 1, "quota": 1}]},
    )
    check(st == 400, "start > end -> 400")
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": [1] * 5,
         "zones": [{"start": 0, "end": 2, "quota": 1}, {"start": 2, "end": 4, "quota": 1}]},
    )
    check(st == 400 and "overlap" in body["error"], "overlapping zones -> 400")
    st, _ = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": [1] * 5, "zones": []},
    )
    check(st == 400, "zero zones -> 400")

    print("[audit: feasible plan is optimal, non-adjacent, quotas met]")
    risks = [10, 95, 20, 90, 30]
    st, plan = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 2, "risks": risks,
         "zones": [{"start": 0, "end": 1, "quota": 1}, {"start": 3, "end": 4, "quota": 1}]},
    )
    check(st == 200 and plan["feasible"] is True, f"feasible plan -> 200: {plan}")
    selected = plan["selected"]
    check(len(selected) == 2, f"exactly target chunks: {selected}")
    check(all(b != a + 1 for a, b in zip(selected, selected[1:])),
          f"no adjacent selected chunks: {selected}")
    # one pick in each mandatory zone, maximizing risk -> {1, 3}
    check(selected == [1, 3], f"optimum + lexicographic selection [1,3]: {selected}")
    check(plan["risk_total"] == 95 + 90, "risk total maximized")
    check(all(z["quota_met"] for z in plan["zones"]), "every zone quota met")
    check(len(plan["per_chunk"]) == 5, "per-chunk rows present")
    rows = {row["index"]: row for row in plan["per_chunk"]}
    check(rows[0]["zone"] == 0 and rows[4]["zone"] == 1 and rows[2]["zone"] is None,
          "per-chunk zone attribution correct")
    check(rows[1]["selected"] is True and rows[1]["risk"] == 95, "per-chunk risk shown")

    print("[audit: plan is persisted next to the receipt]")
    st, body = call("GET", f"/api/uploads/{sa}")
    check(st == 200 and body["sealed"] is True and body["receipt"] is not None,
          "sealed status + receipt intact after audit planning")
    check(body.get("audit_plan", {}).get("selected") == [1, 3],
          "audit plan exposed by status query")
    check(body["audit_plan"]["session"] == sa and body["audit_plan"]["chunks"] == 5,
          "persisted plan references session/chunks")

    print("[audit: refilling conditions replaces; infeasible never fakes]")
    st, plan2 = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 3, "risks": [50] * 5, "zones": [{"start": 0, "end": 4, "quota": 1}]},
    )
    check(st == 200 and len(plan2["selected"]) == 3, "regeneration accepted")
    st, body = call("GET", f"/api/uploads/{sa}")
    check(body["audit_plan"]["target"] == 3, "newest plan replaces the old one")

    # Infeasible: 5 chunks cannot yield 4 pairwise non-adjacent picks (cap 3).
    st, body = post_json(
        f"/api/uploads/{sa}/audit-plan",
        {"target": 4, "risks": [1] * 5, "zones": [{"start": 0, "end": 4, "quota": 1}]},
    )
    check(st == 409 and body.get("feasible") is False and body.get("blocking"),
          f"infeasible -> 409 with blocking conditions: {body}")
    check(any(b["type"] == "global_capacity" for b in body["blocking"]),
          "global capacity blocker named")
    st, body = call("GET", f"/api/uploads/{sa}")
    check(body["audit_plan"]["target"] == 3,
          "infeasible request did not overwrite the prior plan")

    print("[audit: forced adjacent zones block with located reason]")
    # Two touching, saturated odd zones: [0,2] quota 2 forces {0,2} and
    # [3,5] quota 2 forces {3,5}; chunks 2 and 3 collide. n=7 keeps the
    # global non-adjacency capacity (4) >= target so this is the blocking
    # reason, not a capacity shortfall.
    s7 = f"AUD7{suffix}"
    blob7 = b"k" * (6 * CHUNK + 1)
    d7 = hashlib.sha256(blob7).hexdigest()
    for i in range(7):
        put(s7, i * CHUNK, blob7[i * CHUNK:(i + 1) * CHUNK], len(blob7), d7)
    call("POST", f"/api/uploads/{s7}/seal")
    st, body = post_json(
        f"/api/uploads/{s7}/audit-plan",
        {"target": 4, "risks": [1] * 7,
         "zones": [{"start": 0, "end": 2, "quota": 2},
                   {"start": 3, "end": 5, "quota": 2}]},
    )
    check(st == 409 and any(b["type"] == "zone_boundary_conflict"
                            for b in body.get("blocking", [])),
          f"forced-adjacent zones -> 409 boundary blocker: {body}")
    st, status7 = call("GET", f"/api/uploads/{s7}")
    check(status7.get("audit_plan") is None, "no fabricated plan for infeasible session")


if __name__ == "__main__":
    main()
