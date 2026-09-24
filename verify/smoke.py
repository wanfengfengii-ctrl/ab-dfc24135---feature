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

    print("[audit plan: unsealed sessions are blocked]")
    plan_req = json.dumps(
        {"target": 2, "risk_scores": [1, 2, 3],
         "ranges": [{"start": 0, "end": 2, "quota": 1}]}
    ).encode()
    status, body = call("POST", f"/api/uploads/{sb}/audit-plan", plan_req,
                        {"Content-Type": "application/json"})
    check(status == 409 and "not sealed" in body["error"],
          f"audit plan on unsealed session -> 409, got {status} {body}")
    status, body = call("POST", f"/api/uploads/NOPE9999/audit-plan", plan_req,
                        {"Content-Type": "application/json"})
    check(status == 400 and "unknown session" in body["error"],
          "audit plan on unknown session -> 400")

    print("[audit plan: malformed requests are located 400s]")
    status, body = call(
        "POST", f"/api/uploads/{s}/audit-plan",
        json.dumps({"target": 2, "risk_scores": [1, 2],
                    "ranges": [{"start": 0, "end": 2, "quota": 1}]}).encode(),
        {"Content-Type": "application/json"})
    check(status == 400 and "must match actual block count 3" in body["error"],
          f"risk score count mismatch -> 400 located, got {status} {body}")
    status, body = call(
        "POST", f"/api/uploads/{s}/audit-plan",
        json.dumps({"target": 17, "risk_scores": [1, 2, 3],
                    "ranges": [{"start": 0, "end": 2, "quota": 1}]}).encode(),
        {"Content-Type": "application/json"})
    check(status == 400 and "between 2 and 16" in body["error"],
          "target out of 2..16 -> 400")
    status, body = call(
        "POST", f"/api/uploads/{s}/audit-plan",
        json.dumps({"target": 2, "risk_scores": [1, 2, 3],
                    "ranges": [{"start": 0, "end": 1, "quota": 1},
                               {"start": 1, "end": 2, "quota": 1}]}).encode(),
        {"Content-Type": "application/json"})
    check(status == 400 and "must not overlap" in body["error"],
          "overlapping focus ranges -> 400 located")

    print("[audit plan: optimal, non-adjacent, quota-satisfying]")
    # 3 blocks; target 2 leaves only {0,2}; scores make the intent explicit.
    req = {"target": 2, "risk_scores": [10, 90, 10],
           "ranges": [{"start": 0, "end": 2, "quota": 1}]}
    status, body = call("POST", f"/api/uploads/{s}/audit-plan",
                        json.dumps(req).encode(),
                        {"Content-Type": "application/json"})
    check(status == 200 and body["solvable"] is True,
          f"audit plan created: {body}")
    check(body["blocks"] == [0, 2] and body["risk_sum"] == 20,
          f"only legal non-adjacent pair chosen, got {body.get('blocks')}")
    check(len(body["blocks"]) == body["target"] == 2, "exactly target blocks")
    check(all(b - a >= 2 for a, b in zip(body["blocks"], body["blocks"][1:])),
          "no two selected blocks are adjacent")
    check(body["ranges"][0]["selected"] >= body["ranges"][0]["quota"],
          "focus quota met and accounted")
    # persisted and surfaced by the status endpoint
    status, got = call("GET", f"/api/uploads/{s}")
    check(status == 200 and got["audit_plan"] is not None
          and got["audit_plan"]["blocks"] == [0, 2],
          "audit plan persisted and returned by status")

    print("[audit plan: infeasible conditions block without a fake plan]")
    # quota 2 inside the adjacent pair [0,1] cannot be met non-adjacently
    bad_req = {"target": 2, "risk_scores": [10, 90, 10],
               "ranges": [{"start": 0, "end": 1, "quota": 2}]}
    status, body = call("POST", f"/api/uploads/{s}/audit-plan",
                        json.dumps(bad_req).encode(),
                        {"Content-Type": "application/json"})
    check(status == 200 and body["solvable"] is False
          and body["blocks"] == [] and "no feasible plan" in body["block_reason"],
          f"infeasible -> 200 with explicit blocking condition, got {body}")
    status, got = call("GET", f"/api/uploads/{s}")
    check(got["audit_plan"] is None,
          "the infeasible resubmission removed the previously stored plan")

    print(f"\nSMOKE OK against {BASE} (sessions {s}, {sb})")


if __name__ == "__main__":
    main()
