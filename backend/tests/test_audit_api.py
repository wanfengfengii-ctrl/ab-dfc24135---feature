"""HTTP tests for POST /api/uploads/{session}/audit-plan."""

from __future__ import annotations

import hashlib
import os

import pytest
from fastapi.testclient import TestClient

from app import main as web
from app.storage import CHUNK_SIZE, UploadStore


@pytest.fixture()
def client(tmp_path):
    web.store = UploadStore(str(tmp_path / "data"))
    return TestClient(web.app)


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _upload_and_seal(client, session="aud1", blob: bytes | None = None):
    blob = blob if blob is not None else os.urandom(2 * CHUNK_SIZE + 1)
    sha = _digest(blob)
    count = (len(blob) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(count):
        part = blob[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        r = client.put(
            f"/api/uploads/{session}/chunks",
            content=part,
            headers={
                "X-Chunk-Offset": str(i * CHUNK_SIZE),
                "X-Total-Size": str(len(blob)),
                "X-Content-SHA256": sha,
            },
        )
        assert r.status_code == 200, r.text
    r = client.post(f"/api/uploads/{session}/seal")
    assert r.status_code == 200, r.text
    return blob, sha


def _plan(session, **overrides):
    body = {
        "target": 2,
        "risk_scores": [10, 90, 10],
        "ranges": [{"start": 0, "end": 2, "quota": 1}],
    }
    body.update(overrides)
    return body


def test_plan_happy_path(client):
    _upload_and_seal(client, "aud1")
    r = client.post("/api/uploads/aud1/audit-plan", json=_plan("aud1"))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["solvable"] is True
    # 3 blocks, target 2: only {0,2} is non-adjacent (sum 20)
    assert out["blocks"] == [0, 2]
    assert out["risk_sum"] == 20
    quota = out["ranges"][0]
    assert quota["start"] == 0 and quota["quota"] == 1
    assert quota["selected"] == 2
    assert out["chunk_count"] == 3
    assert out["receipt_id"].startswith("sha256:")


def test_plan_persisted_and_visible_in_status(client):
    _upload_and_seal(client, "persist")
    r = client.post(
        "/api/uploads/persist/audit-plan", json=_plan("persist", risk_scores=[5, 5, 5])
    )
    assert r.status_code == 200
    blocks = r.json()["blocks"]

    status = client.get("/api/uploads/persist").json()
    assert status["sealed"] is True
    assert status["audit_plan"]["blocks"] == blocks


def test_plan_survives_restart(client, tmp_path):
    data_dir = tmp_path / "data"
    _upload_and_seal(client, "restart")
    r = client.post("/api/uploads/restart/audit-plan", json=_plan("restart"))
    assert r.status_code == 200
    blocks = r.json()["blocks"]

    web.store = UploadStore(str(data_dir))  # simulated restart
    status = client.get("/api/uploads/restart").json()
    assert status["audit_plan"]["blocks"] == blocks


def test_unsealed_session_is_409_and_no_plan_file(client, tmp_path):
    # upload chunks but never seal
    blob = os.urandom(CHUNK_SIZE + 1)
    client.put(
        "/api/uploads/unsealed/chunks",
        content=blob[:CHUNK_SIZE],
        headers={
            "X-Chunk-Offset": "0",
            "X-Total-Size": str(len(blob)),
            "X-Content-SHA256": _digest(blob),
        },
    )
    r = client.post(
        "/api/uploads/unsealed/audit-plan",
        json=_plan("unsealed", risk_scores=[0, 0]),
    )
    assert r.status_code == 409
    assert "not sealed" in r.json()["error"]
    assert not (tmp_path / "data" / "unsealed" / "audit-plan.json").exists()


def test_unknown_session_is_400(client):
    r = client.post("/api/uploads/ghost/audit-plan", json=_plan("ghost"))
    assert r.status_code == 400
    assert "unknown session" in r.json()["error"]


def test_invalid_session_id_is_400(client):
    r = client.post("/api/uploads/bad-id/audit-plan", json=_plan("bad-id"))
    assert r.status_code == 400


@pytest.mark.parametrize(
    "field,value,fragment",
    [
        ("target", 1, "between 2 and 16"),
        ("target", 17, "between 2 and 16"),
        ("target", 4, "must not exceed"),
        ("risk_scores", [0, 0], "must match"),
        ("risk_scores", [0, 0, 101], "between 0 and 100"),
        ("ranges", [], "1-4"),
        ("ranges", [{"start": 0, "end": 2, "quota": 4}], "exceeds interval"),
    ],
)
def test_malformed_requests_are_located_400(client, field, value, fragment):
    _upload_and_seal(client, "val")
    body = _plan("val")
    body[field] = value
    r = client.post("/api/uploads/val/audit-plan", json=body)
    assert r.status_code == 400, r.text
    assert fragment in r.json()["error"]


def test_overlapping_ranges_are_located_400(client):
    _upload_and_seal(client, "ov")
    body = _plan("ov")
    body["ranges"] = [
        {"start": 0, "end": 1, "quota": 1},
        {"start": 1, "end": 2, "quota": 1},
    ]
    r = client.post("/api/uploads/ov/audit-plan", json=body)
    assert r.status_code == 400
    assert "must not overlap" in r.json()["error"]


def test_range_index_out_of_bounds_is_400(client):
    _upload_and_seal(client, "oob")
    body = _plan("oob")
    body["ranges"] = [{"start": 0, "end": 3, "quota": 0}]
    r = client.post("/api/uploads/oob/audit-plan", json=body)
    assert r.status_code == 400
    assert "out of bounds" in r.json()["error"]


def test_target_larger_than_blocks_is_400(client):
    blob = b"a"
    client.put(
        "/api/uploads/one/chunks",
        content=blob,
        headers={
            "X-Chunk-Offset": "0",
            "X-Total-Size": "1",
            "X-Content-SHA256": _digest(blob),
        },
    )
    client.post("/api/uploads/one/seal")
    body = {
        "target": 2,
        "risk_scores": [50],
        "ranges": [{"start": 0, "end": 0, "quota": 0}],
    }
    r = client.post("/api/uploads/one/audit-plan", json=body)
    assert r.status_code == 400, r.text
    assert "must not exceed" in r.json()["error"]


def test_infeasible_quota_returns_200_blocking_condition(client, tmp_path):
    # two-chunk package (2 * 64KiB), quota 2 on the whole interval forces
    # selecting both adjacent blocks -> impossible
    blob = os.urandom(2 * CHUNK_SIZE)
    _upload_and_seal(client, "two", blob)
    body = {
        "target": 2,
        "risk_scores": [50, 60],
        "ranges": [{"start": 0, "end": 1, "quota": 2}],
    }
    r = client.post("/api/uploads/two/audit-plan", json=body)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["solvable"] is False
    assert out["blocks"] == []
    assert out["block_reason"].startswith("no feasible plan")
    assert not (tmp_path / "data" / "two" / "audit-plan.json").exists()


def test_refilling_after_infeasible_replaces_old_plan(client):
    _upload_and_seal(client, "repl")  # default 3-chunk package
    good = {
        "target": 2,
        "risk_scores": [5, 6, 7],
        "ranges": [{"start": 0, "end": 2, "quota": 0}],
    }
    r = client.post("/api/uploads/repl/audit-plan", json=good)
    assert r.status_code == 200 and r.json()["solvable"] is True

    # infeasible resubmission must not keep the stale plan around
    bad = dict(good)
    # quota 2 inside the adjacent pair [0,1] cannot be met non-adjacently
    bad["ranges"] = [{"start": 0, "end": 1, "quota": 2}]
    r = client.post("/api/uploads/repl/audit-plan", json=bad)
    assert r.status_code == 200 and r.json()["solvable"] is False
    status = client.get("/api/uploads/repl").json()
    assert status["audit_plan"] is None

    # and a feasible resubmission creates a plan again
    r = client.post("/api/uploads/repl/audit-plan", json=good)
    assert r.status_code == 200 and r.json()["solvable"] is True
    assert client.get("/api/uploads/repl").json()["audit_plan"] is not None


def test_body_must_be_json_object(client):
    _upload_and_seal(client, "json")
    r = client.post("/api/uploads/json/audit-plan", json=[1, 2, 3])
    assert r.status_code == 400
    r = client.post(
        "/api/uploads/json/audit-plan",
        content=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 400


def test_existing_upload_and_seal_semantics_unchanged(client):
    # regression: audit feature must not disturb status/seal shapes
    blob = os.urandom(CHUNK_SIZE)
    _upload_and_seal(client, "reg", blob)
    status = client.get("/api/uploads/reg").json()
    assert status["sealed"] is True
    assert status["receipt"]["chunks"] == 1
    assert "audit_plan" in status  # new field, null before any plan
    assert status["audit_plan"] is None
    # duplicate seal returns the same receipt
    again = client.post("/api/uploads/reg/seal").json()
    assert again == status["receipt"]
