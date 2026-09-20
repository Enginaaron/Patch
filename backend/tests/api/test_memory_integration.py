"""aa_dev memory routes share the rover's candidate and mission lifecycle."""
from sqlmodel import select

from app.db import get_session
from app.models import Search
from app.config import settings
from app.rover.vision import LiveVision
from tests.api.helpers import create_search, target_in_view
from tests.rover.helpers import fast_config


def test_confirm_alias_saves_find_and_exposes_memory(client, rover, tap):
    rover(*target_in_view())
    created = create_search(client, "my bottle")
    sid = created["search_id"]
    cid = tap.wait_type(sid, "candidate_found")["payload"]["candidate_id"]
    response = client.post(f"/api/candidates/{cid}/confirm")
    assert response.status_code == 200
    assert response.json()["find"]["crop_url"]
    item = client.get(f"/api/items/{created['item_id']}").json()
    assert item["finds"]
    assert any(f["item_id"] == created["item_id"] for f in client.get("/api/finds").json())


def test_find_again_busy_does_not_leave_orphan_search(client, rover, tap):
    rover(*target_in_view())
    created = create_search(client, "my bottle")
    tap.wait_type(created["search_id"], "candidate_found")
    with get_session() as session:
        before = len(session.exec(select(Search)).all())
    response = client.post(f"/api/items/{created['item_id']}/search")
    assert response.status_code == 409
    with get_session() as session:
        assert len(session.exec(select(Search)).all()) == before


def test_quick_check_missing_item(client):
    assert client.post("/api/items/missing/quick-check").status_code == 404


def test_blurry_frame_never_calls_model(monkeypatch):
    import numpy as np
    import pytest
    from app.rover.types import FramePacket, IdentifyRequest
    monkeypatch.setattr(settings, "omni_api_key", "test")
    monkeypatch.setattr(settings, "blur_variance_threshold", 60)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    calls = []
    vision = LiveVision(fast_config(), detect_text_only=lambda *a, **kw: calls.append(1))
    with pytest.raises(RuntimeError, match="blurry"):
        vision.identify(FramePacket(frame=frame, seq=1, captured_at=0, captured_wall=0), IdentifyRequest(target_text="bottle"))
    assert not calls
