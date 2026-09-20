"""POST /api/searches: comprehension first, one search at a time, replace on
request. Motion and detections are SIMULATED."""

import io

from PIL import Image
from sqlmodel import select

from app.db import get_session
from app.models import Item, Search, SearchStatus
from app.services.comprehension import (
    ResolvedTarget,
    clear_comprehension_provider,
    register_comprehension_provider,
)
from tests.api.helpers import create_search, detail, nothing_to_find, search_count, target_in_view


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 120, 200)).save(buffer, format="PNG")
    return buffer.getvalue()


def _items_named(name: str) -> int:
    with get_session() as session:
        return len(session.exec(select(Item).where(Item.name == name)).all())


def test_create_starts_the_mission_and_the_detail_reports_the_body(client, rover, tap, media_root):
    h = rover(*target_in_view())

    created = create_search(client, "Find my bottle beside my backpack")
    sid = created["search_id"]
    assert created["status"] == "SEARCHING" and created["target_text"] == "Find my bottle beside my backpack"

    found = tap.wait_type(sid, "candidate_found")["payload"]
    tap.wait_phase(sid, "waiting_for_confirmation")
    body = detail(client, sid)

    # Recognition lifecycle and the rover body are reported separately.
    assert body["status"] == "CANDIDATE_PENDING"
    movement = body["movement"]
    assert (movement["phase"], movement["active"], movement["search_id"]) == ("waiting_for_confirmation", True, sid)
    assert movement["message"] == "Is this your bottle beside your backpack?"
    assert (movement["vision_mode"], movement["driver"]) == ("simulation", "sim")
    assert movement["category"] == "bottle" and movement["candidates_presented"] == 1
    assert body["terminal"] is False and body["can_resume"] is False and body["accepted_candidate"] is None

    # The target phrase decides the category; the landmark is context only.
    assert body["resolved_target"] == {"category": "bottle", "landmarks": ["backpack"], "resolver": "rule_based_fallback"}

    pending = body["pending_candidate"]
    assert pending["candidate_id"] == found["candidate_id"]
    assert pending["description"] == "a blue water bottle (simulated)"
    assert pending["image_url"] == found["image_url"] and pending["crop_url"] == found["crop_url"]
    assert pending["image_url"] != pending["crop_url"]
    assert pending["box"] == found["box"] and len(pending["box"]) == 4
    for url in (pending["image_url"], pending["crop_url"]):
        assert (media_root / url.removeprefix("/media/")).is_file()

    # It was straight ahead: found on the first look, without a single pulse.
    assert h.drive.moves == [] and h.controller.active_search_id() == sid


def test_blank_text_and_too_many_images_are_refused_before_anything_happens(client, rover):
    h = rover(*target_in_view())
    before = search_count()

    assert client.post("/api/searches", data={"target_text": "   "}).status_code == 400
    too_many = [("reference_images", (f"{i}.png", _png_bytes(), "image/png")) for i in range(4)]
    assert client.post("/api/searches", data={"target_text": "my bottle"}, files=too_many).status_code == 400

    assert search_count() == before and h.controller.active_search_id() is None and h.drive.records == []


def test_reference_images_are_saved_with_the_search(client, rover, tap, media_root):
    rover(*target_in_view())
    files = [("reference_images", ("mine.png", _png_bytes(), "image/png"))]
    response = client.post("/api/searches", data={"target_text": "my striped bottle"}, files=files)
    assert response.status_code == 200, response.text
    created = response.json()
    assert created["reference_count"] == 1

    references = detail(client, created["search_id"])["reference_images"]
    assert len(references) == 1
    assert (media_root / references[0]["url"].removeprefix("/media/")).is_file()


def test_a_file_that_is_not_an_image_creates_no_search(client, rover):
    h = rover(*target_in_view())
    before = search_count()
    files = [("reference_images", ("notes.png", b"definitely not a png", "image/png"))]

    response = client.post("/api/searches", data={"target_text": "my dotted bottle"}, files=files)

    assert response.status_code == 400
    assert search_count() == before and h.controller.active_search_id() is None


def test_an_unresolvable_reference_asks_instead_of_searching(client, rover, tap):
    h = rover(*target_in_view())
    first = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(first, "waiting_for_confirmation")
    before_rows, before_seq = search_count(), detail(client, first)["movement"]["seq"]
    before_records = len(h.drive.records)

    response = client.post("/api/searches", data={"target_text": "the other one", "replace_active": "true"})

    assert response.status_code == 422
    problem = response.json()["detail"]
    assert problem["code"] == "needs_clarification" and problem["raw_text"] == "the other one"
    assert problem["question"].startswith("Which item do you mean?")
    # No search was created and the rover was left alone -- even though the
    # request said replace_active: nothing may be replaced by a non-request.
    assert search_count() == before_rows and _items_named("the other one") == 0
    assert h.controller.active_search_id() == first
    still = detail(client, first)
    assert still["status"] == "CANDIDATE_PENDING" and still["movement"]["seq"] == before_seq
    assert len(h.drive.records) == before_records


def test_a_second_search_is_refused_while_the_rover_is_busy(client, rover, tap):
    h = rover(*target_in_view())
    first = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(first, "waiting_for_confirmation")   # still active: the wheels are its to move
    before = search_count()

    response = client.post("/api/searches", data={"target_text": "my silver laptop"})

    assert response.status_code == 409
    problem = response.json()["detail"]
    assert problem["code"] == "rover_busy"
    assert (problem["active_search_id"], problem["active_target_text"]) == (first, "my bottle")
    assert "my bottle" in problem["message"]
    # NO rows: no search, not even the item.
    assert search_count() == before and _items_named("my silver laptop") == 0
    assert h.controller.active_search_id() == first
    assert detail(client, first)["status"] == "CANDIDATE_PENDING"
    assert tap.events(first, "search_cancelled") == []


def test_replace_active_cancels_the_old_search_and_starts_the_new_one(client, rover, tap):
    h = rover(*nothing_to_find())
    old = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)                     # the old mission is really turning the wheels

    new = create_search(client, "my laptop", replace_active="true")["search_id"]

    # Exactly like a user cancel, with the reason saying why.
    cancelled = tap.wait_type(old, "search_cancelled")
    assert cancelled == {"type": "search_cancelled", "payload": {"status": "CANCELLED", "reason": "replaced"}, "terminal": True}
    old_detail = detail(client, old)
    assert old_detail["status"] == "CANCELLED" and old_detail["terminal"] is True and old_detail["can_resume"] is False
    assert (old_detail["movement"]["phase"], old_detail["movement"]["reason"]) == ("stopped", "replaced")
    assert old_detail["movement"]["active"] is False

    assert h.controller.active_search_id() == new
    new_detail = detail(client, new)
    assert new_detail["status"] == "SEARCHING" and new_detail["movement"]["active"] is True
    assert new_detail["resolved_target"]["category"] == "laptop"


def test_losing_the_race_for_the_rover_leaves_no_rows_behind(client, rover, tap, monkeypatch):
    """The busy check says "idle", and in the instant before start_search
    another search claims the rover. The answer is still 409 with nothing
    created -- the half-made search row is removed again."""
    h = rover(*target_in_view())
    first = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(first, "waiting_for_confirmation")
    before = search_count()
    snapshot = h.controller.snapshot
    monkeypatch.setattr(
        h.controller, "snapshot", lambda: {**snapshot(), "active_search_id": None, "active_target_text": None}
    )

    response = client.post("/api/searches", data={"target_text": "my green suitcase"})

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "rover_busy"
    assert response.json()["detail"]["active_search_id"] == first
    assert search_count() == before
    with get_session() as session:   # (the database is shared by the whole test session: match this text only)
        assert session.exec(select(Search).where(Search.target_text == "my green suitcase")).all() == []
    assert h.controller.active_search_id() == first


def test_a_shut_down_controller_refuses_new_searches_cleanly(client, rover):
    h = rover(*target_in_view())
    h.controller.shutdown()
    before = search_count()

    response = client.post("/api/searches", data={"target_text": "my bottle"})

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "rover_unavailable"
    assert search_count() == before and h.drive.moves == []


def test_new_search_rows_start_in_searching(client, rover, tap):
    rover(*nothing_to_find())
    sid = create_search(client, "my keys")["search_id"]
    with get_session() as session:
        assert session.get(Search, sid).status == SearchStatus.SEARCHING
    # "keys" is no COCO class: no local screening, OMNI sampling on every stop.
    assert detail(client, sid)["resolved_target"]["category"] is None


def test_simulated_vision_never_moves_a_real_motor_driver(client, rover, tap):
    """Defence in depth behind the startup check: the controller is wired to
    SIMULATED vision but the driver says "gpio". The search is recorded, the
    rover refuses to move, and the API says why."""
    h = rover(*nothing_to_find(), driver_name="gpio")

    sid = create_search(client, "my bottle")["search_id"]

    body = detail(client, sid)
    assert body["status"] == "SEARCHING"
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("error", "unsafe_vision", False)
    assert (body["movement"]["vision_mode"], body["movement"]["driver"]) == ("simulation", "gpio")
    assert h.controller.active_search_id() is None
    assert h.drive.records == []                         # not one call reached the drive port


def test_a_registered_comprehension_provider_is_used_once_per_search(client, rover, tap):
    """The seam for the comprehension work: the provider decides (here it even
    resolves a reference the fallback would have asked about), and the detail
    page, which is refetched on every event, does not call it again."""
    rover(*target_in_view())
    calls: list[str] = []

    def provider(raw_text, context):
        calls.append(raw_text)
        if "weird" in raw_text:
            return ResolvedTarget(raw_text, raw_text, None, (), True, "Do you mean the bottle or the mug?", "ignored")
        return ResolvedTarget(raw_text, raw_text, "bottle", ("desk",), False, None, "ignored")

    register_comprehension_provider("test_provider", provider)
    try:
        asked = client.post("/api/searches", data={"target_text": "the weird one"})
        assert asked.status_code == 422
        assert asked.json()["detail"]["question"] == "Do you mean the bottle or the mug?"

        sid = create_search(client, "the other one")["search_id"]   # the provider resolved it
        tap.wait_phase(sid, "waiting_for_confirmation")
        calls_after_create = calls.count("the other one")
        for _ in range(3):
            assert detail(client, sid)["resolved_target"] == {
                "category": "bottle", "landmarks": ["desk"], "resolver": "test_provider"
            }
        assert calls.count("the other one") == calls_after_create   # the API asked once, at creation
    finally:
        clear_comprehension_provider()
