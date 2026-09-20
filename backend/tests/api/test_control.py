"""E-stop, rover state, manual driving, and the endpoints that are gone.
Motion and detections are SIMULATED; the motor driver is the sim driver."""

import threading
from contextlib import contextmanager

import anyio.to_thread
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.rover import simulation
from app.routers import control, searches, speech
from app.services.drive_service import drive_service
from app.services.omni_speech import TranscriptionResult
from tests.api.helpers import create_search, detail, nothing_to_find, now, target_in_view

TIMEOUT = 10.0


# --- e-stop / state ----------------------------------------------------------------


def test_rover_stop_halts_now_and_reports_the_whole_rover(client, rover, tap):
    h = rover(*nothing_to_find())
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)

    response = client.post("/api/rover/stop")
    answered_at = now()

    assert response.status_code == 200
    state = response.json()
    assert state["active_search_id"] is None and state["active_target_text"] is None
    assert (state["movement"]["phase"], state["movement"]["reason"], state["movement"]["search_id"]) == ("stopped", "user_stop", sid)
    assert (state["vision_mode"], state["driver"]) == ("simulation", "sim")
    assert state["drive"]["command"] == "stop" and state["drive"]["deadline_in"] is None
    assert set(state["camera"]) >= {"available", "error", "last_seq", "last_frame_age_seconds", "consecutive_failures"}

    assert h.controller.join_missions(5)
    assert h.drive.records[-1].command == "stop" and not h.world.is_moving()
    assert h.moves_since(answered_at) == []
    assert tap.wait_phase(sid, "stopped")["payload"]["reason"] == "user_stop"
    assert tap.events(sid, "search_cancelled") == []     # a stop is not a cancel

    # Pressing it again with nothing running is fine, and still commands a stop.
    records = len(h.drive.records)
    assert client.post("/api/rover/stop").status_code == 200
    assert len(h.drive.records) == records + 1 and h.drive.records[-1].command == "stop"


def test_rover_state_while_a_mission_is_waiting(client, rover, tap):
    rover(*target_in_view())
    sid = create_search(client, "my bottle")["search_id"]
    tap.wait_phase(sid, "waiting_for_confirmation")

    state = client.get("/api/rover/state").json()

    assert (state["active_search_id"], state["active_target_text"]) == (sid, "my bottle")
    assert state["movement"]["phase"] == "waiting_for_confirmation" and state["movement"]["active"] is True
    assert state["drive"]["driver"] in ("sim", "none") and state["camera"]["available"] is False


# --- a busy threadpool must never delay a stop ---------------------------------------


@contextmanager
def _saturated_threadpool():
    """The routers under test on an app whose threadpool has ONE worker, held
    by a slow plain-``def`` request (think speech synthesis). Anything that
    needs the threadpool now queues until ``release`` -- exactly the situation
    a stop must not be caught in."""
    started, release = threading.Event(), threading.Event()
    probe = FastAPI()
    probe.include_router(control.router)
    probe.include_router(searches.router)

    @probe.post("/test/one-worker")
    async def one_worker() -> dict:
        anyio.to_thread.current_default_thread_limiter().total_tokens = 1
        return {}

    @probe.get("/test/hold-the-worker")
    def hold_the_worker() -> dict:
        started.set()
        release.wait(TIMEOUT)
        return {}

    threads: list[threading.Thread] = []

    def in_background(call) -> threading.Thread:
        thread = threading.Thread(target=call, daemon=True)
        threads.append(thread)
        thread.start()
        return thread

    # As a context manager: one event loop (and so one threadpool) for every request.
    with TestClient(probe) as probe_client:
        try:
            yield probe_client, started, in_background
        finally:
            release.set()
            for thread in threads:
                thread.join(TIMEOUT)


def test_a_saturated_threadpool_cannot_delay_stop_or_manual_drive(rover, tap):
    h = rover(*nothing_to_find())
    with _saturated_threadpool() as (probe_client, started, in_background):
        create_search(probe_client, "my bottle")
        assert h.drive.wait_for_moves(2)
        probe_client.post("/test/one-worker")
        in_background(lambda: probe_client.get("/test/hold-the-worker"))
        assert started.wait(TIMEOUT)
        queued = in_background(lambda: probe_client.get("/api/drive/state"))   # plain def: stuck in the queue

        answers: dict = {}
        stop = in_background(lambda: answers.update(stop=probe_client.post("/api/rover/stop")))
        stop.join(TIMEOUT)
        stopped_at = now()
        drive = in_background(lambda: answers.update(drive=probe_client.post("/api/drive", json={"command": "stop"})))
        drive.join(TIMEOUT)

        assert not stop.is_alive() and answers["stop"].status_code == 200
        assert not drive.is_alive() and answers["drive"].status_code == 200
        assert queued.is_alive()                         # the threadpool really was stuck the whole time
        assert h.controller.active_search_id() is None
        assert h.controller.join_missions(5) and h.moves_since(stopped_at) == [] and not h.world.is_moving()


def test_a_saturated_threadpool_cannot_delay_the_halt_of_a_cancel(rover, tap):
    h = rover(*nothing_to_find())
    with _saturated_threadpool() as (probe_client, started, in_background):
        sid = create_search(probe_client, "my bottle")["search_id"]
        assert h.drive.wait_for_moves(2)
        probe_client.post("/test/one-worker")
        in_background(lambda: probe_client.get("/test/hold-the-worker"))
        assert started.wait(TIMEOUT)

        answers: dict = {}
        cancel = in_background(lambda: answers.update(cancel=probe_client.post(f"/api/searches/{sid}/cancel")))
        tap.wait_type(sid, "search_cancelled", timeout=TIMEOUT)
        halted_at = now()

        # The rover is halted and the search CANCELLED although the response
        # (whose detail read needs the threadpool) has not been sent yet.
        assert cancel.is_alive()
        assert h.controller.active_search_id() is None
        assert h.controller.join_missions(5) and h.moves_since(halted_at) == [] and not h.world.is_moving()
    assert answers["cancel"].status_code == 200 and answers["cancel"].json()["status"] == "CANCELLED"


def test_a_transcription_in_flight_cannot_delay_a_stop(rover, tap, monkeypatch):
    """The stop endpoints run inline on the event loop, so nothing else may
    block that loop. /api/speech/transcribe is ``async def`` and used to call
    the blocking OMNI client right there: for the seconds a transcription takes
    (the user answering a candidate by voice) every stop waited behind it."""
    h = rover(*nothing_to_find())
    started, release = threading.Event(), threading.Event()

    def slow_transcribe(audio_bytes: bytes, audio_format: str) -> TranscriptionResult:
        started.set()
        release.wait(TIMEOUT)
        return TranscriptionResult(transcript="yes", latency_seconds=0.0)

    monkeypatch.setattr(speech, "transcribe_audio", slow_transcribe)   # no network
    probe = FastAPI()
    for router in (control.router, searches.router, speech.router):
        probe.include_router(router)

    answers: dict = {}
    with TestClient(probe) as probe_client:
        create_search(probe_client, "my bottle")
        assert h.drive.wait_for_moves(2)
        transcribe = threading.Thread(
            target=lambda: answers.update(
                transcribe=probe_client.post("/api/speech/transcribe", files={"audio": ("answer.webm", b"not really audio", "audio/webm")})
            ),
            daemon=True,
        )
        transcribe.start()
        try:
            assert started.wait(TIMEOUT)
            stop = threading.Thread(target=lambda: answers.update(stop=probe_client.post("/api/rover/stop")), daemon=True)
            stop.start()
            stop.join(3.0)
            stopped_at = now()

            assert not stop.is_alive() and answers["stop"].status_code == 200
            assert transcribe.is_alive()                     # the transcription really was in flight the whole time
            assert h.controller.active_search_id() is None
            assert h.controller.join_missions(5) and h.moves_since(stopped_at) == [] and not h.world.is_moving()
        finally:
            release.set()
            transcribe.join(TIMEOUT)
    assert answers["transcribe"].status_code == 200 and answers["transcribe"].json()["transcript"] == "yes"


def test_a_search_being_created_cannot_delay_a_stop(rover, tap, monkeypatch):
    """Creating a search awaits its reference-image uploads (a phone photo is
    read in the threadpool), and the event loop serves other requests in the
    meantime -- such as a stop, whose movement write needs the database. If the
    create handler sat on an open SQLite write transaction across that await,
    the stop would block the event loop on the database lock, the create could
    never get back to commit, and everything would hang until SQLite's busy
    timeout."""
    h = rover(*nothing_to_find())
    uploading, release = threading.Event(), threading.Event()

    async def slow_upload(item_id: str, upload) -> str:
        uploading.set()
        await run_in_threadpool(release.wait, TIMEOUT)     # yields to the event loop, like a big upload.read()
        return f"items/{item_id}/reference.jpg"

    monkeypatch.setattr(searches, "save_reference_image", slow_upload)
    probe = FastAPI()
    for router in (control.router, searches.router):
        probe.include_router(router)

    answers: dict = {}
    with TestClient(probe) as probe_client:
        old = create_search(probe_client, "my bottle")["search_id"]
        assert h.drive.wait_for_moves(2)
        create = threading.Thread(
            target=lambda: answers.update(
                create=probe_client.post(
                    "/api/searches",
                    data={"target_text": "a brand new item for the upload test", "replace_active": "true"},
                    files={"reference_images": ("photo.jpg", b"stand-in bytes", "image/jpeg")},
                )
            ),
            daemon=True,
        )
        create.start()
        try:
            assert uploading.wait(TIMEOUT)
            stop = threading.Thread(target=lambda: answers.update(stop=probe_client.post("/api/rover/stop")), daemon=True)
            stop.start()
            stop.join(3.0)
            stopped_at = now()

            assert not stop.is_alive() and answers["stop"].status_code == 200
            assert create.is_alive()                         # the new search really was still being created
            assert answers["stop"].json()["movement"]["reason"] == "user_stop"
            assert h.controller.active_search_id() is None
            assert h.moves_since(stopped_at) == [] and not h.world.is_moving()
        finally:
            release.set()
            create.join(TIMEOUT)
    # The create then finishes normally and the new search owns the rover. The
    # old one was already stopped by then, so there was nothing left to replace:
    # it stays resumable instead of being cancelled.
    assert answers["create"].status_code == 200, answers["create"].text
    created = answers["create"].json()
    assert created["reference_count"] == 1                 # item, reference image and search were all committed
    assert h.controller.active_search_id() == created["search_id"]
    before = detail(TestClient(app), old)
    assert (before["status"], before["movement"]["reason"]) == ("SEARCHING", "user_stop")


# --- manual driving -------------------------------------------------------------------


def test_manual_drive_takes_the_wheels_from_the_mission(client, rover, tap):
    h = rover(*nothing_to_find(), through_drive_service=True)
    drive_service.start()                                # the sim motor driver (MOTOR_DRIVER=sim in tests)
    sid = create_search(client, "my bottle")["search_id"]
    assert h.drive.wait_for_moves(2)

    response = client.post("/api/drive", json={"command": "forward", "speed": 0.5})
    answered_at = now()

    assert response.status_code == 200
    state = response.json()
    assert (state["command"], state["left"], state["right"], state["driver"]) == ("forward", 0.5, 0.5, "sim")
    # The deadman is armed: no re-send within the ttl, and the wheels stop by themselves.
    assert 0 < state["deadline_in"] <= settings.drive_manual_ttl_seconds

    # The mission is over (the search is not): a human on the controls always wins.
    assert h.controller.active_search_id() is None
    body = detail(client, sid)
    assert body["status"] == "SEARCHING" and body["can_resume"] is True
    assert (body["movement"]["phase"], body["movement"]["reason"], body["movement"]["active"]) == ("stopped", "manual_override", False)
    # The dying mission neither moved again nor cut the human's command short.
    assert h.controller.join_missions(5)
    assert h.moves_since(answered_at) == []
    assert h.drive.records[-1].command == "stop" and h.drive.records[-1].at < answered_at
    assert drive_service.state()["command"] == "forward"

    stopped = client.post("/api/drive/stop")
    assert stopped.status_code == 200
    assert (stopped.json()["command"], stopped.json()["left"], stopped.json()["deadline_in"]) == ("stop", 0.0, None)
    assert tap.events(sid, "search_cancelled") == []


def test_an_invalid_manual_command_is_refused_with_the_wheels_stopped(client, rover, tap):
    h = rover(*nothing_to_find(), through_drive_service=True)
    drive_service.start()
    create_search(client, "my bottle")
    assert h.drive.wait_for_moves(2)

    response = client.post("/api/drive", json={"command": "sideways"})

    assert response.status_code == 400 and "sideways" in response.json()["detail"]
    # Someone reached for the controls: the autonomous search stops either way.
    assert h.controller.active_search_id() is None and h.controller.join_missions(5)
    assert drive_service.state()["command"] == "stop" and not h.world.is_moving()


def test_manual_drive_without_a_mission_and_with_the_deadman_disabled(client, rover, monkeypatch):
    h = rover(*target_in_view())
    drive_service.start()
    monkeypatch.setattr(settings, "drive_manual_ttl_seconds", 0.0)

    state = client.post("/api/drive", json={"command": "turn_left", "speed": 0.4}).json()

    assert (state["command"], state["left"], state["right"]) == ("turn_left", -0.4, 0.4)
    assert state["deadline_in"] is None                  # 0 disables the deadman
    assert h.drive.records == []                         # no mission: the override had nothing to do
    assert client.get("/api/drive/state").json()["command"] == "turn_left"
    assert client.post("/api/drive/stop").json()["command"] == "stop"


def test_manual_commands_move_the_simulated_world_in_the_laptop_demo(client, rover, monkeypatch):
    rover(*target_in_view())
    monkeypatch.setattr(settings, "rover_simulation", True)
    simulation.reset_sim_world()
    try:
        world = simulation.get_sim_world()
        assert client.post("/api/drive", json={"command": "turn_right", "speed": 0.4}).status_code == 200
        assert world.is_moving()
        assert client.post("/api/drive/stop").status_code == 200
        assert not world.is_moving() and world.pose()[2] > 0   # it turned right, then stopped
    finally:
        simulation.reset_sim_world()


# --- the old loop is gone -----------------------------------------------------------------


def test_the_old_auto_search_endpoints_are_gone(client, rover):
    h = rover(*target_in_view())

    assert client.post("/api/searches/some-id/start-auto", json={"target_text": "my bottle"}).status_code == 404
    assert client.post("/api/searches/stop-auto").status_code == 405
    assert client.get("/api/searches/auto-status").status_code == 404   # now just an unknown search id

    paths = set(app.openapi()["paths"])
    assert not any("auto" in path for path in paths)
    assert {"/api/rover/stop", "/api/rover/state", "/api/searches/{search_id}/resume"} <= paths
    assert h.drive.records == []
