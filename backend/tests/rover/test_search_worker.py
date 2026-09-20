"""search_worker after the slimming: Aleesha's helpers still behave the same,
and start_worker / stop_worker are shims onto the controller."""

import io

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.models import SearchStatus
from app.rover.controller import set_rover_controller
from app.rover.types import MovementPhase
from app.services import search_worker
from tests.rover.helpers import add_bottle, fast_config, make_search, make_world, search_status


def test_the_per_search_thread_and_registry_are_gone():
    for name in ("SearchWorker", "_active_workers", "_registry_lock", "_unregister"):
        assert not hasattr(search_worker, name)
    assert (search_worker.FRAME_TARGET_WIDTH, search_worker.JPEG_QUALITY) == (768, 85)


def test_prepare_frame_resizes_to_768_wide_jpeg():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, jpeg = search_worker._prepare_frame(frame)
    assert resized.shape == (576, 768, 3)
    assert Image.open(io.BytesIO(jpeg)).size == (768, 576)


def test_crop_jpeg_cuts_the_box_out_of_the_resized_frame():
    frame = np.zeros((576, 768, 3), dtype=np.uint8)
    frame[:, 384:] = (255, 255, 255)
    crop = Image.open(io.BytesIO(search_worker._crop_jpeg(frame, [250, 500, 750, 1000])))
    assert crop.size == (384, 288)                        # right half, middle half
    assert np.asarray(crop.convert("L")).min() > 200      # all white: it really is the right half


def test_save_search_image_writes_under_media_root(media_root):
    relative = search_worker._save_search_image("some-search", b"\xff\xd8jpeg", "full")
    assert relative.startswith("searches/some-search/") and relative.endswith("_full.jpg")
    assert (media_root / relative).read_bytes() == b"\xff\xd8jpeg"


def test_meets_threshold_default_and_explicit(monkeypatch):
    monkeypatch.setattr(settings, "candidate_likelihood_threshold", "high")
    assert search_worker._meets_threshold("high") and not search_worker._meets_threshold("medium")
    assert search_worker._meets_threshold("medium", "medium") and not search_worker._meets_threshold("low", "medium")
    with pytest.raises(KeyError):
        search_worker._meets_threshold("certain")


def test_start_and_stop_worker_delegate_to_the_controller(harness):
    cfg = fast_config()
    world = make_world(cfg)
    add_bottle(world, bearing=0, distance=1.5, is_target=True)
    h = harness(cfg, world)
    sid, log = h.search("my bottle")
    other = make_search("something else")
    set_rover_controller(h.controller)
    try:
        search_worker.start_worker(sid)
        log.wait_phase("waiting_for_confirmation")
        assert h.controller.active_search_id() == sid

        search_worker.stop_worker(other)                   # not the active search: nothing happens
        assert h.controller.active_search_id() == sid

        search_worker.stop_worker(sid)
        state = h.controller.state_for(sid)
        assert (state.phase, state.reason) == (MovementPhase.STOPPED, "user_stop")
        assert h.controller.active_search_id() is None
        assert search_status(sid) == SearchStatus.CANDIDATE_PENDING   # a stop is not a cancel
    finally:
        set_rover_controller(None)
