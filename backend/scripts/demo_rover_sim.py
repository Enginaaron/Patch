"""Simulated end-to-end rover demo -- no camera, no network, no GPIO.

    cd backend
    python -m scripts.demo_rover_sim            # ~8x faster than real time
    python -m scripts.demo_rover_sim --speed 1  # real pulse/settle timings

Runs the REAL controller (app/rover/controller.py) against the simulated world
in app/rover/simulation.py, scenario ``two_bottles``:

    scan -> decoy bottle offered -> "no" -> keep scanning -> the user's bottle
    offered -> "yes" -> centre -> approach (slowing down) -> arrived

A scripted "user" answers the candidate questions. The script prints a
timestamped timeline of every event and a PASS/FAIL summary, and exits
non-zero on failure.

Everything here is SIMULATED: poses, frames, detections, turn rates and
distances are made-up numbers. It demonstrates the control logic; it verifies
nothing about the physical rover. It uses a throw-away database and media
folder under the system temp directory and never touches backend/data.
"""

from __future__ import annotations

import argparse
import logging
import os
import queue
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

TARGET_TEXT = "my water bottle beside my backpack"
RESTING = {"arrived", "target_lost", "stopped", "exhausted", "error"}


def _isolate_environment() -> Path:
    """Point the app at a temp database/media root BEFORE app.config loads.
    Process environment beats backend/.env, so the real data is never used."""
    root = Path(tempfile.mkdtemp(prefix="patch-rover-demo-"))
    os.environ["DATABASE_URL"] = f"sqlite:///{(root / 'demo.db').as_posix()}"
    os.environ["MEDIA_ROOT"] = (root / "media").as_posix()
    os.environ["MOTOR_DRIVER"] = "sim"
    os.environ["OMNI_API_KEY"] = ""          # the demo never calls the gateway
    os.environ["ROVER_SIMULATION"] = "false"  # ports are injected below, not taken from the app wiring
    return root


def _demo_resolver():
    """The real comprehension adapter when it is available, else a one-line
    stand-in (category only matters for the simulated local screening)."""
    try:
        from app.services.comprehension import resolve_target

        resolve_target(TARGET_TEXT)
        return resolve_target, "app.services.comprehension.resolve_target"
    except Exception:  # noqa: BLE001 - optional seam; the demo must run without it
        def fallback(text: str) -> SimpleNamespace:
            return SimpleNamespace(category="bottle" if "bottle" in text.lower() else None, landmarks=())

        return fallback, "built-in demo fallback"


def run_demo(speed: float = 8.0, timeout: float = 120.0, out=print) -> bool:
    """Run the scenario once. Returns True when every check passed."""
    from sqlmodel import select

    from app.db import get_session, init_db
    from app.models import Candidate, CandidateDecision, Find, Item, Search, SearchStatus
    from app.rover.config import RoverConfig, SIMULATION_PROXIMITY_PROFILES_JSON
    from app.rover.controller import RoverController
    from app.rover.simulation import RecordingDrive, SimCamera, SimDrive, SimVision, SimWorld, build_scenario
    from app.services.event_bus import event_bus

    init_db()

    # Default tuning, compressed in time: pulses and settling are `speed` times
    # shorter and the simulated rover is `speed` times quicker, so every pulse
    # turns / advances exactly as far as it would in real time.
    base = RoverConfig()
    cfg = replace(
        base,
        proximity_profiles_json=SIMULATION_PROXIMITY_PROFILES_JSON,
        approach_steering_pulse_seconds=base.approach_steering_pulse_seconds / speed,
        lost_target_hold_seconds=base.lost_target_hold_seconds / speed,
        scan_pulse_seconds=base.scan_pulse_seconds / speed,
        settle_seconds=base.settle_seconds / speed,
        turn_pulse_min_seconds=base.turn_pulse_min_seconds / speed,
        turn_pulse_max_seconds=base.turn_pulse_max_seconds / speed,
        forward_pulse_seconds=base.forward_pulse_seconds / speed,
        slow_forward_pulse_seconds=base.slow_forward_pulse_seconds / speed,
        turn_degrees_per_second=base.turn_degrees_per_second * speed,
    )
    world = build_scenario("two_bottles", SimWorld.for_config(cfg, forward_metres_per_second=0.25 * speed))
    target = world.target()
    drive = RecordingDrive(SimDrive(world))
    vision = SimVision(world, latency_seconds=0.5 / speed)
    resolver, resolver_name = _demo_resolver()
    controller = RoverController(drive, SimCamera(world), vision, cfg, resolver=resolver, tracking_sink=lambda box: None)

    with get_session() as session:
        item = Item(name=TARGET_TEXT)
        session.add(item)
        session.commit()
        search = Search(item_id=item.id, target_text=TARGET_TEXT)
        session.add(search)
        session.commit()
        search_id = search.id

    out("Patch rover demo -- SIMULATED world, SIMULATED detections, sim motor driver")
    out(f"  scenario two_bottles, time compression x{speed:g}, target resolver: {resolver_name}")
    out(f'  search: "{TARGET_TEXT}"')
    out(f"  start: target {world.distance_to(target.id):.2f} m away at {world.relative_bearing(target.id):+.0f} deg, "
        f"decoy {world.distance_to('decoy_bottle'):.2f} m away at {world.relative_bearing('decoy_bottle'):+.0f} deg")
    out("")

    events = event_bus.subscribe(search_id)
    started = time.monotonic()
    timeline: list[dict] = []
    decisions: list[tuple[str, str]] = []        # (description, "accept" | "reject")
    final: dict | None = None
    arrived_at: float | None = None
    asked: dict | None = None                    # the candidate currently on screen

    def stamp() -> str:
        return f"[+{time.monotonic() - started:6.2f}s]"

    try:
        controller.start_search(search_id)
        deadline = started + timeout
        while final is None:
            try:
                event = events.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                out(f"{stamp()} TIMEOUT after {timeout:g}s")
                break
            timeline.append(event)
            kind, payload = event["type"], event["payload"]

            if kind == "movement":
                reason = f" ({payload['reason']})" if payload["reason"] else ""
                out(f"{stamp()} movement   {payload['phase']:<25}{reason} \"{payload['message']}\"")
                if payload["phase"] in RESTING:
                    final = payload
                    arrived_at = time.monotonic()
                elif payload["phase"] == "waiting_for_confirmation" and asked is not None:
                    # The scripted user answers the question once it is asked.
                    candidate, asked = asked, None
                    is_target = candidate["description"] == target.description
                    decisions.append((candidate["description"], "accept" if is_target else "reject"))
                    if is_target:
                        out(f"{stamp()} USER       \"yes, that's it\"")
                        controller.accept_candidate(search_id, candidate["candidate_id"])
                    else:
                        out(f"{stamp()} USER       \"no, not that one\"")
                        controller.reject_candidate(search_id, candidate["candidate_id"])
            elif kind == "candidate_found":
                out(f"{stamp()} candidate  \"{payload['description']}\" box={payload['box']}")
                asked = payload
            else:
                out(f"{stamp()} {kind:<10} {payload}")
    finally:
        event_bus.unsubscribe(search_id, events)
        controller.shutdown()

    # ---- checks -----------------------------------------------------------------
    with get_session() as session:
        status = session.get(Search, search_id).status
        candidates = session.exec(
            select(Candidate).where(Candidate.search_id == search_id).order_by(Candidate.created_at)
        ).all()
        finds = session.exec(select(Find).where(Find.search_id == search_id)).all()

    phases = [e["payload"]["phase"] for e in timeline if e["type"] == "movement"]
    messages = [e["payload"]["message"] for e in timeline if e["type"] == "movement"]
    moves = drive.moves

    def ordered(*wanted: str) -> bool:
        position = 0
        for phase in phases:
            if position < len(wanted) and phase == wanted[position]:
                position += 1
        return position == len(wanted)

    checks = [
        ("the decoy was offered first and rejected",
         len(decisions) >= 1 and decisions[0] == ("a green glass bottle on the floor", "reject")),
        ("the user's bottle was offered next and accepted",
         len(decisions) == 2 and decisions[1] == (target.description, "accept")),
        ("the rejected decoy was never offered again",
         [c.decision for c in candidates] == [CandidateDecision.REJECTED, CandidateDecision.ACCEPTED]),
        ("phases ran scan -> question -> scan -> question -> check -> approach -> arrived",
         ordered("scanning", "waiting_for_confirmation", "scanning", "waiting_for_confirmation",
                 "checking", "approaching", "arrived")),
        ("it slowed down before arriving", "Almost there — slowing down" in messages),
        ("final phase is arrived / proximity_confirmed",
         final is not None and (final["phase"], final["reason"]) == ("arrived", "proximity_confirmed")),
        ("Search.status is FOUND since the acceptance (FOUND never meant 'arrived') with one Find row",
         status == SearchStatus.FOUND and len(finds) == 1),
        ("simulated rover ended next to the target, not the decoy",
         world.distance_to(target.id) < 0.8 < world.distance_to("decoy_bottle")),
        ("simulated rover faces the target", abs(world.relative_bearing(target.id)) < 10.0),
        ("every motor pulse armed the watchdog ttl", bool(moves) and all(m.ttl is not None for m in moves)),
        ("wheels stopped at the end; no motor command after arrival",
         not world.is_moving() and drive.records[-1].command == "stop"
         and arrived_at is not None and not [m for m in moves if m.at > arrived_at]),
    ]

    out("")
    out(f"  simulated end state: {world.distance_to(target.id):.2f} m from the target "
        f"({world.relative_bearing(target.id):+.1f} deg), {len(moves)} motor pulses, "
        f"{vision.identify_calls} identify calls, {vision.screen_calls} local screenings")
    out("")
    for label, ok in checks:
        out(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    passed = all(ok for _, ok in checks)
    out("")
    out(f"RESULT: {'PASS' if passed else 'FAIL'} ({sum(ok for _, ok in checks)}/{len(checks)} checks) -- simulation only, "
        "nothing here was verified on the physical rover")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the simulated Patch rover demo (no hardware, no network).")
    parser.add_argument("--speed", type=float, default=8.0, help="time compression factor (1 = real pulse timings)")
    parser.add_argument("--timeout", type=float, default=120.0, help="give up after this many seconds")
    parser.add_argument("--verbose", action="store_true", help="also show the controller's INFO log")
    args = parser.parse_args()
    if not args.speed > 0:
        parser.error("--speed must be positive")

    for stream in (sys.stdout, sys.stderr):
        # Messages contain an ellipsis and dashes; never die (or print
        # mojibake into a pipe) on a legacy Windows code page.
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = _isolate_environment()
    print(f"(throw-away database and media in {root})")
    return 0 if run_demo(speed=args.speed, timeout=args.timeout) else 1


if __name__ == "__main__":
    sys.exit(main())
