# Rover autonomy

How Patch gets from "find my water bottle" to standing next to it, and what
keeps the wheels safe on the way. Code: `backend/app/rover/`.

**Status of this work:** the control logic is covered by tests and a scripted
demo that run entirely in **simulation**. Nothing here has been verified on the
physical rover yet. Turn rates, proximity thresholds and pulse timings are
starting points to calibrate on the floor (see [HARDWARE.md](HARDWARE.md)).

This is a controlled-area demo: a forward RGB camera and two driven wheels, no
encoders, no distance sensor. So: no obstacle avoidance, no room navigation,
rotation angles are approximate (dead-reckoned from pulse time), and an RGB
frame gives no reliable centimetres.

## One controller owns the wheels

`RoverController` (`app/rover/controller.py`) is the only thing that drives
autonomously, and `drive_service` is the only path to the motor driver. One
search owns the rover at a time; a second `POST /api/searches` gets
`409 rover_busy` unless it asks to replace the running one.

A search runs as one mission thread that alternates **look while stopped** and
**move in one short pulse**:

```
scan:      fresh frame -> (local YOLO screen) -> OMNI check -> candidate?  -> turn pulse -> ...
question:  wheels stopped, no timeout, until the user answers yes / no
approach:  re-acquire (stationary) -> centre with turn pulses -> short forward pulses
           -> repeated stationary proximity checks -> arrived
```

OMNI returns a single strongest candidate, not a list of alternatives. YOLO is
a cheap per-frame screen for a COCO class; it does not track identity. After a
"yes", identity rests on OMNI being shown the crop the user confirmed, plus a
geometric association gate between consecutive boxes; if the two disagree the
rover stops (`target_lost / identity_uncertain`) rather than follow a guess.

## Status vs movement phase

Two separate things are stored and reported, on purpose:

* `Search.status` — the **recognition lifecycle** (`SEARCHING`,
  `CANDIDATE_PENDING`, `FOUND`, `CANCELLED`).
* `movement.phase` — what the **rover body** is doing (`RoverMovement` table,
  `MovementState` in the API).

| Moment | `Search.status` | Candidate | `movement.phase` | SSE event(s) |
|---|---|---|---|---|
| search created | SEARCHING | – | scanning / checking | `movement` |
| OMNI reports a plausible match | CANDIDATE_PENDING | PENDING | waiting_for_confirmation | `candidate_found`, `movement` |
| user rejects | SEARCHING | REJECTED | scanning (resumes) | `candidate_rejected`, `movement` |
| user accepts | FOUND (`ended_at` set, `Find` row) | ACCEPTED | checking → centering / approaching | `candidate_accepted`, `movement` |
| proximity confirmed | FOUND (unchanged) | ACCEPTED | **arrived** | `movement` (+ `terminal: true`) |
| accepted target lost / identity uncertain | FOUND | ACCEPTED | target_lost | `movement` |
| scan or candidate budget used up | SEARCHING | – | exhausted | `movement` |
| user cancel | CANCELLED | unchanged | stopped (`cancelled`) | `search_cancelled` (terminal) |
| e-stop / manual override | unchanged | unchanged | stopped | `movement` |

**`FOUND` means "the user confirmed Patch identified the item". It never means
the rover reached it.** Arrival is only ever `movement.phase == "arrived"`, and
OMNI finding an object never sets it. A search is `terminal` when it is
`CANCELLED` or `arrived`; everything else can still be resumed
(`POST /api/searches/{id}/resume`). Nothing resumes by itself — after a backend
restart a mission that was running is reported as `stopped / interrupted`.

## Safety properties

These are requirements, each pinned by tests in `backend/tests/rover` and
`backend/tests/api`:

* **Generation token.** Every start / stop / cancel / replace / manual override
  / shutdown bumps the controller's generation under one lock and stops the
  wheels before returning. Every side effect of a mission (motor command,
  candidate row, status change, movement write, event, tracking box) re-checks
  that its generation is still current at the moment of the effect. A late OMNI
  answer from a cancelled mission cannot move the rover, create a candidate or
  flip a `CANCELLED` search back.
* **Bounded pulses with an independent deadline.** A mission only ever moves in
  pulses (`motion.py`, hard cap 1.5 s). Each one arms
  `ttl = duration + ROVER_PULSE_WATCHDOG_MARGIN_SECONDS` in `drive_service`,
  whose watchdog thread zeroes the wheels at the deadline even if the mission
  thread never comes back. Manual `/api/drive` commands carry their own deadman
  (`DRIVE_MANUAL_TTL_SECONDS`); the ControlPad re-sends while a button is held.
* **Wheels stopped during inference.** A frame is only requested after the
  pulse's stop, and OMNI / YOLO only ever run with the wheels stopped.
* **Fresh frames.** After a pulse the controller waits for a frame that is
  newer than the last one it used *and* captured after
  `wheels stopped + ROVER_SETTLE_SECONDS`. A dead camera reads as "no frame"
  (`error / stale_frames`), never as the last good frame.
* **Stop cannot queue.** `/api/rover/stop`, `/api/drive`, `/api/drive/stop` and
  cancel run inline on the event loop and call only non-blocking code, so a busy
  threadpool (speech, MJPEG, slow reads) cannot delay them. Nothing blocking may
  be added to an `async def` handler for the same reason.
* **No fabricated detections near motors.** Without `OMNI_API_KEY` the live
  vision path raises. The simulation is explicit (`ROVER_SIMULATION=true`),
  refuses to start unless `MOTOR_DRIVER=sim`, and the pulse code additionally
  refuses simulated vision paired with a non-sim driver (`error / unsafe_vision`).
* **A box is required to move.** A `found` answer without a location may be
  shown as a question, but accepting it ends in `target_lost / no_box` with zero
  motor commands.
* **Manual control wins.** Any `/api/drive` command invalidates the mission
  first; the mission's end-of-pulse stop will not cut a manual command short.
* **Shutdown order:** mission → motor driver → camera, each step attempted even
  if an earlier one fails.

## Arrival is a conservative heuristic

`proximity.py`: arrival is declared from the *apparent size* of the accepted
object's box (per-category thresholds, overridable with
`ROVER_PROXIMITY_PROFILES_JSON`), confirmed by `ROVER_ARRIVAL_CONFIRMATIONS`
consecutive stationary observations, the last one validated by OMNI. No
universal box size means "arrived", timed movement does not measure distance,
and the thresholds shipped here are uncalibrated. The approach also gives up on
its own: time / pulse budget (`approach_budget`) and no growth in box size over
several pulses (`no_progress`).

## Understanding the request

`app/services/comprehension.py` is the seam for the comprehension work:
`resolve_target(text)` yields a COCO category for local screening (or `None` →
OMNI samples every stop) and landmarks; OMNI always receives the user's full
text. The built-in fallback is rule-based English parsing. References such as
"the other one" cannot be resolved from a transcript, so they are answered with
a clarification question (`422 needs_clarification`) instead of a guess.
A real provider plugs in through `register_comprehension_provider`.

## Trying it without hardware (simulation)

```bash
cd backend
python -m scripts.demo_rover_sim          # scan -> decoy -> "no" -> target -> "yes" -> approach -> arrived
ROVER_SIMULATION=true MOTOR_DRIVER=sim uvicorn app.main:app --port 8000
python -m pytest tests -q
```

With `ROVER_SIMULATION=true` the camera frames, detections, poses, turn rates
and distances all come from `app/rover/simulation.py` and are made up; the UI
labels the page **SIMULATION**. It demonstrates the control logic and says
nothing about the physical rover.

## Known limits

* Rejected-object bearings are dead-reckoned and kept per mission; after a
  resume, rejection memory relies on the description / crop hints given to OMNI.
  The number of questions per search is bounded (`ROVER_MAX_CANDIDATES`).
* A true target standing within `ROVER_REJECT_BEARING_TOLERANCE_DEGREES` of a
  rejected object is suppressed too, until the rover moves.
* With two look-alike objects close together, which one is followed depends on
  OMNI's single answer fitting the association gates.
* `movement.seq` restarts after a backend restart.
