# Patch

A small multimodal search-and-assistance rover for the **Huawei OMNI Live Challenge**.
Patch sees through a camera, understands natural language about what it sees, and
helps you find things ("find my water bottle — I think it's near my backpack").

## Structure

- `frontend/` — Vite + React (TypeScript). The live screen shows Patch's camera
  feed and an **OMNI Live** chat panel.
- `backend/` — FastAPI. Streams the camera as MJPEG (`/video`) and answers chat
  turns grounded in the current frame (`/api/chat`).

## Run

Backend:

```bash
cd backend
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173, proxies /api and /video to :8000
```

## OMNI Live chat

The chat on the live screen is powered by a single seam:
`backend/app/services/omni_service.py`.

- **No key set (default):** it returns fake, context-aware replies so the whole
  UI works end to end. The chat badge shows **Demo**.
- **Key set:** it sends the live camera frame plus the conversation to Huawei's
  OMNI Live model and returns the real grounded answer. The badge shows **Live**.

To switch to the real model, just fill in `.env` — no code changes:

```env
OMNI_API_KEY=your-key
OMNI_BASE_URL=https://api.omni-live.huaweicloud.com/v1   # adjust if the gateway differs
OMNI_MODEL=omni-live
OMNI_CHAT_PATH=/chat/completions
```

The request is built against the common OpenAI-compatible multimodal chat shape
(text + `image_url` parts). If the OMNI gateway expects a different envelope,
`_generate_live` in `omni_service.py` is the one place to adjust it.

## Driving & autonomous search

Same "works today, real later" pattern as the chat. The drive stack has a
pluggable `MotorDriver` (`backend/app/motors/`):

- **`sim` (default):** logs the intended left/right wheel speeds — develop and
  demo the whole loop on a laptop, no hardware needed.
- **`gpio`:** drives the real base — a TB6612FNG dual H-bridge on a Raspberry
  Pi 5 — via `gpiozero`. Set `MOTOR_DRIVER=gpio`, confirm the `MOTOR_*` pins in
  `.env`, and `pip install gpiozero lgpio` on the Pi.

**Building the physical rover** (parts, TB6612 ↔ Pi wiring table, battery/power,
flashing, and a motor-test script) is documented in
[`docs/HARDWARE.md`](docs/HARDWARE.md).

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/drive` | Manual teleop: `{command, speed}` where command is `forward`, `backward`, `turn_left`, `turn_right`, `veer_left`, `veer_right`, `stop`. Ends the running autonomous mission (manual control wins); auto-stops after `DRIVE_MANUAL_TTL_SECONDS` unless re-sent. |
| `POST` | `/api/drive/stop` | Stop the wheels (also ends the running mission). |
| `GET`  | `/api/drive/state` | Current command + wheel speeds. |
| `POST` | `/api/rover/stop` | **E-stop.** Halts the wheels now and ends the running mission; the search itself stays resumable. |
| `GET`  | `/api/rover/state` | Controller snapshot + drive state + camera health. |
| `POST` | `/api/searches` | Start a search (form: `target_text`, optional `reference_images`, `replace_active`). `409 rover_busy` while another search owns the rover, `422 needs_clarification` for "the other one". |
| `GET`  | `/api/searches/{id}` | Search detail: recognition `status`, rover `movement`, pending / accepted candidate, `can_resume`, `terminal`. |
| `POST` | `/api/searches/{id}/candidates/{candidate_id}/accept` \| `/reject` | Answer "is this it?". Accept → `FOUND` and the rover starts to approach. |
| `POST` | `/api/searches/{id}/resume` | Continue after a stop, a manual override, a lost target or a restart. Nothing resumes by itself. |
| `POST` | `/api/searches/{id}/cancel` | Halt the rover first, then mark the search `CANCELLED`. |
| `GET`  | `/api/searches/{id}/events` | Server-sent events; every message means "refetch the detail". |

**One controller owns the wheels** (`backend/app/rover/controller.py`; design,
status/phase mapping and limits in [`docs/AUTONOMY.md`](docs/AUTONOMY.md)). A
search scans in short turn pulses with the wheels stopped for every look, asks
the user about a plausible match, and after a "yes" centres on it and
approaches in short forward pulses until a conservative, camera-based proximity
check holds. Two things are reported separately: `status` (`FOUND` = the user
confirmed the identification — never "the rover got there") and
`movement.phase` (arrival is only ever `arrived`). Manual driving always wins:
any `/api/drive` command ends the mission, and manual commands carry a
server-side deadman (`DRIVE_MANUAL_TTL_SECONDS`).

Live vision requests normalized `[xmin, ymin, xmax, ymax]` boxes from the model
and explicitly converts them to Patch's internal `[ymin, xmin, ymax, xmax]` format.
Before using a box, a second model call checks that the actual crop contains the
requested object. Both calls share the vision time budget. A rejected crop,
invalid coordinates, or a timeout causes a stationary retry and eventually a
stop, rather than authorizing movement from the description alone. This is a
model-based check, not a guarantee of correct identification.

There are no fabricated detections anywhere near the motors: without an
`OMNI_API_KEY` the live vision path raises instead of inventing a result. To
try the whole loop on a laptop use the explicit simulation, which is refused
unless the motor driver is `sim`:

```bash
cd backend
python -m scripts.demo_rover_sim                                   # scripted scenario, PASS/FAIL summary
ROVER_SIMULATION=true MOTOR_DRIVER=sim uvicorn app.main:app --port 8000   # the real server on a synthetic camera
```

Everything the simulation shows (poses, turn rates, distances, detections) is
made up; physical movement has not been verified yet.

The live screen exposes all of this: a **STOP** button, **Cancel search** /
**Resume**, the candidate question (buttons or hold-to-speak yes/no), the
movement phase, and a **Controls** panel with a hold-to-drive teleop d-pad.

## Voice, memory, and deployment

Hold the microphone button on the home page to fill the search field, then
submit it. On the search page, confirm or reject a candidate by button or voice.
Speech uses the OMNI transcription/synthesis adapter. The memory gallery keeps
confirmed finds; item pages support Find Again and a stationary Quick Check.
Identification and physical arrival are separate states.

Build the frontend on a development machine with `cd frontend && npm ci && npm run build`.
Copy `frontend/dist/` to the same location in the Pi checkout. FastAPI serves the
built app, API, media, and USB camera stream from port 8000; Node is not needed
on the Pi. Run from `backend/` with the configured environment:

```bash
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For microphone access from your laptop, use an SSH tunnel and open localhost:

```bash
ssh -N -L 8765:127.0.0.1:8000 scout@scout.local
```

Open http://127.0.0.1:8765 while that tunnel is running. A direct HTTP connection
to the Pi can show video but browsers generally require localhost or HTTPS for
microphone capture. Startup does not resume movement automatically.
