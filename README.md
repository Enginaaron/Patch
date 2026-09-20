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
- **`gpio`:** drives a real differential base over a two-motor H-bridge
  (TB6612 / L298N style) via `gpiozero`. Set `MOTOR_DRIVER=gpio`, confirm the
  `MOTOR_*` pins in `.env`, and `pip install gpiozero` on the Pi.

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/drive` | Manual teleop: `{command, speed}` where command is `forward`, `backward`, `turn_left`, `turn_right`, `veer_left`, `veer_right`, `stop`. Cancels any running search. |
| `POST` | `/api/drive/stop` | Stop the wheels. |
| `GET`  | `/api/drive/state` | Current command + wheel speeds. |
| `POST` | `/api/searches/{id}/start-auto` | Start autonomous search for `{target_text}`. |
| `POST` | `/api/searches/stop-auto` | Stop the search. |
| `GET`  | `/api/searches/auto-status` | Live phase / message / confidence. |

**The search loop** (`backend/app/services/search_service.py`) runs closed-loop
on vision: each tick it grabs the current frame, asks
`OmniLiveService.locate_target` where the target is (`x`, `size`, `confidence`),
then steers — rotate to **scan**, turn to **center**, drive to **approach**, and
**stop** once the target fills enough of the frame. With no OMNI key the locate
step is simulated so the rover visibly scans, centers, and arrives in demo mode.

The live screen exposes all of this: a **Controls** panel with a teleop d-pad,
an **Auto search** toggle, and the live search phase/confidence.

## Not wired yet

- **Voice:** speech-out via ElevenLabs and speech-in (mic) are stubbed in the UI
  (`ELEVENLABS_API_KEY` has a home in `.env`). Text chat works today.
