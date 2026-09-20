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

## Not wired yet

- **Voice:** speech-out via ElevenLabs and speech-in (mic) are stubbed in the UI
  (`ELEVENLABS_API_KEY` has a home in `.env`). Text chat works today.
