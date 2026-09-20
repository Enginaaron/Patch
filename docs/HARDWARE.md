# Building Patch — wiring & bring-up

This is the physical build for Patch's drivetrain: a **Raspberry Pi 5** driving
**two 4.5 V gearmotors** (front-wheel differential drive) through a
**TB6612FNG** motor driver, with the Pi and the motors on **separate power**.

> ⚠️ **Do the first power-on with the wheels off the ground**, and always
> disconnect the motor battery while you change wiring.

---

## 1. Parts

| Part | Notes |
| --- | --- |
| Raspberry Pi 5 | Runs the backend, camera, and GPIO. |
| USB-C power bank | Powers the **Pi only**. 5 V, 3 A+ (a 5 V/5 A PD bank is ideal for the Pi 5). |
| TB6612FNG breakout | Dual H-bridge between the Pi and the motors. |
| 2× gearmotors (4.5 V) | Front wheels, side by side. |
| Motor battery pack | **4× AA NiMH (~4.8 V)** is the best match for 4.5 V motors. 3× AA alkaline (4.5 V) also works. This powers the motors, **not** the Pi. |
| Chassis + 2 wheels + **1 rear caster/skid** | Two driven front wheels need a rear caster so the rover sits on 3 points and can rotate in place. |
| Female–female jumper wires | Pi header → TB6612 (9 signal wires). |
| USB webcam | Plugs into a Pi USB port; shows up as `CAMERA_INDEX=0`. (Simplest option — the Pi Camera module needs `picamera2`, see §6.) |

---

## 2. How the pieces connect

```
   USB-C power bank ──5V──> Raspberry Pi 5           (Pi powered on its own)
                                │
              3.3V logic + 9 GPIO signal wires
                                │
                                v
   Motor battery ──VM──>  [ TB6612FNG ]  ──A01/A02──> LEFT gearmotor
   (4.5–4.8V)   ──GND─┐        │         ──B01/B02──> RIGHT gearmotor
                      └────────┴── GND shared with Pi GND  ← IMPORTANT
```

Two rules that keep the Pi safe:

1. **Common ground:** the motor battery's **−** and the **Pi GND** must meet at
   the TB6612's GND pins. Without a shared ground the driver won't read the Pi's
   signals reliably.
2. **Never feed motor-battery voltage into the Pi.** In this build the only wire
   shared between the two power domains is ground. The power bank alone powers
   the Pi.

---

## 3. Pi ↔ TB6612 wiring (logic side)

BCM = the GPIO number the code uses (in `.env`). "Pin" = the physical pin on the
Pi's 40-pin header.

| TB6612 pin | Wire to | Pi header pin | BCM |
| --- | --- | --- | --- |
| VCC | Pi **3V3** | pin 17 | 3.3 V |
| GND (logic) | Pi **GND** | pin 34 | — |
| STBY | GPIO16 | pin 36 | 16 |
| PWMA | GPIO12 | pin 32 | 12 |
| AIN1 | GPIO5 | pin 29 | 5 |
| AIN2 | GPIO6 | pin 31 | 6 |
| PWMB | GPIO13 | pin 33 | 13 |
| BIN1 | GPIO20 | pin 38 | 20 |
| BIN2 | GPIO21 | pin 40 | 21 |

These BCM numbers are the defaults in `backend/.env.example`. If you wire to
different pins, just change the `MOTOR_*` values in `.env` — no code edits.

Existing `.env` files override these defaults: update their pin values to match
the table above as well.

## 4. Motor & battery wiring (power side)

| TB6612 pin | Wire to |
| --- | --- |
| VM | Motor battery **+** |
| GND (power) | Motor battery **−** (and thus shared with Pi GND) |
| A01, A02 | LEFT gearmotor's two terminals |
| B01, B02 | RIGHT gearmotor's two terminals |

Motor terminal polarity just sets spin direction — if a wheel turns the wrong
way you fix it in software (§5), no rewiring needed.

- **VCC vs VM:** VCC is the 3.3 V *logic* supply from the Pi; VM is the *motor*
  supply from the battery. Don't swap them.
- Channel **A = left**, channel **B = right** in the default config.

---

## 5. Software: flash, upload, run

On the Pi (over SSH — you already reach it at `scout.local`):

```bash
# 5.1 System packages (lgpio drives the Pi 5's GPIO; libgl1 is for OpenCV)
sudo apt update
sudo apt install -y git python3-venv python3-pip python3-lgpio libgl1

# 5.2 Get the code (this branch)
git clone https://github.com/Enginaaron/Patch.git
cd Patch && git checkout robot-controls

# 5.3 Backend deps
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt gpiozero lgpio
#   If opencv-python is slow to build on the Pi, use the lighter headless build:
#   pip install opencv-python-headless

# 5.4 Config — turn on the real motors
cp .env.example .env
sed -i 's/^MOTOR_DRIVER=sim/MOTOR_DRIVER=gpio/' .env   # or edit by hand
```

**Test the motors before anything else** (wheels off the ground):

```bash
python -m scripts.test_motors
# Individual commands (speed 0..1, duration in seconds):
python -m scripts.test_motors --command forward --speed 0.4 --duration 1
python -m scripts.test_motors --command turn_left --speed 0.4 --duration 0.5
# Laptop-only dry run:
python -m scripts.test_motors --sim
```

It drives each wheel forward/back and then rotates. Confirm:
- "LEFT forward" turns the left wheel the way that drives the rover forward.
- Same for "RIGHT forward".

Fixes:
- Wheel spins the **wrong direction** → set `MOTOR_LEFT_INVERT=true` (or right)
  in `.env`.
- The **wrong wheel** moves (left command drives the right wheel) → swap the
  A01/A02 pair with B01/B02 on the TB6612.

Then run the app:

```bash
# Backend — bind to all interfaces so your laptop can reach it
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

```bash
# Frontend (needs Node 20+; install via nvm or NodeSource if apt's is older)
cd ../frontend
npm install
npm run dev -- --host      # then open http://scout.local:5173 on your laptop
```

The live screen's **Controls** panel now drives the real wheels: the d-pad is
teleop, **Auto search** runs the vision loop. (Search uses fake vision until
`OMNI_API_KEY` is set — see the main README.)

### Optional: auto-start the backend on boot

`/etc/systemd/system/patch.service`:

```ini
[Unit]
Description=Patch backend
After=network.target

[Service]
User=YOUR_PI_USER
WorkingDirectory=/home/YOUR_PI_USER/Patch/backend
ExecStart=/home/YOUR_PI_USER/Patch/backend/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now patch.service
```

---

## 6. Camera

- **USB webcam (recommended):** plug it in; it's `/dev/video0`, i.e.
  `CAMERA_INDEX=0` (the default). Nothing else to do.
- **Pi Camera module:** needs the `picamera2` stack rather than OpenCV's
  `VideoCapture`. Not wired up yet — open an issue/ping me if you go this route
  and I'll add a `PiCamera2` source alongside the existing `WebcamCamera`.

---

## 7. First-drive checklist

1. Motor battery **disconnected**, all 9 logic wires + VCC/GND to the Pi in.
2. Motor wires to A/B, VM/GND to the battery leads (battery still disconnected).
3. Power the Pi from the USB-C bank; SSH in; `MOTOR_DRIVER=gpio`.
4. Wheels **off the ground**, connect the motor battery.
5. `python -m scripts.test_motors`, calibrate invert/swap.
6. Start backend + frontend, drive from the Controls d-pad.
7. Put it on the floor and try **Auto search**.
