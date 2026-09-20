# Approach control and OMNI replies

The detection pipeline and physical motor drivers are unchanged. The existing
`DriveService.set_speeds(left, right, command, ttl)` interface carries the mixed
commands, including its calibration, power cap and independent watchdog.

## Diagnosis of the previous controller

Baseline: commit `d850adb`.

- `backend/app/rover/controller.py:1113`: `if abs(offset) > cfg.center_tolerance`
  selects rotation; the `else` at line 1123 is the only forward path. Rotation
  and forward travel were mutually exclusive.
- `backend/app/rover/config.py:37`: the old tolerance was 0.08 of full image
  width. The old offset was -0.5..0.5, not -1..1.
- `backend/app/rover/controller.py:1116`: offset determined turn duration, with
  0.08-0.30 second limits. Turn power was a fixed 0.4, not a proportional mixer.
- `backend/app/rover/motion.py:100`: each pulse ended with a stop before the next
  settled frame and recognition call. Repeated turns plus cloud delays looked
  like micro-adjustments. There was no deadzone hysteresis or minimum PWM floor.
- Whether 40% PWM physically stalls either motor cannot be determined from
  these files. The effective minimum still needs calibration on the rover.

## Complete implementation

All files in the repository are complete; no code fragments need to be spliced in.

- `backend/app/rover/config.py`: named tuning constants at the top, with units
  and directions for adjustment. New approach values are dataclass defaults;
  they are not overridden by the old `ROVER_CENTER_TOLERANCE` setting.
- `backend/app/rover/steering.py`: filtered proportional steering, wheel mixing,
  centring/pivot hysteresis, minimum effective wheel commands and arrival latch.
- `backend/app/rover/controller.py`: uses that mixer inside the existing
  approach loop, preserving identity checks, arrival confirmation and mission cancellation.
- `backend/app/rover/motion.py`: runs mixed wheel commands under the existing
  lock, mission-generation check, stop handling and watchdog.
- `backend/app/rover/types.py`: declares the already-existing wheel-speed API.
- `backend/app/rover/simulation.py`: simulates the actual differential wheel
  speeds so tests exercise arcs rather than treating them as straight travel.

The horizontal error is `2 * (target_center_x / frame_width - 0.5)`. For a
moderate offset, forward speed decreases with error while the differential
turn command increases proportionally. Both wheels remain positive. Within
12% of full frame width either side of the centre, steering becomes zero;
it resumes outside 15%. Rotation in place starts above normalized error 0.55
and ends at 0.50. Each nonzero requested wheel command has a 0.30 minimum,
with a 0.75 maximum. Existing driver calibration and power caps still apply.

Arrival uses the existing category profiles, scaled to 80% of their old size
thresholds to stop earlier. The arrival latch releases below 70%, avoiding
boundary chatter. Multiple stationary observations, including OMNI validation,
are still required. These thresholds are image sizes, not measured centimetres.

Motion remains **pulse, stop, observe** because the existing recognition path
requires stationary frames and cloud calls can take seconds. Steering and
forward travel happen together *within* each pulse. A command is normally held
for at most `LOST_TARGET_HOLD_SECONDS` (0.25 s); the independent watchdog has
the existing extra margin. If detection disappears during a pulse, that pulse
ends and no fresh movement is issued without a usable observation. Misses retry
while stationary, bounded by the existing attempt budget and the new 2 s retry
window checked between responses. An outstanding cloud call retains its existing
inference timeout, but the wheels are already stopped throughout that wait.
There is no deliberate blind continuation after a reported miss.

## Tuning order

Calibrate `APPROACH_MIN_MOTOR` first so both wheels reliably start under load.
It must remain below `APPROACH_WHEEL_MAX`. Check existing per-wheel calibration
and power caps if actual delivered PWM differs from the request.

| Symptom | First change | Then, if needed |
| --- | --- | --- |
| Oscillates left/right | Reduce `STEER_KP` | Reduce `STEER_FILTER_ALPHA` slightly; widen the deadzone gap. Too much filtering adds lag. |
| Drives past the target | Reduce `APPROACH_V_MAX` | Shorten `LOST_TARGET_HOLD_SECONDS`; lower `ARRIVAL_ENTER_SCALE` if arrival itself is late. |
| Stops too early | Increase `ARRIVAL_ENTER_SCALE` | Increase `ARRIVAL_EXIT_SCALE` proportionally, keeping it below enter. |
| Stops too late | Decrease `ARRIVAL_ENTER_SCALE` | Decrease exit proportionally. Tune category profiles for unusually large/small objects. |
| Turns too slowly | Increase `STEER_KP` | If saturated, raise `STEER_MAX`; if wheels only twitch, recalibrate the minimum motor command. |

## OMNI speech through the laptop

Click **Enable Patch's voice** once in the webapp. This unlocks browser audio
from a user gesture. The existing `/api/speech/synthesize` endpoint creates
OMNI audio; the laptop plays the returned WAV. No browser TTS fallback is used.

The shared speaker handles search starts, candidate questions, acceptance,
arrival, loss, stop/error announcements and chat replies. New replies replace
stale audio. Recording silences Patch to avoid recording its own voice.
Failures are visible, with Retry, Replay and Mute controls.

Implementation: `frontend/src/lib/omniSpeaker.ts`,
`frontend/src/components/VoiceOutput.tsx`, its CSS, the existing voice hook,
announcement selection, chat panel and app entry point.

## Verification and deployment

Backend: 450 tests passed, including simulated complete missions and new
steering boundary tests. Frontend: build/lint and the mocked OMNI playback test
pass. Browser inspection confirms Enable and visible retry/error controls.
The Pi was switched off during development; these changes have not been
deployed or physically calibrated. Live OMNI audio has not been auditioned for
this change because the configured key is on the Pi.

When the Pi is online again, pull main, copy the new `frontend/dist` build to
`/home/scout/Patch/frontend/dist`, then restart the Patch service. Start a fresh
search; automatic service startup does not resume a motor mission.
