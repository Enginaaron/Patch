"""The laptop demo must keep passing: it is the repeatable end-to-end check of
scan -> decoy -> reject -> target -> accept -> centre -> approach -> arrived.
(Simulation only. Here it runs against the test database, time-compressed.)"""

from scripts import demo_rover_sim


def test_demo_passes_and_prints_a_timeline():
    lines: list[str] = []
    assert demo_rover_sim.run_demo(speed=40.0, timeout=60.0, out=lines.append) is True

    text = "\n".join(lines)
    assert "SIMULATED" in text and "RESULT: PASS" in text and "[FAIL]" not in text
    timeline = [line for line in lines if line.startswith("[+")]
    assert any("waiting_for_confirmation" in line for line in timeline)
    assert any('USER       "no, not that one"' in line for line in timeline)
    assert any('USER       "yes, that\'s it"' in line for line in timeline)
    assert "arrived" in timeline[-1] and "proximity_confirmed" in timeline[-1]
