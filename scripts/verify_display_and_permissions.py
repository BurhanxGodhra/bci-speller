"""
Phase 1 diagnostic script.

1. Measures ACTUAL delivered frame timing (not the OS-reported nominal refresh rate)
   over a fixed window, reporting mean FPS, jitter (std dev of frame time), and the
   set of SSVEP-safe integer-divisor frequencies for whatever refresh rate is measured.
2. Checks whether PyAutoGUI can query the screen / control the mouse, which on macOS
   requires Accessibility + Screen Recording permissions to be granted to the terminal
   or IDE running this script.

Run:
    python scripts/verify_display_and_permissions.py
"""

import time
import statistics
import sys

import pygame


def verify_refresh_rate(measure_seconds: float = 5.0, target_fallback_hz: int = 60):
    pygame.init()
    # Windowed, not fullscreen, so this is safe to run without disrupting your desktop.
    screen = pygame.display.set_mode((600, 300))
    pygame.display.set_caption("SSVEP Frame Timing Verification")
    clock = pygame.time.Clock()

    try:
        detected = pygame.display.get_current_refresh_rate()
    except Exception:
        detected = None

    nominal_hz = detected if detected and detected > 0 else target_fallback_hz
    print(f"OS-reported nominal refresh rate: {nominal_hz} Hz")
    print(f"Measuring actual delivered frame timing for {measure_seconds:.0f}s...")

    frame_times = []
    font = pygame.font.SysFont(None, 28)
    start = time.perf_counter()
    last = start

    while time.perf_counter() - start < measure_seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)

        now = time.perf_counter()
        frame_times.append(now - last)
        last = now

        screen.fill((15, 15, 20))
        msg = font.render("Measuring frame timing... do not resize window", True, (230, 230, 230))
        screen.blit(msg, (20, 130))
        pygame.display.flip()

        clock.tick(nominal_hz * 2)  # allow headroom; vsync (below) is what actually paces us

    pygame.quit()

    # Drop the first few frames (startup transient)
    frame_times = frame_times[5:]
    mean_dt = statistics.mean(frame_times)
    measured_hz = 1.0 / mean_dt
    jitter_ms = statistics.pstdev(frame_times) * 1000

    print("\n--- Results ---")
    print(f"Measured refresh rate:  {measured_hz:.2f} Hz")
    print(f"Frame time jitter (std): {jitter_ms:.3f} ms")

    if jitter_ms > 1.5:
        print("⚠️  Jitter is high (>1.5ms). SSVEP frequency precision may be degraded.")
        print("    Close other GPU-heavy apps and re-run before proceeding to Phase 4.")
    else:
        print("✅  Jitter is low enough for reliable SSVEP stimulus timing.")

    print("\nSSVEP-safe frequencies (exact integer divisors of measured refresh rate):")
    safe_freqs = [round(measured_hz / d, 3) for d in range(2, 16) if measured_hz / d >= 4]
    print(", ".join(f"{f} Hz (÷{d})" for d, f in zip(range(2, 16), safe_freqs)))
    print("\nUse ONLY frequencies from this list (or close to them) in Phase 4 stimulus design —")
    print("non-divisor frequencies will alias against your true frame rate and corrupt CCA detection.")

    return measured_hz, jitter_ms


def verify_pyautogui_permissions():
    print("\n--- PyAutoGUI / macOS Permission Check ---")
    try:
        import pyautogui
    except ImportError:
        print("❌ pyautogui is not installed. Run: pip install -r requirements.txt")
        return False

    try:
        size = pyautogui.size()
        print(f"Screen size detected: {size}")
        pos_before = pyautogui.position()
        # Non-destructive: move 1px and back, proves OS-level control is authorized
        pyautogui.moveTo(pos_before.x + 1, pos_before.y, duration=0)
        pyautogui.moveTo(pos_before.x, pos_before.y, duration=0)
        print("✅ PyAutoGUI can control the mouse — Accessibility permission is granted.")
        return True
    except Exception as e:
        print(f"❌ PyAutoGUI could not control the input devices: {e}")
        print("   On macOS: System Settings → Privacy & Security → Accessibility →")
        print("   enable your terminal app (Terminal.app, iTerm2, or your IDE).")
        print("   You will need this working before Phase 7 (OS keypress injection).")
        return False


if __name__ == "__main__":
    measured_hz, jitter_ms = verify_refresh_rate()
    perms_ok = verify_pyautogui_permissions()

    print("\n=== SUMMARY ===")
    print(f"Refresh rate measured: {measured_hz:.2f} Hz (jitter {jitter_ms:.3f} ms)")
    print(f"PyAutoGUI permissions: {'OK' if perms_ok else 'NEEDS ATTENTION'}")
