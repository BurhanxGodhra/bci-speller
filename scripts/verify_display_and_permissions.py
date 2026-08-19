import platform
import sys
import time
import statistics

import pygame


def get_authoritative_refresh_rate():
    """Query macOS directly for the real panel refresh rate via NSScreen.

    Returns (hz, is_variable) or (None, None) if not on macOS / query fails.
    """
    if platform.system() != "Darwin":
        print("Not running on macOS — skipping NSScreen query.")
        return None, None

    try:
        from AppKit import NSScreen
    except ImportError:
        print("❌ pyobjc-framework-Cocoa not installed. Run: pip install -r requirements.txt")
        return None, None

    try:
        screen = NSScreen.mainScreen()
        max_fps = screen.maximumFramesPerSecond()  # macOS 12+; ceiling for variable-refresh panels
        # Heuristic: ProMotion panels report a max well above 60 (90 or 120); fixed panels
        # (e.g. MacBook Air, external displays) report their actual fixed rate.
        is_variable = max_fps > 60
        return float(max_fps), is_variable
    except Exception as e:
        print(f"Could not query NSScreen: {e}")
        return None, None


def rough_pygame_estimate(measure_seconds: float = 3.0):
    """Best-effort, NON-authoritative frame-timing loop. See module docstring."""
    pygame.init()
    screen = pygame.display.set_mode((500, 200))
    pygame.display.set_caption("Rough timing check (non-authoritative)")
    font = pygame.font.SysFont(None, 24)

    frame_times = []
    start = time.perf_counter()
    last = start

    while time.perf_counter() - start < measure_seconds:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
        screen.fill((15, 15, 20))
        screen.blit(font.render("rough estimate only...", True, (200, 200, 200)), (20, 90))
        pygame.display.flip()
        now = time.perf_counter()
        frame_times.append(now - last)
        last = now

    pygame.quit()
    frame_times = frame_times[3:] or frame_times
    mean_dt = statistics.mean(frame_times)
    return 1.0 / mean_dt if mean_dt > 0 else 0.0


def compute_safe_ssvep_frequencies(hz: float, max_divisor: int = 15):
    return [
        (d, round(hz / d, 3))
        for d in range(2, max_divisor + 1)
        if hz / d >= 4
    ]


def verify_pyautogui_permissions():
    print("\n--- PyAutoGUI / macOS Permission Check ---")
    try:
        import pyautogui
    except ImportError:
        print("pyautogui is not installed")
        return False

    try:
        size = pyautogui.size()
        print(f"Screen size detected: {size}")
        pos_before = pyautogui.position()
        pyautogui.moveTo(pos_before.x + 1, pos_before.y, duration=0)
        pyautogui.moveTo(pos_before.x, pos_before.y, duration=0)
        print("Accessibility permission is granted.")
        return True
    except Exception as e:
        print(f"could not control the input devices: {e}")
        return False


if __name__ == "__main__":
    print("--- Authoritative Display Refresh Rate (macOS NSScreen) ---")
    auth_hz, is_variable = get_authoritative_refresh_rate()

    if auth_hz:
        print(f"Panel max refresh rate: {auth_hz:.1f} Hz")
        print(f"Variable-refresh (ProMotion-style) panel: {'YES' if is_variable else 'NO (fixed rate)'}")
    else:
        print("(query failed or non-macOS)")
        auth_hz = 60.0
        is_variable = False

    print("\n--- Rough pygame timing estimate (NON-authoritative, sanity check only) ---")
    rough_hz = rough_pygame_estimate()
    print(f"Rough windowed-frame estimate: {rough_hz:.1f} Hz")

    print(f"\nSSVEP-safe frequencies (exact integer divisors of {auth_hz:.1f} Hz):")
    safe_freqs = compute_safe_ssvep_frequencies(auth_hz)
    print(", ".join(f"{f} Hz (÷{d})" for d, f in safe_freqs))

    if is_variable:
        print("\nYour panel supports variable refresh (ProMotion)")

    perms_ok = verify_pyautogui_permissions()

    print("\n=== SUMMARY ===")
    print(f"Authoritative panel refresh: {auth_hz:.1f} Hz (variable: {is_variable})")
    print(f"Rough pygame estimate (ignore if far off): {rough_hz:.1f} Hz")
    print(f"PyAutoGUI permissions: {'OK' if perms_ok else 'NEEDS ATTENTION'}")
