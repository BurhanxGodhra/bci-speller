"""
Phase 1 diagnostic script.

1. Queries the AUTHORITATIVE display refresh rate directly from macOS (via NSScreen's
   maximumFramesPerSecond), which correctly reports fixed-rate panels (e.g. MacBook Air's
   locked 60Hz) as well as ProMotion's variable-rate ceiling (MacBook Pro 14"/16", up to
   120Hz) — unlike trying to time a small windowed SDL2 surface, which SDL2 on macOS does
   NOT reliably vsync-lock, especially for unfocused/non-fullscreen windows.
2. Runs a best-effort pygame frame-timing loop as a secondary sanity check, but labels it
   clearly as non-authoritative — real stimulus-loop jitter will be verified properly in
   Phase 4, once we're running the actual fullscreen SSVEP flicker engine.
3. Checks whether PyAutoGUI can query the screen / control the mouse, which on macOS
   requires Accessibility permission to be granted to the terminal or IDE running this
   script.

Run:
    python scripts/verify_display_and_permissions.py
"""

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
        print(f"❌ Could not query NSScreen: {e}")
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
        print("❌ pyautogui is not installed. Run: pip install -r requirements.txt")
        return False

    try:
        size = pyautogui.size()
        print(f"Screen size detected: {size}")
        pos_before = pyautogui.position()
        pyautogui.moveTo(pos_before.x + 1, pos_before.y, duration=0)
        pyautogui.moveTo(pos_before.x, pos_before.y, duration=0)
        print("✅ PyAutoGUI can control the mouse — Accessibility permission is granted.")
        return True
    except Exception as e:
        print(f"❌ PyAutoGUI could not control the input devices: {e}")
        print("   On macOS: System Settings → Privacy & Security → Accessibility →")
        print("   enable your terminal app (Terminal.app, iTerm2, or your IDE).")
        return False


if __name__ == "__main__":
    print("--- Authoritative Display Refresh Rate (macOS NSScreen) ---")
    auth_hz, is_variable = get_authoritative_refresh_rate()

    if auth_hz:
        print(f"Panel max refresh rate: {auth_hz:.1f} Hz")
        print(f"Variable-refresh (ProMotion-style) panel: {'YES' if is_variable else 'NO (fixed rate)'}")
    else:
        print("Falling back to assuming 60 Hz nominal (query failed or non-macOS).")
        auth_hz = 60.0
        is_variable = False

    print("\n--- Rough pygame timing estimate (NON-authoritative, sanity check only) ---")
    rough_hz = rough_pygame_estimate()
    print(f"Rough windowed-frame estimate: {rough_hz:.1f} Hz")
    print("(This number is expected to be noisy/wrong on macOS — ignore large discrepancies")
    print(" vs. the authoritative value above. True jitter gets verified in Phase 4's")
    print(" fullscreen stimulus loop, where SDL2 vsync behaves correctly.)")

    print(f"\nSSVEP-safe frequencies (exact integer divisors of {auth_hz:.1f} Hz):")
    safe_freqs = compute_safe_ssvep_frequencies(auth_hz)
    print(", ".join(f"{f} Hz (÷{d})" for d, f in safe_freqs))

    if is_variable:
        print("\n⚠️  Your panel supports variable refresh (ProMotion). For SSVEP we need a")
        print("    FIXED rate during the whole session — in Phase 4 we'll force the display")
        print("    to a fixed refresh mode (e.g. via System Settings > Displays, or")
        print("    programmatically) before deriving stimulus frequencies from it.")

    perms_ok = verify_pyautogui_permissions()

    print("\n=== SUMMARY ===")
    print(f"Authoritative panel refresh: {auth_hz:.1f} Hz (variable: {is_variable})")
    print(f"Rough pygame estimate (ignore if far off): {rough_hz:.1f} Hz")
    print(f"PyAutoGUI permissions: {'OK' if perms_ok else 'NEEDS ATTENTION'}")
