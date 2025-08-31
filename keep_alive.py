#!/usr/bin/env python3
"""
Kaggle Keep-Alive / Anti-Sleep helper (PyAutoGUI-based)

What it does
------------
- Gently "wiggles" the mouse or taps SHIFT at random intervals to keep the OS
  and browser active while your Kaggle training runs.
- Tries to prevent the machine from going to sleep (Windows/macOS/Linux).

How to use
----------
1) Install deps once:
   pip install pyautogui

   (Optional for better Linux sleep inhibit)
   - systemd-based Linux: ensure `systemd-inhibit` exists (usually preinstalled).
2) Open your browser and bring the Kaggle notebook tab to the FRONT.
3) Run this script in a separate terminal:
   python keep_alive.py
4) To emergency-stop: move the mouse to the TOP-LEFT corner of the screen.
   (PyAutoGUI failsafe).

Notes
-----
- On Linux Wayland (e.g., Fedora KDE Wayland), global automation may be blocked.
  If nothing happens:
  * Log out and choose an "X11 / Xorg" session (e.g., "Plasma (X11)"),
  * or enable Wayland automation tools (varies by distro/DE).
- This is a best-effort helper. Kaggle's server-side policies can still time out.
"""

import os
import sys
import random
import time
import subprocess
import threading
from contextlib import contextmanager

try:
    import pyautogui
except ImportError as e:
    print("PyAutoGUI is not installed. Run: pip install pyautogui")
    sys.exit(1)

# ------------------ Configuration (edit here) ------------------
MIN_INTERVAL_SEC = 45     # minimum seconds between actions
MAX_INTERVAL_SEC = 75     # maximum seconds between actions
WIGGLE_PIXELS   = 1       # small mouse move to avoid messing with UI
RANDOM_JITTER   = 2       # occasional extra jitter (pixels)
USE_MOUSE       = True    # perform mouse wiggles
USE_KEYBOARD    = True    # press SHIFT occasionally
SHIFT_RATIO     = 0.35    # probability to use SHIFT instead of mouse
TOTAL_DURATION_HOURS = None  # set to a number (e.g., 8) to stop after N hours; or None to run indefinitely
# ---------------------------------------------------------------

pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort

def human_sleep(seconds: float):
    """Sleep with tiny random jitter to avoid exact rhythms."""
    jitter = random.uniform(-0.5, 0.5)
    time.sleep(max(0.0, seconds + jitter))

def gentle_mouse_wiggle():
    try:
        x, y = pyautogui.position()
        dx = random.choice([-WIGGLE_PIXELS, WIGGLE_PIXELS])
        dy = random.choice([-WIGGLE_PIXELS, WIGGLE_PIXELS])
        # occasional extra jitter
        dx += random.randint(-RANDOM_JITTER, RANDOM_JITTER)
        dy += random.randint(-RANDOM_JITTER, RANDOM_JITTER)
        pyautogui.moveRel(dx, dy, duration=random.uniform(0.05, 0.15))
        # move back a bit so cursor doesn't drift
        pyautogui.moveRel(-dx, -dy, duration=random.uniform(0.05, 0.15))
    except pyautogui.FailSafeException:
        raise
    except Exception as e:
        print(f"[mouse] warning: {e}")

def gentle_shift_tap():
    try:
        pyautogui.keyDown("shift")
        time.sleep(0.05 + random.uniform(0, 0.05))
        pyautogui.keyUp("shift")
    except pyautogui.FailSafeException:
        raise
    except Exception as e:
        print(f"[keyboard] warning: {e}")

@contextmanager
def prevent_system_sleep():
    """
    Best-effort cross-platform sleep inhibitor.
    - Windows: SetThreadExecutionState
    - macOS: spawn `caffeinate`
    - Linux (systemd): spawn `systemd-inhibit` keeping a child process alive
    """
    platform = sys.platform
    inhibitor = None
    caffeinate_proc = None
    try:
        if platform.startswith("win"):
            # Windows
            import ctypes
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_AWAYMODE_REQUIRED = 0x00000040  # prevents sleep on some systems
            ret = ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
            )
            if ret == 0:
                print("[sleep] Windows SetThreadExecutionState failed")
            inhibitor = "windows"
        elif platform == "darwin":
            # macOS
            try:
                caffeinate_proc = subprocess.Popen(["caffeinate", "-dimsu"])
                inhibitor = "macOS caffeinate"
            except FileNotFoundError:
                print("[sleep] caffeinate not found; continuing without system sleep lock")
        else:
            # Linux
            # Use systemd-inhibit to hold a blocker while a harmless child sleeps
            try:
                caffeinate_proc = subprocess.Popen(
                    ["systemd-inhibit", "--why=Kaggle keep-alive", "--mode=block", "bash", "-c", "while :; do sleep 3600; done"]
                )
                inhibitor = "linux systemd-inhibit"
            except FileNotFoundError:
                print("[sleep] systemd-inhibit not found; continuing without system sleep lock")

        yield
    finally:
        # Cleanup
        if sys.platform.startswith("win"):
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # clear continuous flag
            except Exception:
                pass
        if caffeinate_proc:
            try:
                caffeinate_proc.terminate()
            except Exception:
                pass

def main():
    start_time = time.time()
    print("Kaggle Keep-Alive running.")
    print("Bring your Kaggle tab to the FRONT, then leave this window alone.")
    print("Emergency stop: move mouse to TOP-LEFT corner.")
    if TOTAL_DURATION_HOURS is not None:
        print(f"Will auto-stop after {TOTAL_DURATION_HOURS} hours.")

    with prevent_system_sleep():
        while True:
            # stop after duration (optional)
            if TOTAL_DURATION_HOURS is not None:
                hours = (time.time() - start_time) / 3600.0
                if hours >= TOTAL_DURATION_HOURS:
                    print("Time limit reached. Exiting.")
                    break

            try:
                do_shift = USE_KEYBOARD and (random.random() < SHIFT_RATIO or not USE_MOUSE)
                if do_shift:
                    gentle_shift_tap()
                elif USE_MOUSE:
                    gentle_mouse_wiggle()
                else:
                    # if both disabled, still no-op to maintain timing
                    pass
            except pyautogui.FailSafeException:
                print("Failsafe triggered. Exiting.")
                break

            wait_s = random.uniform(MIN_INTERVAL_SEC, MAX_INTERVAL_SEC)
            human_sleep(wait_s)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Bye!")
