"""
IntelliCrash Edge - Driver Buzzer (GPIO 18)
===========================================
Rash-driving warning (Phase 1) and, optionally, an in-cabin alarm when a crash is confirmed.

Common reasons a buzzer stays silent, all handled or reported here:
  1. Hardware GPIO could not be opened (e.g. RPi.GPIO does not work on a Raspberry Pi 5). The old code then fell
     back to a silent "simulation" without trying the other library. Now: RPi.GPIO -> gpiozero -> simulation, and the
     simulation prints a loud warning.
  2. Active-LOW buzzer module (sounds when the pin is LOW). Use  --buzzer-active-low.
  3. Wiring: signal on BCM GPIO 18 (physical pin 12), ground to a GND pin; a module with a transistor is needed
     if the buzzer draws more than a few mA.
Self-test:   python buzzer.py --pin 18            (add --active-low for active-low modules)
"""

import sys
import time
import threading

CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


class BuzzerController:
    """Non-blocking buzzer driver for driver alerts."""

    def __init__(self, pin: int = 18, enabled: bool = True, active_low: bool = False):
        self.pin = pin
        self.enabled = enabled
        self.active_low = active_low
        self.hardware_active = False
        self.mode = "disabled"
        self._backend = None
        self._lock = threading.Lock()
        self._why = []

        if not self.enabled:
            print("[INFO] Buzzer is disabled by configuration.")
            return

        # try the real hardware backends in order
        for name, fn in (("RPi.GPIO", self._init_rpi), ("gpiozero", self._init_gpiozero)):
            try:
                fn()
                self._backend = name
                self.hardware_active = True
                self.mode = "hardware"
                print(f"{GREEN}[OK] Hardware buzzer ready on BCM GPIO {self.pin} via {name}"
                      f"{' (active-LOW)' if self.active_low else ''}.{RESET}")
                return
            except Exception as e:
                self._why.append(f"{name}: {e}")

        self.mode = "simulation"
        print(f"{RED}{'=' * 66}\n[BUZZER] NO SOUND WILL BE PRODUCED - hardware GPIO could not be opened.\n"
              f"         Reasons: {' | '.join(self._why) if self._why else 'no GPIO library installed'}\n"
              f"         Fix: install gpiozero+lgpio (Pi 5) or RPi.GPIO, run on the Pi, check wiring/pin.\n"
              f"{'=' * 66}{RESET}")

    # ---- backends ----
    def _init_rpi(self):
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=(GPIO.HIGH if self.active_low else GPIO.LOW))
        self._GPIO = GPIO

    def _init_gpiozero(self):
        from gpiozero import Buzzer
        self._gz = Buzzer(self.pin, active_high=not self.active_low, initial_value=False)

    def _set(self, on: bool):
        if self._backend == "RPi.GPIO":
            level = self._GPIO.LOW if (on == self.active_low) else self._GPIO.HIGH
            self._GPIO.output(self.pin, level)
        elif self._backend == "gpiozero":
            (self._gz.on if on else self._gz.off)()

    # ---- patterns ----
    def _play_pattern(self, duration: float, count: int, interval: float, label: str = "ALERT"):
        with self._lock:
            for i in range(count):
                if self.hardware_active:
                    try:
                        self._set(True)
                        time.sleep(duration)
                        self._set(False)
                    except Exception as e:
                        print(f"{YELLOW}[BUZZER] GPIO write failed: {e}{RESET}")
                else:
                    print(f"  {YELLOW}[BUZZER {label} - simulation, no sound] beep {i + 1}/{count}{RESET}", flush=True)
                    time.sleep(duration)
                if i < count - 1:
                    time.sleep(interval)

    def _run(self, duration, count, interval, blocking, label):
        if not self.enabled:
            return
        if blocking:
            self._play_pattern(duration, count, interval, label)
        else:
            threading.Thread(target=self._play_pattern, args=(duration, count, interval, label),
                             daemon=True, name="BuzzerThread").start()

    def trigger_rash_alert(self, count: int = 2, duration: float = 0.15, interval: float = 0.1, blocking: bool = False):
        """Short double beep: rash-driving warning."""
        self._run(duration, count, interval, blocking, "RASH")

    def trigger_crash_alert(self, count: int = 3, duration: float = 0.5, interval: float = 0.25, blocking: bool = False):
        """Three long beeps: a crash was confirmed (the driver is being asked 'Are you OK?')."""
        self._run(duration, count, interval, blocking, "CRASH")

    def cleanup(self):
        if not self.hardware_active:
            return
        try:
            self._set(False)
            if self._backend == "RPi.GPIO":
                self._GPIO.cleanup(self.pin)
            elif self._backend == "gpiozero":
                self._gz.close()
        except Exception:
            pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="IntelliCrash buzzer self-test")
    ap.add_argument("--pin", type=int, default=18)
    ap.add_argument("--active-low", action="store_true")
    a = ap.parse_args()
    b = BuzzerController(pin=a.pin, enabled=True, active_low=a.active_low)
    print(f"Mode: {b.mode} | backend: {b._backend} | pin: BCM {a.pin} | active_low: {a.active_low}")
    print("Rash pattern (2 short beeps)...")
    b.trigger_rash_alert(blocking=True)
    time.sleep(0.6)
    print("Crash pattern (3 long beeps)...")
    b.trigger_crash_alert(blocking=True)
    b.cleanup()
    print("Done. If you heard nothing: mode=simulation means GPIO failed; otherwise try --active-low or check wiring.")
