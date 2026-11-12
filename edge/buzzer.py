"""
IntelliCrash Edge — Active Piezo Buzzer Module (GPIO 18)
========================================================
Controls the driver warning buzzer on the Raspberry Pi node.
Triggered during Phase-1 Pre-Crash Monitoring upon detecting
rash driving maneuvers (hard braking, high yaw oscillation, etc.).
"""

import time
import threading
import sys

# Color formatting
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Try importing RPi.GPIO or gpiozero for hardware control
GPIO_AVAILABLE = False
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
    GPIO_BACKEND = "RPi.GPIO"
except ImportError:
    try:
        from gpiozero import Buzzer
        GPIO_AVAILABLE = True
        GPIO_BACKEND = "gpiozero"
    except ImportError:
        GPIO_BACKEND = None


class BuzzerController:
    """Non-blocking active buzzer driver for driver alert actuation."""

    def __init__(self, pin: int = 18, enabled: bool = True):
        self.pin = pin
        self.enabled = enabled
        self.hardware_active = False
        self._lock = threading.Lock()

        if not self.enabled:
            print("[INFO] Buzzer is disabled by configuration.")
            return

        if GPIO_AVAILABLE and GPIO_BACKEND == "RPi.GPIO":
            try:
                GPIO.setwarnings(False)
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
                self.hardware_active = True
                print(f"[OK] Hardware Buzzer initialized on BCM GPIO {self.pin} (RPi.GPIO).")
            except Exception as e:
                print(f"[WARN] Failed to initialize RPi.GPIO on Pin {self.pin}: {e}. Falling back to simulation.")
        elif GPIO_AVAILABLE and GPIO_BACKEND == "gpiozero":
            try:
                self._gz_buzzer = Buzzer(self.pin)
                self.hardware_active = True
                print(f"[OK] Hardware Buzzer initialized on BCM GPIO {self.pin} (gpiozero).")
            except Exception as e:
                print(f"[WARN] Failed to initialize gpiozero on Pin {self.pin}: {e}. Falling back to simulation.")
        else:
            print(f"[INFO] Hardware GPIO unavailable on this host. Buzzer running in console simulation mode.")

    def _play_pattern(self, duration: float, count: int, interval: float):
        with self._lock:
            for i in range(count):
                if self.hardware_active:
                    try:
                        if GPIO_BACKEND == "RPi.GPIO":
                            GPIO.output(self.pin, GPIO.HIGH)
                            time.sleep(duration)
                            GPIO.output(self.pin, GPIO.LOW)
                        elif GPIO_BACKEND == "gpiozero":
                            self._gz_buzzer.on()
                            time.sleep(duration)
                            self._gz_buzzer.off()
                    except Exception:
                        pass
                else:
                    # Simulation mode output
                    print(f"  {YELLOW}[BUZZER BEEP] Alert signal emitted on GPIO {self.pin}! ({i+1}/{count}){RESET}", flush=True)
                    time.sleep(duration)

                if i < count - 1:
                    time.sleep(interval)

    def trigger_rash_alert(self, count: int = 2, duration: float = 0.15, interval: float = 0.1, blocking: bool = False):
        """Dispatches an audible rash-driving warning in a non-blocking thread."""
        if not self.enabled:
            return

        if blocking:
            self._play_pattern(duration, count, interval)
        else:
            t = threading.Thread(
                target=self._play_pattern,
                args=(duration, count, interval),
                daemon=True,
                name="BuzzerThread"
            )
            t.start()

    def cleanup(self):
        """Releases GPIO pins cleanly upon exit."""
        if self.hardware_active:
            try:
                if GPIO_BACKEND == "RPi.GPIO":
                    GPIO.output(self.pin, GPIO.LOW)
                    GPIO.cleanup(self.pin)
                elif GPIO_BACKEND == "gpiozero":
                    self._gz_buzzer.close()
            except Exception:
                pass


if __name__ == "__main__":
    print("Testing BuzzerController standalone...")
    buzzer = BuzzerController(pin=18, enabled=True)
    buzzer.trigger_rash_alert(count=3, duration=0.1, interval=0.1, blocking=True)
    buzzer.cleanup()
    print("Buzzer test completed.")

