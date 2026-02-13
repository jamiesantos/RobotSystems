"""Simple sensing, interpretation and control helpers for 3-channel
grayscale/line sensors.

This module provides three small classes used in the lesson:

- Sensor: wraps three ADC channels and returns raw readings
- Interpreter: turns three grayscale readings into a position in [-1, 1]
- Controller: maps the interpreted offset to a steering command

The implementations are intentionally small and dependency-light so they
work with the existing `ADC` and `Picarx` APIs in this repository.
"""
from typing import Iterable, List, Optional
import time
import os
import sys

import sys
#sys.path.insert(0, "../picarx")
#import picarx_control as ctrl
import picarx.picarx_control as ctrl

# When this file is executed directly (python3 sim_robot_hat/line_control.py)
# the package context is missing which breaks relative imports used by
# sibling modules (for example adc.py imports .i2c). Ensure the repository
# root is on sys.path so absolute imports like 'sim_robot_hat.adc' work.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    # when imported as a package
    from .adc import ADC
except Exception:
    # fall back to absolute import (works when sys.path contains repo root)
    from sim_robot_hat.adc import ADC


import subprocess
import cv2
import numpy as np

class Sensor:
    """
    Camera-based replacement for the 3-channel grayscale sensor.

    Captures a still image using rpicam-still, converts to grayscale,
    and samples three vertical regions (left, center, right).
    """

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # Best-effort capture: capture a frame and save three crops that match
        # the left/center/right regions used by `read()` so users can verify
        # the camera and the sampling regions. Do not raise on failure.
        try:
            p = subprocess.run(
                [
                    "rpicam-still",
                    "--nopreview",
                    "--width", str(self.width),
                    "--height", str(self.height),
                    "-o", "-"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=10
            )
            img_color = cv2.imdecode(
                np.frombuffer(p.stdout, np.uint8),
                cv2.IMREAD_COLOR
            )
            if img_color is not None:
                h, w = img_color.shape[:2]
                slice_w = w // 3
                slice_h = h // 3
                # use the same vertical band as read(): middle third in height
                top = slice_h
                bottom = 2 * slice_h
                left_crop = img_color[top:bottom, 0:slice_w]
                middle_crop = img_color[top:bottom, slice_w:2*slice_w]
                right_crop = img_color[top:bottom, 2*slice_w:w]
                #left_crop   = img_color[slice_h:2*slice_h, slice_w :slice_w + slice_w //3]
                #middle_crop = img_color[slice_h:2*slice_h, slice_w + slice_w //3:2*slice_w - slice_w //3]
                #right_crop  = img_color[slice_h:2*slice_h, 2*slice_w-slice_w //3:2*slice_w]
                # save color crops so they're easy to inspect
                print(h,w)
                cv2.imwrite('hi_left.png', left_crop)
                cv2.imwrite('hi_middle.png', middle_crop)
                cv2.imwrite('hi_right.png', right_crop)
            else:
                # fallback: try local Image_Sensing helper (OpenCV or Picamera2)
                try:
                    from .test import Image_Sensing
                    cam = Image_Sensing(width=self.width, height=self.height)
                    frame = cam.read_values()
                    if frame is not None:
                        # frame may be RGB (picam) or BGR (opencv); normalize to BGR
                        if getattr(cam, 'backend', None) == 'picam':
                            frame = frame[:, :, ::-1]
                        h, w = frame.shape[:2]
                        slice_w = w // 3
                        slice_h = h // 3
                        top = slice_h
                        bottom = 2 * slice_h
                        #left_crop = frame[top:bottom, 0:slice_w]
                        #middle_crop = frame[top:bottom, slice_w:2*slice_w]
                        #right_crop = frame[top:bottom, 2*slice_w:w]
                        left_crop   = frame[slice_h:2*slice_h, slice_w :slice_w + slice_w //3]
                        middle_crop = frame[slice_h:2*slice_h, slice_w + slice_w //3:2*slice_w - slice_w //3]
                        right_crop  = frame[slice_h:2*slice_h, 2*slice_w-slice_w //3:2*slice_w]
                        cv2.imwrite('hi_left.png', left_crop)
                        cv2.imwrite('hi_middle.png', middle_crop)
                        cv2.imwrite('hi_right.png', right_crop)
                    try:
                        cam.close()
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            # don't raise on camera errors during Sensor init
            pass

    def read(self):
        # Capture a single frame to stdout
        p = subprocess.run(
            [
                "rpicam-still",
                "--nopreview",
                "--width", str(self.width),
                "--height", str(self.height),
                "-o", "-"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        , timeout=8
        )

        img = cv2.imdecode(
            np.frombuffer(p.stdout, np.uint8),
            cv2.IMREAD_GRAYSCALE
        )

        if img is None:
            raise RuntimeError("Failed to capture camera frame")

        h, w = img.shape

        # Define three vertical slices (columns). We'll look only at the
        # bottom row of each slice and treat the box as "containing the line"
        # when more than 50% of that bottom row's pixels are darker than
        # a threshold.
        slice_w = w // 3
        bottom_row_index = h - 1

        # per-box thresholds and test
        DARK_THRESHOLD = 120  # pixel value <= this counts as "dark"
        DARK_FRACTION_REQUIRED = 0.5

        results = []
        for i in range(3):
            start = i * slice_w
            end = (i + 1) * slice_w if i < 2 else w
            row = img[bottom_row_index, start:end]
            if row.size == 0:
                # defensive: treat empty as not-detected
                results.append(255.0)
                continue
            dark_count = int((row <= DARK_THRESHOLD).sum())
            frac = dark_count / float(row.size)
            if frac > DARK_FRACTION_REQUIRED:
                # line present -> return dark (low) value
                results.append(0.0)
            else:
                # no line -> return bright (high) value
                results.append(255.0)

        return [float(v) for v in results]



class Interpreter:
    """Convert three grayscale readings into a signed offset in [-1, 1].

    Strategy (robust and simple): build a "presence" score for how much
    the line influences each sensor, using a polarity parameter:

    - polarity='dark' (default): the line is darker than the floor ->
      sensors over the line return lower ADC values, so presence = max - value
    - polarity='light': the line is lighter than the floor -> presence = value - min

    The position is the (weighted) center of mass of the presence vector
    where left maps to +1, center to 0 and right to -1 (so positive means
    the line is to the left of the robot).

    sensitivity: fraction of the observed range to treat as a minimal
    detectable signal. If the line signal is below this threshold the
    interpreter returns 0.0 (centered / no strong edge).
    """

    def __init__(self, sensitivity: float = 0.15, polarity: str = "dark"):
        if polarity not in ("dark", "light"):
            raise ValueError("polarity must be 'dark' or 'light'")
        self.sensitivity = float(sensitivity)
        self.polarity = polarity

    def process(self, readings: Iterable[float]):
        """Return offset in [-1, 1], or None when no clear line is detected.

        readings must be an iterable of three numeric values: [L, M, R].
        If the sensors cannot detect a clear line (signal too weak or
        all sensors read the same) the function returns None so callers can
        decide how to recover (for example use the last-seen offset).
        """
        vals = [float(v) for v in readings]
        if len(vals) != 3:
            raise ValueError("Interpreter.process expects three readings [L, M, R]")

        lo = min(vals)
        hi = max(vals)
        rng = hi - lo
        # If all sensors read the same, we don't have a detectable line
        if rng <= 1e-9:
            return None
        
        presence = [hi - v for v in vals]
        total = sum(presence)
        if total <= 1e-9:
            return None

        # If the maximum presence is small compared to the observed range
        # assume no clear line detected
        #max_presence = max(presence)
        #if max_presence < self.sensitivity * rng:
        #    return None

        # center-of-mass mapping: left -> +1, middle -> 0, right -> -1
        weighted = presence[0] * 1.0 + presence[1] * 0.0 + presence[2] * -1.0
        offset = weighted / total

        # clamp safety
        if offset > 1.0:
            offset = 1.0
        elif offset < -1.0:
            offset = -1.0
        print(offset)
        offset = -1 * offset
        return offset


class Controller:
    """Map interpreted offset to steering commands.

    The controller accepts a `steer_target` callable (for example
    Picarx.set_dir_servo_angle) or a Picarx-like object as `car` and a
    `scale` factor that converts offset[-1,1] to steering angle units.
    """

    def __init__(self, car_or_steer_fn, scale: float = 30.0):
        # accept either a callable or an object that provides
        # `set_dir_servo_angle(angle)`
        if callable(car_or_steer_fn):
            self._steer = car_or_steer_fn
        else:
            # try to bind to object's method
            if not hasattr(car_or_steer_fn, "set_dir_servo_angle"):
                raise TypeError("car_or_steer_fn must be callable or have set_dir_servo_angle")
            self._steer = car_or_steer_fn.set_dir_servo_angle
        self.scale = float(scale)

    def command(self, offset: float) -> float:
        """Send steering command for the given offset and return the angle.

        angle = offset * scale. Caller may clamp based on car limits; we
        convert to int before calling the steering function to match
        Picarx.set_dir_servo_angle usage.
        """
        if offset is None:
            raise ValueError("offset must be a numeric value in [-1,1]")
        # clamp offset
        if offset > 1.0:
            offset = 1.0
        elif offset < -1.0:
            offset = -1.0
        angle = int(offset * self.scale)
        # call the steering function; allow it to perform its own constraining
        self._steer(angle)
        return angle


def run_line_follow(sensor: Optional[Sensor] = None,
                    interpreter: Optional[Interpreter] = None,
                    controller: Optional[Controller] = None,
                    car: Optional[object] = None,
                    speed: int = 5,
                    sensitivity: float = 0.15,
                    polarity: str = "dark",
                    scale: float = 30.0,
                    loop_delay: float = 0.06,
                    timeout: Optional[float] = None,
                    verbose: bool = False,
                    smoothing_alpha: float = 0.0) -> float:
    """Compose sensor, interpreter and controller and run a follow loop.

    Parameters:
    - sensor/interpreter/controller: optional pre-built objects. If any is
      None and `car` is provided, the function will construct them using the
      car's `grayscale.pins` (ADC objects) and the provided sensitivity/
      polarity/scale parameters.
    - car: optional Picarx-like object providing `grayscale.pins`,
      `forward(speed)` and `stop()` methods. If provided the car will
      be commanded to move forward while steering. If not provided the
      function will only perform sensing+steering commands via the
      provided controller (which must be callable in that case).
    - speed: forward speed when `car` is provided.
    - loop_delay: seconds between updates.
    - timeout: optional seconds to run; None means until KeyboardInterrupt.

    Returns the last commanded steering angle (int) before stopping.
    """
    # Build missing pieces when car is available
    if sensor is None or interpreter is None or controller is None:
        if car is None:
            raise ValueError("If sensor/interpreter/controller are not provided, 'car' must be provided to construct them")
        # get ADC pins from the car's grayscale module
        try:
            adc0, adc1, adc2 = car.grayscale.pins
        except Exception as e:
            raise RuntimeError("Could not access car.grayscale.pins to build sensors") from e

        if sensor is None:
            sensor = Sensor()
        if interpreter is None:
            interpreter = Interpreter(sensitivity=sensitivity, polarity=polarity)
        if controller is None:
            controller = Controller(car, scale=scale)

    # Start moving forward if car provided
    last_angle = 0
    last_detected_offset = None
    lost = False
    current_speed = speed
    start = time.time()
    print("Starting line-follow loop. Press Ctrl+C to stop.")

    try:
        if car is not None:
            if verbose:
                print(f"[line_control] commanding car.forward({speed})")
            print(speed)
            car.forward(speed)

        while True:
            if verbose:
                print("[line_control] about to read sensors")
            readings = sensor.read()
            print(readings)
            if verbose:
                print(f"[line_control] sensor read -> {readings}")
            offset = interpreter.process(readings)
            # update last_detected_offset when we actually see the line
            if offset is not None:
                last_detected_offset = offset

            use_offset = offset if offset is not None else last_detected_offset
            
            # If we still don't have any offset (no line seen yet), skip steering
            if use_offset is None:
                if verbose:
                    print("[line_control] no line detected yet, skipping steering", flush=True)
            else:
                last_angle = controller.command(use_offset)
                if verbose:
                    print(f"[line_control] readings={readings}, raw_offset={'N/A' if offset is None else f'{offset:.3f}'}, ema_offset={ema_offset if ema_offset is not None else 'N/A'}, angle={last_angle}", flush=True)

            if car is not None:
                current_speed = speed
                if verbose:
                    print(f"[line_control] setting car speed to {current_speed}")
                car.forward(current_speed)
                #ctrl.maneuver_forward(car, speed=current_speed, angle=0, duration=loop_delay)
                #print("hi")
                # small sleep to make motion smooth
                time.sleep(loop_delay)
            if timeout is not None and (time.time() - start) > timeout:
                break
    except KeyboardInterrupt:
        # allow user to stop with Ctrl+C
        pass
    finally:
        if car is not None:
            car.stop()
        return last_angle


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run simple line-follow test using the RGB grayscale sensors")
    parser.add_argument("--speed", type=int, default=2, help="forward speed when running on a car")
    parser.add_argument("--sensitivity", type=float, default=0.15, help="interpreter sensitivity (fraction of range)")
    parser.add_argument("--polarity", choices=("dark", "light"), default="dark", help="is the line darker or lighter than the floor")
    parser.add_argument("--scale", type=float, default=30.0, help="steering scale (angle = offset * scale)")
    parser.add_argument("--timeout", type=float, default=None, help="seconds to run (default: until KeyboardInterrupt)")
    parser.add_argument("--verbose", action="store_true", help="print sensor/offset/angle each loop")
    parser.add_argument("--power-scale", type=float, default=None, help="optional override for motor PWM scale (px.SPEED_PWM_SCALE)")
    parser.add_argument("--power-min", type=float, default=None, help="optional override for motor PWM minimum (px.SPEED_PWM_MIN)")
    args = parser.parse_args()

    print("line_control: starting test. Press Ctrl+C to stop.")

    # Ensure repo root is available for imports when running as a script
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    car = None
    try:
        from picarx.picarx import Picarx
        car = Picarx()
        car.set_cam_tilt_angle(-35)
        print("Picarx created — starting hardware line-follow test")
        # allow runtime tuning of motor mapping without editing picarx.py
    except Exception as e:
        print(f"Picarx not available or failed to initialize: {e}")
        print("Falling back to headless read/print mode (no motors)")

    try:
        run_line_follow(car=car, speed=args.speed, sensitivity=args.sensitivity,
                        polarity=args.polarity, scale=args.scale,
                        timeout=args.timeout,
                        verbose=args.verbose)
    finally:
        if car is not None:
            try:
                ecar.stop()
                car.set_dir_servo_angle(0)
            except Exception:
                pass