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
import threading
from collections import deque

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from readerwriterlock import rwlock

from sim_robot_hat.rossros import Bus
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

class Sensor:
    """Read three ADC channels and return raw values.

    The constructor accepts either ADC instances or channel identifiers
    (for example 'A0', 'A1', 'A2') which will be used to construct
    ADC objects.
    """

    def __init__(self, left, middle, right):
        # Accept ADC objects or channel specifiers
        # Accept ADC-like objects (from either sim_robot_hat.ADC or robot_hat.ADC)
        # or channel specifiers (e.g. 'A0'). Use duck-typing: if the object
        # has a callable `read()` method treat it as an ADC instance.
        self.left = left if (hasattr(left, 'read') and callable(getattr(left, 'read'))) else ADC(left)
        self.middle = middle if (hasattr(middle, 'read') and callable(getattr(middle, 'read'))) else ADC(middle)
        self.right = right if (hasattr(right, 'read') and callable(getattr(right, 'read'))) else ADC(right)
        # Quick camera check: attempt to capture a single frame and save as hi.png
        # This is best-effort and must not raise on failure (we don't want to
        # prevent sensor construction if the camera or OpenCV/Picamera2 isn't
        # available).
        try:
            # import local helper that wraps Picamera2/OpenCV
            from .test import Image_Sensing
            import cv2

            try:
                cam = Image_Sensing()
                frame = cam.read_values()
                if frame is not None:
                    # Picamera2 returns RGB frames while OpenCV uses BGR.
                    # Image_Sensing sets cam.backend to 'picam' when Picamera2 is used.
                    if getattr(cam, 'backend', None) == 'picam':
                        # convert RGB -> BGR for correct colors when saving with OpenCV
                        frame = frame[:, :, ::-1]
                    cv2.imwrite('hi.png', frame)
                cam.close()
            except Exception:
                # ignore camera capture errors
                try:
                    cam.close()
                except Exception:
                    pass
        except Exception:
            # either Image_Sensing or cv2 isn't available — skip camera check
            pass

    def read(self) -> List[float]:
        """Poll the three ADC channels and return a list [L, M, R]."""
        print("left: " + str(self.left.read()))
        print("middle: " + str(self.middle.read()))
        print("right: " + str(self.right.read()))
        return [self.left.read(), self.middle.read(), self.right.read()]
    
    def run(self, output_bus, shutdown_event, delay=0.06):
        while not shutdown_event.is_set():
            readings = self.read()
            output_bus.write(readings)
            time.sleep(delay)


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
        self.last_version = -1


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

    def run(self, input_bus, output_bus, shutdown_event, delay=0.01):

        while not shutdown_event.is_set():

            readings, version = input_bus.read()

            if readings is None or version == self.last_version:
                time.sleep(delay)
                continue

            self.last_version = version

            offset = self.process(readings)

            if offset is not None:
                output_bus.write(offset)

            time.sleep(delay)



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
        self.last_version = -1

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
    
    def run(self, input_bus, shutdown_event, delay=0.01):

        while not shutdown_event.is_set():

            offset, version = input_bus.read()

            if offset is None or version == self.last_version:
                time.sleep(delay)
                continue

            self.last_version = version

            self.command(offset)

            time.sleep(delay)


def handle_exception(future):
    exception = future.exception()
    if exception:
        print(f'Exception in worker thread: {exception}')


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
            sensor = Sensor(adc0, adc1, adc2)
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

    if car is not None:
        if verbose:
            print(f"[line_control] commanding car.forward({speed})")
        print(speed)
        car.forward(speed)

    sensor_bus = Bus("sensor_bus")
    offset_bus = Bus("offset_bus")

    '''
    while True:
        if verbose:
            print("[line_control] about to read sensors")
        readings = sensor.read()
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
            print("hi")
            # small sleep to make motion smooth
            time.sleep(loop_delay)
        if timeout is not None and (time.time() - start) > timeout:
            break
    '''
    shutdown_event = Event()

    with ThreadPoolExecutor(max_workers=3) as executor:

        futures = []

        futures.append(
            executor.submit(sensor.run,
                            sensor_bus,
                            shutdown_event,
                            loop_delay)
        )

        futures.append(
            executor.submit(interpreter.run,
                            sensor_bus,
                            offset_bus,
                            shutdown_event)
        )

        futures.append(
            executor.submit(controller.run,
                            offset_bus,
                            shutdown_event)
        )

        for f in futures:
            f.add_done_callback(handle_exception)

        try:
            if car is not None:
                car.forward(speed)

            start = time.time()

            while not shutdown_event.is_set():
                if timeout is not None and (time.time() - start) > timeout:
                    break
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Ctrl-C received — shutting down")

        finally:
            shutdown_event.set()
            executor.shutdown(wait=True)

            if car is not None:
                car.stop()

    '''
    try:
        if car is not None:
            car.forward(speed)

        start = time.time()

        while True:
            if timeout is not None and (time.time() - start) > timeout:
                break
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        if car is not None:
            car.stop()
    '''

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
                car.stop()
                car.set_dir_servo_angle(0)
            except Exception:
                pass