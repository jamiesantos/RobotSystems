"""
A set of higher-level actions for the Picar-X robot, built on top of
the low-level Picarx class.
"""
import time
import logging
import sys
from typing import Iterable, Tuple
import picarx_improved
from picarx_improved import Picarx

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def clamp_angle(px, angle):
    try:
        return int(max(px.DIR_MIN, min(px.DIR_MAX, angle)))
    except Exception:
        return int(angle)
    

def drive_time(px: Picarx,
               speed: int,
               steering_angle: int = 0,
               duration: float = 1.0,
               forward: bool = True,
               settle: float = 0.08) -> None:
    """Drive the robot for a fixed time with a steering angle.

    Sets steering angle, waits briefly for servo to settle, runs the motors
    for `duration` seconds, then stops.
    """
    # Constrain angle to class limits if available
    try:
        steering_angle = int(max(px.DIR_MIN, min(px.DIR_MAX, steering_angle)))
    except Exception:
        steering_angle = int(steering_angle)

    logger.debug("drive_time: setting steering_angle=%s", steering_angle)
    px.set_dir_servo_angle(steering_angle)
    time.sleep(settle)

    if forward:
        logger.debug("drive_time: starting forward speed=%s duration=%.3fs", speed, duration)
        px.forward(speed)
    else:
        logger.debug("drive_time: starting backward speed=%s duration=%.3fs", speed, duration)
        px.backward(speed)

    time.sleep(max(0.0, float(duration)))

    px.stop()
    logger.debug("drive_time: finished (stopped)")
    
def maneuver_forward(px: Picarx, speed=30, angle=0, duration=1.0):
    angle = clamp_angle(px, angle)
    logger.info("Maneuver: forward speed=%s angle=%s duration=%.2f", speed, angle, duration)
    drive_time(px, speed=speed, steering_angle=angle, duration=duration, forward=True)
    
def maneuver_backward(px: Picarx, speed=30, angle=0, duration=1.0):
    angle = clamp_angle(px, angle)
    logger.info("Maneuver: backward speed=%s angle=%s duration=%.2f", speed, angle, duration)
    drive_time(px, speed=speed, steering_angle=angle, duration=duration, forward=False)
    
def maneuver_parallel_park_left(px: Picarx, speed=25):
    """
    Approximate parallel park left using a short scripted sequence.
    """
    logger.info("Parallel park left")
    # approach forward slightly
    drive_time(px, speed=40, steering_angle=0, duration=2, forward=True)
    # turn hard right and reverse into spot
    drive_time(px, speed=40, steering_angle=20, duration=2, forward=False)
    # straighten and reverse a bit
    drive_time(px, speed=50, steering_angle=-20, duration=1.5, forward=False)
    # forward to align
    #drive_time(px, speed=20, steering_angle=0, duration=1, forward=True)
    drive_time(px, speed=40, steering_angle=-10, duration=1, forward=True)
    
def maneuver_parallel_park_right(px: Picarx, speed=25):
    logger.info("Parallel park right (approx)")
    drive_time(px, speed=20, steering_angle=0, duration=0.6, forward=True)
    drive_time(px, speed=25, steering_angle=-30, duration=0.8, forward=False)
    drive_time(px, speed=20, steering_angle=0, duration=0.4, forward=False)
    drive_time(px, speed=15, steering_angle=20, duration=0.5, forward=True)
    drive_time(px, speed=10, steering_angle=0, duration=0.2, forward=True)
    
def maneuver_k_turn(px: Picarx, start_left=True, speed=50):
    """Three-point turn (K-turn) approximation.

    Steps (start_left=True):
      1) forward-left
      2) backward-right
      3) forward-left (short) to complete

    Inverse the angles if start_left=False.
    """
    if start_left:
        logger.info("Three-point turn: start left")
        drive_time(px, speed=speed, steering_angle=0, duration=3, forward=True)
        drive_time(px, speed=speed, steering_angle=-20, duration=6, forward=False)
        drive_time(px, speed=int(speed * 4.0), steering_angle=20, duration=1, forward=True)
    else:
        logger.info("Three-point turn: start right")
        drive_time(px, speed=speed, steering_angle=-30, duration=0.7, forward=True)
        drive_time(px, speed=speed, steering_angle=30, duration=0.9, forward=False)
        drive_time(px, speed=int(speed * 1.0), steering_angle=-30, duration=0.6, forward=True)
        
def print_help():
    print("Commands:")
    print("  w - forward straight")
    print("  s - backward straight")
    print("  W - forward with steering angle")
    print("  S - backward with steering angle")
    print("  p - parallel park")
    print("  P - parallel park right")
    print("  k - three-point turn")
    print("  K - three-point turn (start right)")
    print("  h - help")
    print("  q - quit")

def main():
    px = Picarx()
    try:
        print_help()
        while True:
            cmd = input('\nEnter command (h for help): ').strip()
            if not cmd:
                continue
            if cmd == 'q':
                print('Quitting')
                break
            elif cmd == 'h':
                print_help()
                continue
            elif cmd == 'w':
                maneuver_forward(px, speed=30, angle=0, duration=1.0)
            elif cmd == 's':
                maneuver_backward(px, speed=30, angle=0, duration=1.0)
            elif cmd == 'W':
                try:
                    angle = int(input('Steering angle (negative=right, positive=left): ').strip())
                except Exception:
                    print('Invalid angle')
                    continue
                maneuver_forward(px, speed=30, angle=angle, duration=1.0)
            elif cmd == 'S':
                try:
                    angle = int(input('Steering angle (negative=right, positive=left): ').strip())
                except Exception:
                    print('Invalid angle')
                    continue
                maneuver_backward(px, speed=30, angle=angle, duration=1.0)
            elif cmd == 'p':
                maneuver_parallel_park_left(px)
            elif cmd == 'P':
                maneuver_parallel_park_right(px)
            elif cmd == 'k':
                maneuver_k_turn(px, start_left=True)
            elif cmd == 'K':
                maneuver_k_turn(px, start_left=False)
            else:
                print('Unknown command:', cmd)
    except KeyboardInterrupt:
        print('\nInterrupted by user')
    finally:
        px.close()
        print('Robot closed')


if __name__ == '__main__':
    main()
