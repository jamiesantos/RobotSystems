#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi/')

import cv2
import time
import math
import threading
import numpy as np

import Camera
from LABConfig import *
from ArmIK.Transform import *
from ArmIK.ArmMoveIK import *
import HiwonderSDK.Board as Board
from CameraCalibration.CalibrationConfig import *


class ArmController:

    def __init__(self):
        self.AK = ArmIK()
        self.servo1 = 500

        self.drop_locations = {
            'red':   (-14.5, 11.5, 1.5),
            'green': (-14.5, 5.5,  1.5),
            'blue':  (-14.5, -0.5, 1.5),
        }

    # =============================
    # Basic Motion Primitives
    # =============================

    def move_home(self):
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10),
                                    -30, -30, -90, 1500)
        time.sleep(1.5)

    def open_gripper(self):
        Board.setBusServoPulse(1, self.servo1 - 280, 500)
        time.sleep(0.8)

    def close_gripper(self):
        Board.setBusServoPulse(1, self.servo1, 500)
        time.sleep(0.8)

    def rotate_gripper(self, x, y, angle):
        servo2_angle = getAngle(x, y, angle)
        Board.setBusServoPulse(2, servo2_angle, 500)
        time.sleep(0.5)

    def move_xyz(self, position, pitch=-90, roll=-90,
                 yaw=0, duration=1000):

        result = self.AK.setPitchRangeMoving(
            position, pitch, roll, yaw, duration
        )

        if result:
            time.sleep(result[2] / 1000)

    # =============================
    # High-Level Tasks
    # =============================

    def pick_block(self, x, y, rotation):

        # Move above block
        self.move_xyz((x, y, 7), duration=1000)

        # Align gripper
        self.rotate_gripper(x, y, rotation)

        # Open
        self.open_gripper()

        # Lower
        self.move_xyz((x, y, 2), duration=1000)

        # Close
        self.close_gripper()

        # Lift
        self.move_xyz((x, y, 12), duration=1000)

    def place_block(self, color):

        drop = self.drop_locations[color]

        # Move above drop
        self.move_xyz((drop[0], drop[1], 12), duration=1500)

        # Lower
        self.move_xyz(drop, duration=1000)

        # Release
        self.open_gripper()

        # Lift back up
        self.move_xyz((drop[0], drop[1], 12), duration=800)

        self.move_home()

    def move_above_block(self, x, y):
        result = self.AK.setPitchRangeMoving(
            (x, y - 2, 5),
            -90, -90, 0
        )
        if result:
            time.sleep(result[2] / 1000)

class BlockDetector:

    def __init__(self, target_color=('red',), move_arm=True):
        self.AK = ArmIK()
        self.camera = Camera.Camera()

        self.target_color = target_color

        # ===== Servo config =====
        self.servo1 = 500

        # ===== Runtime state =====
        self.count = 0
        self.track = False
        self._stop = False
        self.get_roi = False
        self.center_list = []
        self.first_move = True
        self.isRunning = False
        self.detect_color = 'None'
        self.action_finish = True
        self.start_pick_up = False
        self.start_count_t1 = True

        self.rect = None
        self.size = (640, 480)
        self.rotation_angle = 0
        self.unreachable = False
        self.world_X, self.world_Y = 0, 0
        self.world_x, self.world_y = 0, 0
        self.last_x, self.last_y = 0, 0
        self.roi = ()
        self.t1 = 0

        self.color_list = []
        self.draw_color = (0, 0, 0)

        self.arm = ArmController()

        # Start movement thread
        if move_arm:
            self.move_thread = threading.Thread(target=self.move)
            self.move_thread.daemon = True
            self.move_thread.start()

    # =============================
    # Initialization
    # =============================

    def initMove(self):
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def reset(self):
        self.count = 0
        self.track = False
        self._stop = False
        self.get_roi = False
        self.center_list = []
        self.first_move = True
        self.detect_color = 'None'
        self.action_finish = True
        self.start_pick_up = False
        self.start_count_t1 = True

    def initialize(self):
        print("BlockDetector Init")
        self.initMove()
        self.reset()
        self.isRunning = True
        self.camera.camera_open()
        print("BlockDetector Start")

    # =============================
    # Movement Thread (ARM LOGIC)
    # =============================

    def move(self):

        while True:

            if self.isRunning:

                if self.first_move and self.start_pick_up:
                    self.action_finish = False

                    self.arm.move_above_block(self.world_X, self.world_Y)

                    self.start_pick_up = False
                    self.first_move = False
                    self.action_finish = True

                elif not self.first_move and not self.unreachable:

                    if self.track:
                        self.AK.setPitchRangeMoving(
                            (self.world_x, self.world_y - 2, 5),
                            -90, -90, 0, 20
                        )
                        time.sleep(0.02)
                        self.track = False

                    if self.start_pick_up:
                        if self.start_pick_up and self.detect_color != 'None':
                            self.arm.pick_block(
                                self.world_X,
                                self.world_Y,
                                self.rotation_angle
                            )

                            self.arm.place_block(self.detect_color)

                            self.detect_color = 'None'
                            self.first_move = True
                            self.start_pick_up = False
                            self.action_finish = True
                        else:
                            time.sleep(0.01)

                else:
                    time.sleep(0.01)

    def set_rgb(self, color):
        if color == "red":
            Board.RGB.setPixelColor(0, Board.PixelColor(255, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(255, 0, 0))
        elif color == "green":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 255, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 255, 0))
        elif color == "blue":
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 255))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 255))
        else:
            Board.RGB.setPixelColor(0, Board.PixelColor(0, 0, 0))
            Board.RGB.setPixelColor(1, Board.PixelColor(0, 0, 0))

        Board.RGB.show()
            
    # =============================
    # Detection (Former run())
    # =============================

    def process(self, img):

        if not self.isRunning:
            return img

        img_copy = img.copy()
        img_h, img_w = img.shape[:2]

        # Crosshair overlay
        cv2.line(img, (0, img_h // 2), (img_w, img_h // 2), (0, 0, 200), 1)
        cv2.line(img, (img_w // 2, 0), (img_w // 2, img_h), (0, 0, 200), 1)

        frame_resize = cv2.resize(img_copy, self.size)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)

        # === ROI optimization (NEW) ===
        if self.get_roi and not self.start_pick_up:
            self.get_roi = False
            frame_gb = getMaskROI(frame_gb, self.roi, self.size)

        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

        max_area = 0
        areaMaxContour_max = None
        color_area_max = None

        # === Find largest valid contour ===
        for color in color_range:
            if color in self.target_color:
                mask = cv2.inRange(
                    frame_lab,
                    color_range[color][0],
                    color_range[color][1]
                )

                opened = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, np.ones((6, 6), np.uint8)
                )
                closed = cv2.morphologyEx(
                    opened, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)
                )

                contours = cv2.findContours(
                    closed,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )[-2]

                areaMaxContour, area_max = self.getAreaMaxContour(contours)

                if areaMaxContour is not None and area_max > max_area:
                    max_area = area_max
                    areaMaxContour_max = areaMaxContour
                    color_area_max = color

        # === If block found ===
        if max_area > 2500:

            rect = cv2.minAreaRect(areaMaxContour_max)
            box = np.int0(cv2.boxPoints(rect))

            self.roi = getROI(box)
            self.get_roi = True

            img_centerx, img_centery = getCenter(
                rect,
                self.roi,
                self.size,
                square_length
            )

            self.world_x, self.world_y = convertCoordinate(
                img_centerx, img_centery, self.size
            )

            distance = math.sqrt(
                (self.world_x - self.last_x) ** 2 +
                (self.world_y - self.last_y) ** 2
            )

            self.last_x = self.world_x
            self.last_y = self.world_y
            self.track = True

            # === Multi-frame color confirmation (NEW) ===
            if color_area_max == 'red':
                color_id = 1
            elif color_area_max == 'green':
                color_id = 2
            elif color_area_max == 'blue':
                color_id = 3
            else:
                color_id = 0

            self.color_list.append(color_id)

            if len(self.color_list) == 3:
                avg_color = int(round(np.mean(np.array(self.color_list))))
                self.color_list = []

                if avg_color == 1:
                    self.detect_color = 'red'
                    self.draw_color = (0, 0, 255)
                elif avg_color == 2:
                    self.detect_color = 'green'
                    self.draw_color = (0, 255, 0)
                elif avg_color == 3:
                    self.detect_color = 'blue'
                    self.draw_color = (255, 0, 0)
                else:
                    self.detect_color = 'None'
                    self.draw_color = (0, 0, 0)

                self.set_rgb(self.detect_color)

            # === Stable pickup trigger ===
            if self.action_finish and distance < 0.3:
                self.rotation_angle = rect[2]
                self.world_X = self.world_x
                self.world_Y = self.world_y
                self.start_pick_up = True

            cv2.drawContours(img, [box], -1, self.draw_color, 2)

        else:
            self.detect_color = 'None'
            self.draw_color = (0, 0, 0)
            self.set_rgb('None')

        # === On-screen display ===
        cv2.putText(
            img,
            "Color: " + self.detect_color,
            (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.draw_color,
            2
        )

        return img

    def getAreaMaxContour(self, contours):
        contour_area_max = 0
        area_max_contour = None

        for c in contours:
            area = abs(cv2.contourArea(c))
            if area > contour_area_max and area > 300:
                contour_area_max = area
                area_max_contour = c

        return area_max_contour, contour_area_max

    # =============================
    # Main Loop
    # =============================

    def run(self):
        while True:
            img = self.camera.frame
            if img is not None:
                frame = img.copy()
                output = self.process(frame)
                cv2.imshow("Frame", output)

                if cv2.waitKey(1) == 27:
                    break

        self.cleanup()

    def cleanup(self):
        self.isRunning = False
        self.camera.camera_close()
        cv2.destroyAllWindows()


# =============================
# MAIN
# =============================

if __name__ == '__main__':
    detector = BlockDetector(target_color=('red',))
    detector.initialize()
    detector.run()
