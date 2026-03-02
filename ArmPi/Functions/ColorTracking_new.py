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


class BlockDetector:

    def __init__(self, target_color=('red',)):
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

        # Start movement thread
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

        coordinate = {
            'red':   (-14.5, 11.5, 1.5),
            'green': (-14.5, 5.5,  1.5),
            'blue':  (-14.5, -0.5, 1.5),
        }

        while True:

            if self.isRunning:

                if self.first_move and self.start_pick_up:
                    self.action_finish = False

                    result = self.AK.setPitchRangeMoving(
                        (self.world_X, self.world_Y - 2, 5),
                        -90, -90, 0
                    )

                    if result:
                        time.sleep(result[2] / 1000)

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
                        self.action_finish = False

                        Board.setBusServoPulse(1, self.servo1 - 280, 500)

                        servo2_angle = getAngle(
                            self.world_X,
                            self.world_Y,
                            self.rotation_angle
                        )
                        Board.setBusServoPulse(2, servo2_angle, 500)
                        time.sleep(0.8)

                        self.AK.setPitchRangeMoving(
                            (self.world_X, self.world_Y, 2),
                            -90, -90, 0, 1000
                        )
                        time.sleep(2)

                        Board.setBusServoPulse(1, self.servo1, 500)
                        time.sleep(1)

                        self.AK.setPitchRangeMoving(
                            (self.world_X, self.world_Y, 12),
                            -90, -90, 0, 1000
                        )
                        time.sleep(1)

                        drop = coordinate[self.detect_color]

                        result = self.AK.setPitchRangeMoving(
                            (drop[0], drop[1], 12),
                            -90, -90, 0
                        )
                        time.sleep(result[2] / 1000)

                        self.AK.setPitchRangeMoving(
                            drop,
                            -90, -90, 0, 1000
                        )
                        time.sleep(1)

                        Board.setBusServoPulse(1, self.servo1 - 200, 500)
                        time.sleep(0.8)

                        self.initMove()

                        self.detect_color = 'None'
                        self.first_move = True
                        self.start_pick_up = False
                        self.action_finish = True

                else:
                    time.sleep(0.01)

            else:
                time.sleep(0.01)

    # =============================
    # Detection (Former run())
    # =============================

    def process(self, img):

        if not self.isRunning:
            return img

        img_copy = img.copy()
        img_h, img_w = img.shape[:2]

        cv2.line(img, (0, img_h // 2), (img_w, img_h // 2), (0, 0, 200), 1)
        cv2.line(img, (img_w // 2, 0), (img_w // 2, img_h), (0, 0, 200), 1)

        frame_resize = cv2.resize(img_copy, self.size)
        frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
        frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

        area_max = 0
        areaMaxContour = None

        for color in color_range:
            if color in self.target_color:
                self.detect_color = color

                mask = cv2.inRange(
                    frame_lab,
                    color_range[color][0],
                    color_range[color][1]
                )

                contours = cv2.findContours(
                    mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_NONE
                )[-2]

                areaMaxContour, area_max = self.getAreaMaxContour(contours)

        if area_max > 2500:
            rect = cv2.minAreaRect(areaMaxContour)
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

            if self.action_finish and distance < 0.3:
                self.rotation_angle = rect[2]
                self.world_X = self.world_x
                self.world_Y = self.world_y
                self.start_pick_up = True

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
