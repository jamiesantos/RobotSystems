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


# =========================================================
# Arm Controller (Motion Only)
# =========================================================

class ArmController:

    def __init__(self):
        self.AK = ArmIK()
        self.servo1 = 500

        self.isRunning = False
        self.start_pick_up = False
        self.detect_color = 'None'
        self.world_X = 0
        self.world_Y = 0
        self.rotation_angle = 0

        self.thread = threading.Thread(target=self.move)
        self.thread.daemon = True
        self.thread.start()

    def initMove(self):
        Board.setBusServoPulse(1, self.servo1 - 50, 300)
        Board.setBusServoPulse(2, 500, 500)
        self.AK.setPitchRangeMoving((0, 10, 10), -30, -30, -90, 1500)

    def trigger_pickup(self, x, y, angle, color):
        self.world_X = x
        self.world_Y = y
        self.rotation_angle = angle
        self.detect_color = color
        self.start_pick_up = True

    def move(self):

        coordinate = {
            'red':   (-14.5, 11.5, 1.5),
            'green': (-14.5, 5.5,  1.5),
            'blue':  (-14.5, -0.5, 1.5),
        }

        while True:

            if self.isRunning and self.start_pick_up and self.detect_color != 'None':

                # Open gripper
                Board.setBusServoPulse(1, self.servo1 - 280, 500)

                # Rotate wrist
                servo2_angle = getAngle(
                    self.world_X,
                    self.world_Y,
                    self.rotation_angle
                )
                Board.setBusServoPulse(2, servo2_angle, 500)
                time.sleep(0.8)

                # Move down
                self.AK.setPitchRangeMoving(
                    (self.world_X, self.world_Y, 2),
                    -90, -90, 0, 1000
                )
                time.sleep(1.5)

                # Close gripper
                Board.setBusServoPulse(1, self.servo1, 500)
                time.sleep(1)

                # Lift
                self.AK.setPitchRangeMoving(
                    (self.world_X, self.world_Y, 12),
                    -90, -90, 0, 1000
                )
                time.sleep(1)

                # Move to drop location
                drop = coordinate[self.detect_color]

                result = self.AK.setPitchRangeMoving(
                    (drop[0], drop[1], 12),
                    -90, -90, 0
                )
                time.sleep(result[2] / 1000)

                self.AK.setPitchRangeMoving(drop, -90, -90, 0, 1000)
                time.sleep(1)

                # Release
                Board.setBusServoPulse(1, self.servo1 - 200, 500)
                time.sleep(0.8)

                # Return home
                self.initMove()

                # Reset trigger
                self.start_pick_up = False
                self.detect_color = 'None'

            else:
                time.sleep(0.01)


# =========================================================
# Block Detector (Vision Only)
# =========================================================

class BlockDetector:

    def __init__(self, target_color=('red',)):

        self.camera = Camera.Camera()
        self.target_color = target_color

        self.arm = ArmController()

        self.size = (640, 480)
        self.detect_color = 'None'
        self.draw_color = (0, 0, 0)

        self.color_list = []
        self.last_x, self.last_y = 0, 0
        self.world_x, self.world_y = 0, 0

        self.isRunning = False

    def initialize(self):
        print("BlockDetector Init")
        self.arm.initMove()
        self.arm.isRunning = True
        self.camera.camera_open()
        self.isRunning = True
        print("BlockDetector Start")

    def set_rgb(self, color):
        if color == "red":
            c = (255, 0, 0)
        elif color == "green":
            c = (0, 255, 0)
        elif color == "blue":
            c = (0, 0, 255)
        else:
            c = (0, 0, 0)

        Board.RGB.setPixelColor(0, Board.PixelColor(*c))
        Board.RGB.setPixelColor(1, Board.PixelColor(*c))
        Board.RGB.show()

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

        max_area = 0
        areaMaxContour_max = None
        color_area_max = None

        for color in color_range:
            if color in self.target_color:

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

                contour, area = self.getAreaMaxContour(contours)

                if contour is not None and area > max_area:
                    max_area = area
                    areaMaxContour_max = contour
                    color_area_max = color

        if max_area > 2500:

            rect = cv2.minAreaRect(areaMaxContour_max)
            box = np.int0(cv2.boxPoints(rect))

            img_centerx, img_centery = getCenter(
                rect,
                getROI(box),
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

            self.detect_color = color_area_max
            self.set_rgb(self.detect_color)

            if distance < 0.3 and not self.arm.start_pick_up:
                self.arm.trigger_pickup(
                    self.world_x,
                    self.world_y,
                    rect[2],
                    self.detect_color
                )

            cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

        else:
            self.detect_color = 'None'
            self.set_rgb('None')

        cv2.putText(
            img,
            "Color: " + self.detect_color,
            (10, img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
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

    def run(self):
        while True:
            img = self.camera.frame
            if img is not None:
                output = self.process(img.copy())
                cv2.imshow("Frame", output)
                if cv2.waitKey(1) == 27:
                    break

        self.cleanup()

    def cleanup(self):
        self.isRunning = False
        self.arm.isRunning = False
        self.camera.camera_close()
        cv2.destroyAllWindows()


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':
    detector = BlockDetector(target_color=('red', 'green', 'blue'))
    detector.initialize()
    detector.run()
