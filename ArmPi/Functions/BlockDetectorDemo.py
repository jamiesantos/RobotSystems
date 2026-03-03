#!/usr/bin/python3
# coding=utf8

import cv2
import math
import numpy as np

from ColorTracking_new import BlockDetector   # adjust if filename differs
from ArmIK.Transform import *
from CameraCalibration.CalibrationConfig import *
from LABConfig import *


def main():

    # Disable arm motion
    detector = BlockDetector(target_color=('red',), move_arm=False)
    detector.initialize()

    print("Vision-only block detection demo started")

    while True:
        img = detector.camera.frame

        if img is not None:
            frame = img.copy()
            output = process_frame(detector, frame)
            cv2.imshow("Block Location Demo", output)

            if cv2.waitKey(1) == 27:
                break

    detector.cleanup()


def process_frame(detector, img):

    img_copy = img.copy()
    img_h, img_w = img.shape[:2]

    cv2.line(img, (0, img_h // 2), (img_w, img_h // 2), (0, 0, 200), 1)
    cv2.line(img, (img_w // 2, 0), (img_w // 2, img_h), (0, 0, 200), 1)

    frame_resize = cv2.resize(img_copy, detector.size)
    frame_gb = cv2.GaussianBlur(frame_resize, (11, 11), 11)
    frame_lab = cv2.cvtColor(frame_gb, cv2.COLOR_BGR2LAB)

    for color in color_range:
        if color in detector.target_color:

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

            contour, area = detector.getAreaMaxContour(contours)

            if contour is not None and area > 2500:

                rect = cv2.minAreaRect(contour)
                box = np.int0(cv2.boxPoints(rect))

                roi = getROI(box)

                img_centerx, img_centery = getCenter(
                    rect,
                    roi,
                    detector.size,
                    square_length
                )

                world_x, world_y = convertCoordinate(
                    img_centerx,
                    img_centery,
                    detector.size
                )

                # Draw bounding box
                cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

                # Label world coordinates
                label = f"({world_x:.2f}, {world_y:.2f})"
                cv2.putText(
                    img,
                    label,
                    (int(box[0][0]), int(box[0][1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return img


if __name__ == "__main__":
    main()
