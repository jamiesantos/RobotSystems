
from picarx_improved import Picarx
import time
import logging
import numpy as np
from vilib import Vilib
import cv2


logging_format = "%(asctime)s: %(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO, datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)

class Sensors():
    def __init__(self, picar):
        self.px = picar
    
    def photo(self): 
        Vilib.take_photo(photo_name = "test", path = "picarx")

class Interpretation(): 
    def __init__(self, dark_sen = 0, light_sen = 1):
        self.dark_sen = dark_sen  # dark sensitivity
        self.light_sen = light_sen  # light sensitivity

    def process_im(self): 
        im = cv2.imread("picarx/test.jpg")
        _,width, _ = im.shape
        gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        thresh = cv2.Canny(gray_im, 120, 255)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0

        max_contours = max(contours, key=cv2.contourArea)
        cv2.drawContours(im, [max_contours], 0, (0,255,0), 2)
        cv2.imwrite("picarx/test2.jpg", im)
        cv2.imwrite("picarx/test3.jpg", thresh)
        m = cv2.moments(max_contours)

        if m['m00'] !=0:
            cx = m['m10'] / m['m00']
            cy = m['m01'] / m['m00']
            center = (cx-(width/2))/(width/2)
            logging.debug(f"center = {center}")
            return center 
        else: 
            logging.debug("no moments found")
            return 0
            
    def processing(self, center):
        cx = np.array(center)
        if cx < 0: 
            turn = -15
        elif cx > 0: 
            turn = 15  # turn left
        else: 
            turn = 0
        return turn
        
class Controller(): 
    def __init__(self, picar):
        self.picar = picar
        self.sensors = Sensors(picar)
        self.interpretation = Interpretation()
    
    def straight(self): 
        self.picar.set_dir_servo_angle(0)
        self.picar.forward(25)

    def turning(self, value): 
        self.picar.set_dir_servo_angle(value)
        self.picar.forward(25)
        logging.debug(f"turning: {value}")

    def stop(self): 
        self.picar.stop()
    
    def line_following(self): 
        photo = self.sensors.photo()
        image = self.interpretation.process_im()
        turn = self.interpretation.processing(image)
        self.picar.set_cam_tilt_angle(-35)

        if turn == 0 : 
            self.straight()
        elif turn > 0: 
            self.turning(turn)
        elif turn < 0: 
            self.turning(turn)

if __name__ == "__main__":
    px = Picarx()
    Vilib.camera_start()
    time.sleep(0.5)
    
    control = Controller(px)
    interpret = Interpretation(px)
    try:
        while True:
            control.line_following()
    except KeyboardInterrupt:
        pass
    finally:
        if control.picar is not None:
            control.picar.stop()
            
        
        
   









