import cv2
import numpy as np
from time import sleep

class DroneVisionSystem:
    def __init__(self):
       
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            print("Ошибка: не удалось открыть камеру!")
            exit()
         
     
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        self.color_lower_bound = np.array([0, 120, 70])   
        self.color_upper_bound = np.array([10, 255, 255])

        self.drone_controller = DroneController()

        self.previous_center = None

    def start(self):
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Ошибка: не удалось получить кадр!")
                break

            frame = cv2.resize(frame, (640, 480))

            processed_frame, object_center = self.process_frame(frame)

            self.control_drone(object_center)

            self.show_frame(processed_frame)
           
            if self.is_stop_condition():
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
       
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv_frame, self.color_lower_bound, self.color_upper_bound)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        object_center = None
        if contours:
            c = max(contours, key=cv2.contourArea) 
            M = cv2.moments(c)
            if M["m00"] != 0:
                object_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(frame, object_center, 5, (0, 255, 0), -1) 

        if object_center and self.previous_center:
            dx = object_center[0] - self.previous_center[0]
            dy = object_center[1] - self.previous_center[1]
            if abs(dx) > 50 or abs(dy) > 50:  
                object_center = self.previous_center
      
        self.previous_center = object_center

        return frame, object_center

    def control_drone(self, object_center):
        if object_center:
            self.drone_controller.navigate_towards(object_center)
        else:
            self.drone_controller.hover()

    def show_frame(self, frame):
        cv2.imshow("Drone Vision", frame)
        cv2.waitKey(1)

    def is_stop_condition(self):
        return cv2.waitKey(1) & 0xFF == ord('q')

class DroneController:
    def __init__(self):
        self.pid_controller = PIDController()

    def navigate_towards(self, target):
   
        print(f"Дрон движется к точке {target}")

    def hover(self):
        print("Дрон удерживает позицию")

class PIDController:
    def __init__(self, kp=0.1, ki=0.01, kd=0.1):
        self.kp = kp  
        self.ki = ki  
        self.kd = kd  

        self.previous_error = 0
        self.integral = 0

    def calculate(self, target_position, current_position):
        error = target_position - current_position
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

if __name__ == "__main__":
    drone_vision_system = DroneVisionSystem()
    drone_vision_system.start()


            
