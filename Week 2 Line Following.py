import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
import time

# ================= GPIO SETUP =================
GPIO.setmode(GPIO.BOARD)

ENA, IN1, IN2 = 13, 12, 11
ENB, IN3, IN4 = 18, 16, 15
GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)

pwmA = GPIO.PWM(ENA, 300)
pwmB = GPIO.PWM(ENB, 300)
pwmA.start(0)
pwmB.start(0)

# ================= CAMERA SETUP =================
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# ================= PARAMETERS =================
THRESHOLD = 100
FORWARD_SPEED = 80
MAX_TURN_SPEED = 55
CENTER_TOLERANCE = 64  # <- increase this for wider “forward” range
last_turn = "left"

# ================= MOTOR FUNCTIONS =================
def move_forward():
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(FORWARD_SPEED)
    pwmB.ChangeDutyCycle(FORWARD_SPEED)

def turn_left(speed=MAX_TURN_SPEED):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def turn_right(speed=MAX_TURN_SPEED):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)

def stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

# ================= MAIN LOOP =================
try:
    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        h, w = binary.shape
        roi = binary[h//2:, :]  # use lower half
        moments = cv2.moments(roi)
       
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            deviation = cx - w//2

            # Use CENTER_TOLERANCE to move forward
            if abs(deviation) < CENTER_TOLERANCE:
                move_forward()
            elif deviation < 0:
                # proportional turn speed
                speed = min(MAX_TURN_SPEED, int(MAX_TURN_SPEED * abs(deviation) / (w//2)))
                turn_left(speed)
                last_turn = "left"
            else:
                speed = min(MAX_TURN_SPEED, int(MAX_TURN_SPEED * abs(deviation) / (w//2)))
                turn_right(speed)
                last_turn = "right"
        else:
            # Line lost, continue last turn
            if last_turn == "left":
                turn_left()
            else:
                turn_right()

        # Debug window
        cv2.imshow("Line Detection", binary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    stop()
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()