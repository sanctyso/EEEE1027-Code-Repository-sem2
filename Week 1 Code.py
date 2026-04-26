import time
import RPi.GPIO as GPIO
from threading import Lock

GPIO.setmode(GPIO.BOARD)

# ---------------- Motor pins ----------------
ENA, IN1, IN2 = 13, 12, 11
ENB, IN3, IN4 = 18, 16, 15

GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)

pwmL = GPIO.PWM(ENA, 500)
pwmR = GPIO.PWM(ENB, 500)
pwmL.start(0)
pwmR.start(0)

# ---------------- Encoder pins ----------------
L_ENC = 33
R_ENC = 35

GPIO.setup([L_ENC, R_ENC], GPIO.IN, pull_up_down=GPIO.PUD_UP)

# ---------------- Wheel / encoder settings ----------------
PULSES_PER_REV = 20
WHEEL_DIAMETER_M = 0.065

WHEEL_CIRC_M = 3.1415926 * WHEEL_DIAMETER_M
DIST_PER_PULSE = WHEEL_CIRC_M / PULSES_PER_REV

# --------- TURN CALIBRATION (SEPARATE) ---------
PULSES_FOR_90_L = 38 # ↓ reduce if left overshoots
PULSES_FOR_90_R = 40 # right already correct

# ---------------- Control tuning ----------------
Kp_speed = 180
Kp_balance = 70

# ---------------- Encoder counters ----------------
lock = Lock()
l_count = 0
r_count = 0

def _enc_left(_):
global l_count
with lock:
l_count += 1

def _enc_right(_):
global r_count
with lock:
r_count += 1

GPIO.add_event_detect(L_ENC, GPIO.RISING, callback=_enc_left, bouncetime=1)
GPIO.add_event_detect(R_ENC, GPIO.RISING, callback=_enc_right, bouncetime=1)

def get_counts():
with lock:
return l_count, r_count

_last_t = time.time()
_last_l = 0
_last_r = 0

def reset_encoders():
global l_count, r_count, _last_t, _last_l, _last_r
with lock:
l_count = 0
r_count = 0
_last_t = time.time()
_last_l = 0
_last_r = 0

def get_distance_m():
lc, rc = get_counts()
return ((lc + rc) / 2.0) * DIST_PER_PULSE

def get_wheel_speeds_mps():
global _last_t, _last_l, _last_r
now = time.time()
dt = now - _last_t
if dt <= 0:
return 0.0, 0.0

lc, rc = get_counts()
vl = (lc - _last_l) * DIST_PER_PULSE / dt
vr = (rc - _last_r) * DIST_PER_PULSE / dt

_last_t = now
_last_l = lc
_last_r = rc
return vl, vr

# ---------------- Motor helpers ----------------
def _clip(x, lo=0, hi=100):
return max(lo, min(hi, x))

def _set_pwm(l, r):
pwmL.ChangeDutyCycle(_clip(l))
pwmR.ChangeDutyCycle(_clip(r))

def coast_stop():
_set_pwm(0, 0)
GPIO.output([IN1, IN2, IN3, IN4], 0)

def safe_change_dir():
coast_stop()
time.sleep(0.08)

def left_dir(fwd):
GPIO.output(IN1, 1 if fwd else 0)
GPIO.output(IN2, 0 if fwd else 1)

def right_dir(fwd):
GPIO.output(IN3, 1 if fwd else 0)
GPIO.output(IN4, 0 if fwd else 1)

# ---------------- Control state ----------------
mode = "idle"
target_speed_mps = 0.25
pwm_base = 50

def start_forward():
global mode
safe_change_dir()
left_dir(True)
right_dir(True)
mode = "fwd"

def start_backward():
global mode
safe_change_dir()
left_dir(False)
right_dir(False)
mode = "bwd"

def stop():
global mode
mode = "idle"
coast_stop()

def pivot_left_open(speed=55):
safe_change_dir()
left_dir(True)
right_dir(False)
_set_pwm(speed, speed)

def pivot_right_open(speed=55):
safe_change_dir()
left_dir(False)
right_dir(True)
_set_pwm(speed, speed)

def control_step():
global pwm_base
if mode not in ("fwd", "bwd"):
return

vl, vr = get_wheel_speeds_mps()
v_meas = abs((vl + vr) / 2.0)

error = target_speed_mps - v_meas
pwm_base = _clip(pwm_base + Kp_speed * error)

balance = Kp_balance * (vl - vr)
_set_pwm(pwm_base - balance, pwm_base + balance)

# ---------------- TURN FUNCTIONS ----------------
def turn_left_angle(angle, speed=55):
target = PULSES_FOR_90_L * (angle / 90.0)
reset_encoders()
pivot_left_open(speed)

while True:
lc, rc = get_counts()
if (lc + rc) / 2 >= target:
break
time.sleep(0.005)
stop()

def turn_right_angle(angle, speed=55):
target = PULSES_FOR_90_R * (angle / 90.0)
reset_encoders()
pivot_right_open(speed)

while True:
lc, rc = get_counts()
if (lc + rc) / 2 >= target:
break
time.sleep(0.005)
stop()

# ---------------- CLI ----------------
print("""
v X -> set speed (m/s)
f -> forward
b -> backward
l 90 -> left turn
r 90 -> right turn
d -> diagnostics
s -> stop
q -> quit
""")

try:
while True:
if mode in ("fwd", "bwd"):
control_step()
time.sleep(0.03)

cmd = input(">> ").split()
if not cmd:
continue

if cmd[0] == "v":
target_speed_mps = float(cmd[1])

elif cmd[0] == "f":
start_forward()

elif cmd[0] == "b":
start_backward()

elif cmd[0] == "l":
stop()
turn_left_angle(int(cmd[1]))

elif cmd[0] == "r":
stop()
turn_right_angle(int(cmd[1]))

elif cmd[0] == "d":
vl, vr = get_wheel_speeds_mps()
print(f"vl={vl:.3f} vr={vr:.3f} dist={get_distance_m():.3f} counts={get_counts()}")

elif cmd[0] == "s":
stop()

elif cmd[0] == "q":
break

finally:
stop()
pwmL.stop()
pwmR.stop()
GPIO.cleanup()