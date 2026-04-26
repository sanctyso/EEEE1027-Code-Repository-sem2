import time
import os
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2

GPIO.setmode(GPIO.BOARD)

# ===================== HARDWARE / PINS =====================
ENA, IN1, IN2 = 13, 12, 11
ENB, IN3, IN4 = 18, 16, 15

GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)

pwmL = GPIO.PWM(ENA, 500)
pwmR = GPIO.PWM(ENB, 500)
pwmL.start(0)
pwmR.start(0)

def _clip(x, lo=0, hi=100):
    return max(lo, min(hi, float(x)))

# --- Unified Motor Control supporting Negative Values (Pivot Turns!) ---
def set_motors(left_speed, right_speed):
    # Left Motor Direction
    if left_speed >= 0:
        GPIO.output(IN1, 1); GPIO.output(IN2, 0)
    else:
        GPIO.output(IN1, 0); GPIO.output(IN2, 1)
       
    # Right Motor Direction
    if right_speed >= 0:
        GPIO.output(IN3, 1); GPIO.output(IN4, 0)
    else:
        GPIO.output(IN3, 0); GPIO.output(IN4, 1)

    # Apply absolute speed to PWM
    pwmL.ChangeDutyCycle(_clip(abs(left_speed)))
    pwmR.ChangeDutyCycle(_clip(abs(right_speed)))

def coast_stop():
    set_motors(0, 0)
    GPIO.output([IN1, IN2, IN3, IN4], 0)

# ===================== CAMERA SETUP =====================
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("FULL: Line Follow + Detect Saved Templates + Geometry (PID + Anti-Jitter + Smart Brake)")
print("Keys:  t = save detected symbol as template,  q = quit")

# ===================== CONFIGURATION =====================
SHOW_DEBUG = True       # SET TO FALSE FOR COMPETITION/HIGH-SPEED RUNS to boost FPS!
SYMBOL_INTERVAL = 0.2   # Check for symbols every 0.2 seconds (5 FPS) to keep PID lightning fast

# --- LINE FOLLOW ---
THRESH_LINE = 90
MIN_LINE_AREA = 400
STEER_LIMIT = 100
kernel5 = np.ones((5, 5), np.uint8)

# --- LOOK-AHEAD (Adjusted to see 90-degree turns earlier) ---
ROI_START = 0.50        # Look starting from the middle of the screen (50%)

# --- PID & ANTI-JITTER ---
KP_LINE = 0.18      
KI_LINE = 0.00001    
KD_LINE = 0.2      # Increased slightly to fight momentum on 90-degree turns
I_CLAMP = 9000      
D_CLAMP = 300      
DEADBAND = 15       # Ignore errors smaller than 15 pixels on straightaways

# --- SPEED ---
MIN_BASE_PWM = 32      # Drop to 0 to allow perfect on-the-spot pivots
MAX_BASE_PWM = 45     # Max forward speed
RECOVERY_SPEED = 45   # Speed used when searching for a lost line

_pid_i = 0.0
_last_err = 0.0
_smoothed_derr = 0.0  # Low-pass filter memory
_last_t = time.time()
_last_sym_time = 0.0  

# ===================== TEMPLATE SAVE / LOAD =====================
SAVE_DIR = "templates"
os.makedirs(SAVE_DIR, exist_ok=True)

def _next_template_index():
    mx = 0
    for f in os.listdir(SAVE_DIR):
        if f.startswith("template_") and f.lower().endswith(".png"):
            try: mx = max(mx, int(f.split("_")[1].split(".")[0]))
            except: pass
    return mx + 1

tpl_idx = _next_template_index()

def preprocess_template(crop_gray, size=(120, 120)):
    crop_eq = cv2.equalizeHist(crop_gray)
    h, w = crop_eq.shape
    diff = abs(h - w)
    top = bottom = left = right = 0
    if h > w:
        left = diff // 2; right = diff - left
    elif w > h:
        top = diff // 2; bottom = diff - top

    crop_sq = cv2.copyMakeBorder(crop_eq, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    crop_pad = cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(crop_pad, size, interpolation=cv2.INTER_AREA)

def load_templates():
    tpls = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.lower().endswith(".png"):
            img = cv2.imread(os.path.join(SAVE_DIR, f), cv2.IMREAD_GRAYSCALE)
            if img is not None: tpls[f] = img
    return tpls

templates = load_templates()
print(f"✅ Loaded {len(templates)} templates from '{SAVE_DIR}'")

# ===================== SYMBOL / GEOMETRY =====================
MATCH_THRESH = 0.55
COOLDOWN = 1.0
last_print = 0.0

def detect_and_crop_symbol(frame_rgb):
    H, W, _ = frame_rgb.shape
    roi_bottom = int(H * ROI_START)
    roi = frame_rgb[0:roi_bottom, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bin_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k = np.ones((5, 5), np.uint8)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None, bin_inv, None

    valid_contours =[]
    for c in contours:
        area = cv2.contourArea(c)
        if 1500 < area < (W * roi_bottom * 0.8):
            x, y, w, h = cv2.boundingRect(c)
            if 0.25 < (w / float(h)) < 4.0: valid_contours.append(c)

    if not valid_contours: return None, None, bin_inv, None

    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad = 12
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(W, x + w + pad), min(roi_bottom, y + h + pad)

    crop_gray = gray[ry0:ry1, rx0:rx1]
    return crop_gray, (rx0, ry0, rx1, ry1), bin_inv, c

def _order_pts(pts_approx):
    p = pts_approx.reshape(-1, 2).astype(np.float32)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]

def _side_lengths(p4):
    d =[]
    for i in range(4):
        x1, y1 = p4[i]
        x2, y2 = p4[(i + 1) % 4]
        d.append(float(np.hypot(x2 - x1, y2 - y1)))
    return d

def classify_geometry_from_contour(c):
    area = cv2.contourArea(c)
    if area < 1500: return None
    perim = cv2.arcLength(c, True)
    if perim < 1e-6: return None
    approx = cv2.approxPolyDP(c, 0.015 * perim, True)
    verts = len(approx)
    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)
    circ = 4 * np.pi * area / (perim * perim)

    if solidity < 0.80 and verts >= 8: return "STAR"
    if 7 <= verts <= 9 and solidity > 0.92 and circ < 0.85: return "OCTAGON"
    if 0.60 < circ < 0.85 and solidity < 0.90: return "PARTIAL_CIRCLE"
    if circ > 0.86 and solidity > 0.95 and verts >= 12: return "CIRCLE"
    if verts == 4 and solidity > 0.90:
        p = _order_pts(approx)
        s0, s1, s2, s3 = _side_lengths(p)
        rel_diff = lambda a, b: abs(a - b) / max(a, b, 1e-6)
        opp0, opp1 = rel_diff(s0, s2), rel_diff(s1, s3)
        if opp0 < 0.18 and opp1 < 0.18:
            _, _, w, h = cv2.boundingRect(approx)
            if 0.9 < (w / float(h)) < 1.1: return "DIAMOND"
            return "PARALLELOGRAM"
        if (opp0 < 0.18) ^ (opp1 < 0.18): return "TRAPEZOID"
        return "QUAD"
    return None

# ===================== MAIN LOOP =====================
set_motors(0, 0)
time.sleep(0.5)

crop_gray, symbol_box, bin_sym, symbol_contour = None, None, None, None
det_name, det_score, geo_label = None, 0.0, None

try:
    while True:
        frame = picam2.capture_array()
        h, w, _ = frame.shape
        disp = frame.copy() if SHOW_DEBUG else None
        now = time.time()

        # ---------------- 1) LINE FOLLOW (High Frequency) ----------------
        roi_line = frame[int(h * ROI_START):h, :]
        gray_line = cv2.cvtColor(roi_line, cv2.COLOR_RGB2GRAY)
        _, binary_line = cv2.threshold(gray_line, THRESH_LINE, 255, cv2.THRESH_BINARY_INV)
        binary_line = cv2.morphologyEx(binary_line, cv2.MORPH_OPEN, kernel5)

        contours_line, _ = cv2.findContours(binary_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        have_line = False

        if contours_line:
            c_line = max(contours_line, key=cv2.contourArea)
            if cv2.contourArea(c_line) > MIN_LINE_AREA:
                M = cv2.moments(c_line)
                if M["m00"] != 0:
                    have_line = True
                    cx = int(M["m10"] / M["m00"])
                    err = float(cx - (w // 2))

                    # DEADBAND: Ignore tiny wobbles
                    if abs(err) < DEADBAND:
                        err = 0.0

                    # Time delta
                    dt = max(1e-3, now - _last_t)
                    _last_t = now

                    # PID Math
                    _pid_i = max(-I_CLAMP, min(I_CLAMP, _pid_i + err * dt))
                   
                    # FILTERED DERIVATIVE (Anti-Jitter)
                    raw_derr = (err - _last_err) / dt
                    _smoothed_derr = (0.8 * _smoothed_derr) + (0.2 * raw_derr)
                    _smoothed_derr = max(-D_CLAMP, min(D_CLAMP, _smoothed_derr))
                   
                    steer = (KP_LINE * err) + (KI_LINE * _pid_i) + (KD_LINE * _smoothed_derr)
                    steer = max(-STEER_LIMIT, min(STEER_LIMIT, steer))
                    _last_err = err

                    # 3-TIER NON-LINEAR BRAKING LOGIC
                    abs_err = abs(err)
                    if abs_err < 50:
                        # TIER 1: Straightaway -> FULL SPEED
                        base = MAX_BASE_PWM
                       
                    elif abs_err < 160:
                        # TIER 2: Sweeping Curve -> Gradually roll off the throttle
                        brake_force = (abs_err - 50) * 0.25
                        base = MAX_BASE_PWM - brake_force
                       
                    else:
                        # TIER 3: 90-DEGREE TURN DETECTED! -> Drop forward momentum to ZERO
                        base = MIN_BASE_PWM

                    base = max(MIN_BASE_PWM, min(MAX_BASE_PWM, base))

                    # Apply to motors (Allows negative speeds for pivots)
                    left_pwm = base - steer
                    right_pwm = base + steer
                    set_motors(left_pwm, right_pwm)

                    if SHOW_DEBUG:
                        cv2.circle(disp, (cx, int(h * ROI_START) + roi_line.shape[0] // 2), 6, (0, 255, 0), -1)

        if not have_line:
            # --- MEMORY RECOVERY OVERSHOOT ---
            _pid_i = 0.0 # reset integral windup
           
            if _last_err < -20:
                # Line was lost to the left, pivot left
                set_motors(-RECOVERY_SPEED, RECOVERY_SPEED)
            elif _last_err > 20:
                # Line was lost to the right, pivot right
                set_motors(RECOVERY_SPEED, -RECOVERY_SPEED)
            else:
                set_motors(0, 0)

        # ---------------- 2) SYMBOL DETECTION (Low Frequency/Decoupled) ----------------
        if (now - _last_sym_time) > SYMBOL_INTERVAL:
            crop_gray, symbol_box, bin_sym, symbol_contour = detect_and_crop_symbol(frame)
            det_name, det_score, geo_label = None, 0.0, None

            if crop_gray is not None:
                processed_crop = preprocess_template(crop_gray, size=(120, 120))
                geo_label = classify_geometry_from_contour(symbol_contour)

                if templates:
                    for name, tpl in templates.items():
                        res = cv2.matchTemplate(processed_crop, tpl, cv2.TM_CCOEFF_NORMED)
                        score = res[0][0]
                        if score > MATCH_THRESH and score > det_score:
                            det_score = score
                            det_name = name
           
            _last_sym_time = now

        # ---------------- 3) DISPLAY & INPUT ----------------
        if SHOW_DEBUG:
            roi_y = int(h * ROI_START)
            cv2.line(disp, (0, roi_y), (w, roi_y), (255, 0, 0), 2)
           
            if symbol_box is not None:
                x0, y0, x1, y1 = symbol_box
                if det_name is not None:
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
                    cv2.putText(disp, f"{det_name} ({det_score:.2f})", (x0, max(20, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if now - last_print > COOLDOWN:
                        print(f"✅ {det_name} detected (score={det_score:.2f})")
                        last_print = now
                else:
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
               
                if geo_label:
                    cv2.putText(disp, f"Geo: {geo_label}", (x0, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Camera", disp)
            cv2.imshow("Binary (Line)", binary_line)
            if bin_sym is not None:
                cv2.imshow("Binary (Symbol)", bin_sym)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('t') and crop_gray is not None:
                tpl = preprocess_template(crop_gray, size=(120, 120))
                filename = os.path.join(SAVE_DIR, f"template_{tpl_idx:03d}.png")
                cv2.imwrite(filename, tpl)
                print(f"✅ Saved {filename}")
                tpl_idx += 1
                templates = load_templates()
            elif key == ord('q'):
                break

finally:
    coast_stop()
    pwmL.stop()
    pwmR.stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()
    picam2.stop()


