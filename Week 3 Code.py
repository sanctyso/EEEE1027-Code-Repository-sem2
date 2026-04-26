import time
import os
import cv2
import numpy as np
import threading
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ===================== GPIO SETUP =====================
GPIO.setmode(GPIO.BOARD)

ENA, IN1, IN2 = 13, 12, 11
ENB, IN3, IN4 = 18, 16, 15
GPIO.setup([ENA, ENB, IN1, IN2, IN3, IN4], GPIO.OUT)

# 100Hz frequency keeps torque high at lower speeds
pwmA = GPIO.PWM(ENA, 100)
pwmB = GPIO.PWM(ENB, 100)
pwmA.start(0)
pwmB.start(0)

# ===================== SETTINGS & PARAMETERS =====================
# --- Line Follow Settings ---
THRESHOLD = 100
FORWARD_SPEED = 21
MAX_TURN_SPEED = 35
MIN_TURN_SPEED = 35
CENTER_TOLERANCE = 70

# --- Symbol Detection Settings ---
ROI_START = 0.75
MATCH_THRESH = 0.7
COOLDOWN = 1.2
SAVE_DIR = "templates"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- MANEUVER SETTINGS ---
STOP_TIME = 3.0
TURN_180_TIME = 2.2
TURN_90_TIME = 1
SYMBOL_COOLDOWN = 0.9
RESTART_TURN_SPEED = 50   # separate speed for RESTART / 360 turn

# ===================== FULL RED / YELLOW SYSTEM =====================
LOWER_BLACK = np.array([0, 0, 0])
UPPER_BLACK = np.array([180, 255, 88])

LOWER_RED = np.array([100, 80, 80])
UPPER_RED = np.array([140, 255, 255])

LOWER_YELLOW = np.array([59, 153, 0])
UPPER_YELLOW = np.array([108, 255, 255])

# ===================== SHARED STATE =====================
class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.ui_mode = False

        self.frame = None
        self.line_mode = "NORMAL"   # RED, YELLOW, BOTH, NORMAL

        # Line Follow State
        self.motor_action = "stop"
        self.motor_speed = 0
        self.binary_line = None
        self.active_tracking = "BLACK"

        # Symbol State
        self.current_crop = None
        self.symbol_box = None
        self.bin_sym = None
        self.geo_label = None
        self.best_name = None
        self.best_score = 0.0

        # Maneuver State
        self.active_maneuver = None
        self.cooldown_until = 0.0

state = SharedState()

# ===================== MOTOR FUNCTIONS =====================
def set_motors(action, speed):
    if action == "forward":
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

    elif action == "left":
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW);  GPIO.output(IN4, GPIO.HIGH)
        pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

    elif action == "right":
        GPIO.output(IN1, GPIO.LOW);  GPIO.output(IN2, GPIO.HIGH)
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
        pwmA.ChangeDutyCycle(speed); pwmB.ChangeDutyCycle(speed)

    else:  # stop
        GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
        pwmA.ChangeDutyCycle(0); pwmB.ChangeDutyCycle(0)

# ===================== COLOR MASK SYSTEM =====================
def get_masks(frame, mode):
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    mask_black = cv2.inRange(hsv, LOWER_BLACK, UPPER_BLACK)

    mask_priority = None
    if mode == "RED":
        mask_priority = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
    elif mode == "YELLOW":
        mask_priority = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
    elif mode == "BOTH":
        mask_red = cv2.inRange(hsv, LOWER_RED, UPPER_RED)
        mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)
        mask_priority = cv2.bitwise_or(mask_red, mask_yellow)

    return mask_priority, mask_black

def get_combined_mask(frame, mode):
    p_mask, b_mask = get_masks(frame, mode)
    combined = cv2.bitwise_or(p_mask, b_mask) if p_mask is not None else b_mask
    return p_mask, b_mask, combined

# ===================== SYMBOL & GEOMETRY FUNCTIONS =====================
def preprocess_template(crop_gray, size=(120, 120)):
    crop_eq = cv2.equalizeHist(crop_gray)
    h, w = crop_eq.shape
    diff = abs(h - w)

    top = bottom = left = right = 0
    if h > w:
        left = diff // 2
        right = diff - left
    elif w > h:
        top = diff // 2
        bottom = diff - top

    crop_sq = cv2.copyMakeBorder(crop_eq, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    crop_pad = cv2.copyMakeBorder(crop_sq, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(crop_pad, size, interpolation=cv2.INTER_AREA)

def load_templates():
    tpls = {}
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.lower().endswith(".png"):
            img = cv2.imread(os.path.join(SAVE_DIR, f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                tpls[f] = img
    return tpls

templates = load_templates()
print(f"✅ Loaded {len(templates)} templates from '{SAVE_DIR}'")

def detect_and_crop_symbol(frame_rgb):
    H, W, _ = frame_rgb.shape
    roi_bottom = int(H * ROI_START)
    roi = frame_rgb[0:roi_bottom, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    bin_inv = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        81, 8
    )
    k = np.ones((5, 5), np.uint8)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k)
    bin_inv = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(bin_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, bin_inv, None

    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 1000 < area < (W * roi_bottom * 0.6):
            x, y, w, h = cv2.boundingRect(c)
            margin = 5
            if x > margin and y > margin and (x + w) < (W - margin) and (y + h) < (roi_bottom - margin):
                if 0.25 < (w / float(h)) < 4.0:
                    hull = cv2.convexHull(c)
                    if cv2.contourArea(hull) > 0 and (area / cv2.contourArea(hull)) > 0.35:
                        valid_contours.append(c)

    if not valid_contours:
        return None, None, bin_inv, None

    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    pad = 12
    rx0, ry0 = max(0, x - pad), max(0, y - pad)
    rx1, ry1 = min(W, x + w + pad), min(roi_bottom, y + h + pad)

    return gray[ry0:ry1, rx0:rx1], (rx0, ry0, rx1, ry1), bin_inv, c

def _order_pts(pts_approx):
    p = pts_approx.reshape(-1, 2).astype(np.float32)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    return p[np.argsort(ang)]

def _side_lengths(p4):
    d = []
    for i in range(4):
        x1, y1 = p4[i]
        x2, y2 = p4[(i + 1) % 4]
        d.append(float(np.hypot(x2 - x1, y2 - y1)))
    return d

def classify_geometry_from_contour(c):
    area = cv2.contourArea(c)
    if area < 1500:
        return None
    perim = cv2.arcLength(c, True)
    if perim < 1e-6:
        return None
    approx = cv2.approxPolyDP(c, 0.015 * perim, True)
    verts = len(approx)
    hull = cv2.convexHull(c)
    solidity = area / (cv2.contourArea(hull) + 1e-6)
    circ = 4 * np.pi * area / (perim * perim)

    if solidity < 0.80 and verts >= 8:
        return "STAR"
    if 7 <= verts <= 9 and solidity > 0.92 and circ < 0.85:
        return "OCTAGON"
    if 0.60 < circ < 0.85 and solidity < 0.90:
        return "PARTIAL_CIRCLE"
    if circ > 0.86 and solidity > 0.95 and verts >= 12:
        return "CIRCLE"

    if verts == 4 and solidity > 0.90:
        p = _order_pts(approx)
        s0, s1, s2, s3 = _side_lengths(p)
        opp0 = abs(s0 - s2) / max(s0, s2, 1e-6)
        opp1 = abs(s1 - s3) / max(s1, s3, 1e-6)

        if opp0 < 0.18 and opp1 < 0.18:
            _, _, w, h = cv2.boundingRect(approx)
            return "DIAMOND" if 0.9 < (w / float(h)) < 1.1 else "PARALLELOGRAM"

        if (opp0 < 0.18) ^ (opp1 < 0.18):
            return "TRAPEZOID"

        return "QUAD"

    return None

# ===================== THREAD 1: LINE FOLLOW =====================
def thread_line_follow():
    last_turn = "left"

    while state.running:
        with state.lock:
            frame = state.frame.copy() if state.frame is not None else None
            ui_mode = state.ui_mode
            current_mode = state.line_mode

        if frame is None or ui_mode:
            time.sleep(0.01)
            continue

        h, w, _ = frame.shape
        roi_y = h // 2

        p_mask, b_mask, combined = get_combined_mask(frame, current_mode)

        target_roi = None
        active_tracking = "BLACK"

        if current_mode in ["RED", "YELLOW", "BOTH"]:
            if p_mask is not None:
                roi_p = p_mask[roi_y:, :]
                if cv2.moments(roi_p)["m00"] > 800:
                    target_roi = roi_p
                    active_tracking = current_mode if current_mode != "BOTH" else "RED/YELLOW"

            if target_roi is None:
                roi_b = b_mask[roi_y:, :]
                if cv2.moments(roi_b)["m00"] > 500:
                    target_roi = roi_b
                    active_tracking = "BLACK"

        else: # NORMAL mode
            roi_b = b_mask[roi_y:, :]
            if cv2.moments(roi_b)["m00"] > 500:
                target_roi = roi_b
                active_tracking = "BLACK"

        action = "stop"
        speed = 0

        if target_roi is not None:
            M = cv2.moments(target_roi)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                deviation = cx - w // 2

                if abs(deviation) < CENTER_TOLERANCE:
                    action = "forward"
                    speed = FORWARD_SPEED
                elif deviation < 0:
                    action = "left"
                    speed = MAX_TURN_SPEED
                    last_turn = "left"
                else:
                    action = "right"
                    speed = MAX_TURN_SPEED
                    last_turn = "right"
        else:
            action = last_turn
            speed = MAX_TURN_SPEED

        with state.lock:
            state.motor_action = action
            state.motor_speed = speed
            state.binary_line = combined
            state.active_tracking = active_tracking

        time.sleep(0.01)

# ===================== THREAD 2: SYMBOL DETECTOR =====================
def thread_symbol_detect():
    global templates

    while state.running:
        with state.lock:
            frame = state.frame.copy() if state.frame is not None else None
            ui_mode = state.ui_mode

        if frame is None or ui_mode:
            time.sleep(0.05)
            continue

        crop_gray, box, bin_sym, cnt = detect_and_crop_symbol(frame)

        name = None
        score = 0.0
        geo = None

        if crop_gray is not None:
            geo = classify_geometry_from_contour(cnt)
            processed = preprocess_template(crop_gray, size=(120, 120))

            if templates:
                for t_name, t_img in templates.items():
                    res = cv2.matchTemplate(processed, t_img, cv2.TM_CCOEFF_NORMED)
                    s = res[0][0]
                    if s > MATCH_THRESH and s > score:
                        score = s
                        name = t_name

        with state.lock:
            state.current_crop = crop_gray
            state.symbol_box = box
            state.bin_sym = bin_sym
            state.geo_label = geo
            state.best_name = name
            state.best_score = score

            if name is not None and time.time() > state.cooldown_until and state.active_maneuver is None:
                base_name = name.split('.')[0]
                if '_' in base_name and base_name.rsplit('_', 1)[1].isdigit():
                    base_name = base_name.rsplit('_', 1)[0]

                cmd = base_name.upper()
                if cmd in ["STOP", "BUTTON", "RESTART", "LEFT", "RIGHT"]:
                    state.active_maneuver = cmd
                    state.cooldown_until = time.time() + 999

        time.sleep(0.01)

# ===================== THREAD 3: MOTOR CONTROLLER =====================
def thread_motor_ctrl():
    while state.running:
        with state.lock:
            action = state.motor_action
            speed = state.motor_speed
            ui_mode = state.ui_mode
            maneuver = state.active_maneuver

        if ui_mode:
            set_motors("stop", 0)

        elif maneuver is not None:
            print(f"\n⚠️ EXECUTING MANEUVER: {maneuver} ⚠️")

            if maneuver in ["STOP", "BUTTON"]:
                set_motors("stop", 0)
                time.sleep(STOP_TIME)

            elif maneuver == "RESTART":
                set_motors("right", RESTART_TURN_SPEED)
                time.sleep(TURN_180_TIME)

            elif maneuver == "LEFT":
                set_motors("left", MAX_TURN_SPEED)
                time.sleep(TURN_90_TIME)

            elif maneuver == "RIGHT":
                set_motors("right", MAX_TURN_SPEED)
                time.sleep(TURN_90_TIME)

            with state.lock:
                state.active_maneuver = None
                state.cooldown_until = time.time() + SYMBOL_COOLDOWN
                print("✅ Maneuver Complete. Resuming Line Follow.\n")

        else:
            set_motors(action, speed)

        time.sleep(0.05)

# ===================== MENU HELPER =====================
def run_startup_menu(camera):
    print("\n--- STARTUP MENU ---")
    print("Please select mode on the camera window.")

    while True:
        frame = camera.capture_array()
        disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w, _ = disp.shape

        cv2.rectangle(disp, (30, 30), (w - 30, h - 30), (0, 0, 0), -1)
        cv2.rectangle(disp, (30, 30), (w - 30, h - 30), (255, 255, 255), 2)

        cv2.putText(disp, "SELECT LINE MODE:", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(disp, "[R] RED Priority", (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(disp, "[Y] YELLOW Priority", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(disp, "[B] BOTH Priority", (70, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        cv2.putText(disp, "[N] NORMAL Mode", (70, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Camera", disp)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('r') or k == ord('R'):
            return "RED"
        elif k == ord('y') or k == ord('Y'):
            return "YELLOW"
        elif k == ord('b') or k == ord('B'):
            return "BOTH"
        elif k == ord('n') or k == ord('N'):
            return "NORMAL"
        elif k == ord('q'):
            return "QUIT"

# ===================== MAIN LOOP =====================
if __name__ == "__main__":
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    selected_mode = run_startup_menu(picam2)
    if selected_mode == "QUIT":
        picam2.stop()
        exit(0)

    with state.lock:
        state.line_mode = selected_mode
    print(f"✅ Selected Mode: {selected_mode} PRIORITY")

    threads = [
        threading.Thread(target=thread_line_follow, daemon=True),
        threading.Thread(target=thread_symbol_detect, daemon=True),
        threading.Thread(target=thread_motor_ctrl, daemon=True)
    ]
    for t in threads:
        t.start()

    print("✅ System Running!")
    print("Keys: t = save symbol as template, q = quit")
    last_print = time.time()

    try:
        while state.running:
            frame = picam2.capture_array()

            with state.lock:
                state.frame = frame
                ui_mode = state.ui_mode
                box = state.symbol_box
                name = state.best_name
                score = state.best_score
                geo = state.geo_label
                action = state.motor_action
                maneuver = state.active_maneuver
                bin_line = state.binary_line
                bin_sym = state.bin_sym
                curr_speed = state.motor_speed
                current_mode = state.line_mode
                active_tracking = state.active_tracking

            if ui_mode:
                continue

            disp = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w, _ = disp.shape

            roi_y = int(h * ROI_START)
            cv2.line(disp, (0, roi_y), (w, roi_y), (255, 0, 0), 2)

            status_text = f"MANEUVER: {maneuver}" if maneuver else f"State: {action.upper()} ({curr_speed}%)"
            cv2.putText(disp, status_text, (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if maneuver else (0, 255, 0), 2)
            cv2.putText(disp, f"MODE: {current_mode}", (10, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
            cv2.putText(disp, f"TRACKING: {active_tracking}", (10, h - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

            cv2.putText(disp, "Top Symbol Searching Area", (5, roi_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if box is not None:
                x0, y0, x1, y1 = box
                if name is not None:
                    disp_name = name.split('.')[0]
                    if '_' in disp_name and disp_name.rsplit('_', 1)[1].isdigit():
                        disp_name = disp_name.rsplit('_', 1)[0]

                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 3)
                    cv2.putText(disp, f"{disp_name} ({score:.2f})", (x0, max(20, y0 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if time.time() - last_print > COOLDOWN:
                        print(f"✅ {disp_name} detected (score={score:.2f})")
                        last_print = time.time()
                else:
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
                    cv2.putText(disp, "Unknown (press 't')", (x0, max(20, y0 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if geo:
                    cv2.putText(disp, f"Geo: {geo}", (x0, y1 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Camera", disp)
            if bin_line is not None:
                cv2.imshow("Combined Mask (Line+Color)", bin_line)
            if bin_sym is not None:
                cv2.imshow("Binary (Symbol)", bin_sym)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            elif key == ord('t'):
                with state.lock:
                    crop = state.current_crop

                if crop is None:
                    print("❌ No clear symbol detected to save.")
                else:
                    with state.lock:
                        state.ui_mode = True
                    time.sleep(0.1)

                    name_input = ""
                    while True:
                        prompt_disp = disp.copy()
                        cv2.rectangle(prompt_disp, (20, h // 2 - 60), (w - 20, h // 2 + 40), (0, 0, 0), -1)
                        cv2.rectangle(prompt_disp, (20, h // 2 - 60), (w - 20, h // 2 + 40), (255, 255, 255), 2)

                        cv2.putText(prompt_disp, "TYPE SYMBOL NAME (e.g. BUTTON, LEFT):",
                                    (30, h // 2 - 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(prompt_disp, f"> {name_input}_",
                                    (30, h // 2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                        cv2.putText(prompt_disp, "Press ENTER to save, ESC to cancel.",
                                    (30, h // 2 + 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        cv2.imshow("Camera", prompt_disp)

                        k = cv2.waitKey(0) & 0xFF
                        if k == 13 or k == 10:
                            break
                        elif k == 27:
                            name_input = ""
                            break
                        elif k == 8 or k == 127:
                            name_input = name_input[:-1]
                        else:
                            try:
                                if chr(k).isalnum() or k in [ord('-'), ord('_')]:
                                    name_input += chr(k).upper()
                            except:
                                pass

                    if name_input.strip() != "":
                        idx = 1
                        while os.path.exists(os.path.join(SAVE_DIR, f"{name_input}_{idx}.png")):
                            idx += 1

                        tpl = preprocess_template(crop, size=(120, 120))
                        filename = os.path.join(SAVE_DIR, f"{name_input}_{idx}.png")
                        cv2.imwrite(filename, tpl)
                        print(f"✅ Saved on-screen as {filename}")
                        templates = load_templates()

                    with state.lock:
                        state.ui_mode = False

    except KeyboardInterrupt:
        pass

    finally:
        with state.lock:
            state.running = False
        for t in threads:
            t.join()
        set_motors("stop", 0)
        pwmA.stop()
        pwmB.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        picam2.stop()
