import cv2
import numpy as np
import mss
import pyautogui
import time

template = cv2.imread('dino1.png', 0)
t_h, t_w = template.shape
template1 = cv2.imread('dino2.png', 0)

JUMP_THRESHOLD = 70
SCAN_WIDTH = 350
BASE_WIDTH = 22
SPEED_COEFF = 0.5

pyautogui.PAUSE = 0

cv2.namedWindow("Eye_View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eye_View", 600, 150)
cv2.namedWindow("FRONT_DEBUG", cv2.WINDOW_NORMAL)
cv2.resizeWindow("FRONT_DEBUG", 600, 150)

with mss.mss() as sct:
    print("Поиск динозавра")
    found = False

    while not found:
        img = np.array(sct.grab(sct.monitors[1]))
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            dino_x, dino_y = max_loc
            found = True
            print("Динозавр найден")
        else:
            time.sleep(0.5)

    time.sleep(1)
    pyautogui.press('space')

    scan_height = t_h - 15
    start_time = None
    jumping = False
    dynamic_threshold = JUMP_THRESHOLD

    while True:

        if not jumping:
            monitor = {
                "top": dino_y, 
                "left": dino_x + t_w + 12, 
                "width": SCAN_WIDTH, 
                "height": scan_height
            }
            shot = np.array(sct.grab(monitor))
            shot_gray = cv2.cvtColor(shot, cv2.COLOR_BGRA2GRAY)
            _, thresh = cv2.threshold(shot_gray, 200, 255, cv2.THRESH_BINARY_INV)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
            closest_x = None
            closest_width = 0

            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                w = stats[i, cv2.CC_STAT_WIDTH]
                area = stats[i, cv2.CC_STAT_AREA]
                if area < 5:
                    continue
                if closest_x is None or x < closest_x:
                    closest_x = x
                    closest_width = w

            elapsed = (time.time() - start_time) if start_time else 0
            speed_bonus = int(elapsed * SPEED_COEFF)
            extra = max(0, (closest_width - BASE_WIDTH) // 2)
            dynamic_threshold = JUMP_THRESHOLD + extra + speed_bonus

            if closest_x is not None and closest_x < dynamic_threshold:
                if start_time is None:
                    start_time = time.time()
                pyautogui.press('space')
                jumping = True
                print(f"JUMP dist={closest_x} width={closest_width} threshold={dynamic_threshold}")

            debug = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.line(debug, (dynamic_threshold, 0), (dynamic_threshold, scan_height), (0, 0, 255), 1)
            elapsed_show = (time.time() - start_time) if start_time else 0
            cv2.putText(debug, f"t={int(elapsed_show)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Eye_View", cv2.resize(debug, (600, 150)))

        elif jumping == True:
            front_monitor = {
                "top": dino_y, 
                "left": dino_x + t_w + 5, 
                "width": int(dynamic_threshold) + 8, 
                "height": scan_height
            }
            front = np.array(sct.grab(front_monitor))
            u_gray = cv2.cvtColor(front, cv2.COLOR_BGRA2GRAY)
            _, u_thresh = cv2.threshold(u_gray, 200, 255, cv2.THRESH_BINARY_INV)
            danger = np.any(u_thresh > 0)

            front_debug = cv2.cvtColor(u_thresh, cv2.COLOR_GRAY2BGR)
            front_debug = cv2.resize(front_debug, (600, 150))
            cv2.putText(front_debug, f"Dan={danger}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("FRONT_DEBUG", front_debug)

            if not danger:
                pyautogui.keyDown('down')
                jumping = 'landing'

        elif jumping == 'landing':
            dino_monitor = {
                "top": dino_y, 
                "left": dino_x, 
                "width": t_w, 
                "height": t_h
            }
            dino_shot = np.array(sct.grab(dino_monitor))
            dino_gray = cv2.cvtColor(dino_shot, cv2.COLOR_BGRA2GRAY)

            _, match_run, _, _ = cv2.minMaxLoc(cv2.matchTemplate(dino_gray, template, cv2.TM_CCOEFF_NORMED))
            _, match_crouch, _, _ = cv2.minMaxLoc(cv2.matchTemplate(dino_gray, template1, cv2.TM_CCOEFF_NORMED))

            on_ground = match_run > 0.7 or match_crouch > 0.7

            dino_debug = cv2.cvtColor(dino_gray, cv2.COLOR_GRAY2BGR)
            dino_debug = cv2.resize(dino_debug, (600, 150))
            cv2.putText(dino_debug, f"run={match_run:.2f} cr={match_crouch:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("FRONT_DEBUG", dino_debug)

            if on_ground:
                pyautogui.keyUp('down')
                jumping = False
                print(f"On ground run={match_run:.2f} crouch={match_crouch:.2f}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()