import cv2
import mediapipe as mp
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
from mediapipe.python.solutions.hands import HandLandmark as hl

WINDOW_NAME = "Hand Tracking"
LEAVING_KEY = "q"
TIME_PRINT = False
DURATION = 5
FLIP = True
MOVING_MOUSE = True
PRECISION = 0.04

hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

video = cv2.VideoCapture(0)

screenWidth, screenHeight = pyautogui.size()
rightClick = False
leftClick = False

print(f"Press '{LEAVING_KEY}' to exit")
t = time.time()
last_time = t

while video.isOpened():
    _, frame = video.read()
    if frame is None:
        print("End of video")
        break
    if FLIP:
        frame = cv2.flip(frame, 1)

    RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(RGB)

    try:
        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[len(results.multi_hand_landmarks) - 1]
            pointer = hand.landmark[hl.WRIST]
            thumb = hand.landmark[hl.THUMB_TIP]
            index = hand.landmark[hl.INDEX_FINGER_TIP]
            middle = hand.landmark[hl.MIDDLE_FINGER_TIP]
            ring = hand.landmark[hl.RING_FINGER_TIP]
            pinky = hand.landmark[hl.PINKY_TIP]

            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
            )

            if TIME_PRINT:
                t = time.time()
                if t >= last_time + DURATION:
                    last_time = t
                    print(pointer)

            if MOVING_MOUSE:
                x = int(((pointer.x / 0.9) - 0.1) * screenWidth)
                y = int(((pointer.y / 0.9) - 0.1) * screenHeight)
                pyautogui.moveTo(x, y)

                # Left click
                if (
                    thumb.x - PRECISION < index.x
                    and thumb.x + PRECISION > index.x
                    and thumb.y - PRECISION < index.y
                    and thumb.y + PRECISION > index.y
                ):
                    if not leftClick:
                        leftClick = True
                        pyautogui.mouseDown()
                elif leftClick:
                    leftClick = False
                    pyautogui.mouseUp()

                # Right click
                if (
                    thumb.x - PRECISION < middle.x
                    and thumb.x + PRECISION > middle.x
                    and thumb.y - PRECISION < middle.y
                    and thumb.y + PRECISION > middle.y
                ):
                    if not rightClick:
                        rightClick = True
                        pyautogui.mouseDown(button="right")
                elif rightClick:
                    rightClick = False
                    pyautogui.mouseUp(button="right")

                # Exit
                if (
                    thumb.x - PRECISION < pinky.x
                    and thumb.x + PRECISION > pinky.x
                    and thumb.y - PRECISION < pinky.y
                    and thumb.y + PRECISION > pinky.y
                ):
                    break

        cv2.imshow(WINDOW_NAME, frame)

    except Exception as e:
        print("Error :", e)
        break

    try:
        if (
            cv2.waitKey(1) == ord(LEAVING_KEY)
            or cv2.getWindowProperty(WINDOW_NAME, 0) == -1
        ):
            print("Exit")
            break
    except cv2.error as e:
        break

video.release()
cv2.destroyAllWindows()
