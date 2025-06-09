import cv2
import mediapipe as mp
import numpy as np
import time
import math
# Removed screen_brightness_control as sbc
from pynput.mouse import Controller, Button
from pynput.keyboard import Key, Controller as KeyboardController
import platform # To detect the operating system
import subprocess # For running system commands on macOS/Linux volume/mute
import re # For regex to parse amixer output on Linux (only used if Linux volume fetching is enabled)

# --- Global Configuration and Setup ---
# MediaPipe Hand Model Initialization
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # IMPORTANT: Now supports up to 2 hands for two-hand gestures
    min_detection_confidence=0.7, # Increased for better stability
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Volume Control Libraries (Platform Specific)
volume_obj = None
min_vol_db, max_vol_db = -65.25, 0.0 # Default range for Windows, usually in dB

if platform.system() == "Windows":
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        from ctypes import cast, POINTER

        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume_obj = cast(interface, POINTER(IAudioEndpointVolume))
        min_vol_db, max_vol_db, _ = volume_obj.GetVolumeRange()
        print(f"Windows Audio Volume Range: Min={min_vol_db:.2f}dB, Max={max_vol_db:.2f}dB")
    except Exception as e:
        print(f"Error initializing Windows audio control: {e}. Volume control might not work.")
elif platform.system() == "Darwin": # macOS
    print(f"Running on macOS. Volume control will use 'osascript'.")
elif platform.system() == "Linux":
    print(f"Running on Linux. Volume control will use 'amixer' or 'pactl'.")
else:
    print("Unsupported operating system for direct volume control.")

# PyNput Mouse and Keyboard Controller
mouse = Controller()
keyboard = KeyboardController()

# --- HARDCODED SCREEN RESOLUTION ---
# IMPORTANT: REPLACE THESE WITH YOUR ACTUAL SCREEN RESOLUTION!
# This is crucial for accurate mouse control.
# Example: For a 1920x1080 display:
screen_w, screen_h = 1920, 1080 # <<<--- SET YOUR SCREEN WIDTH AND HEIGHT HERE!

# --- Global State Variables ---
class Mode:
    NONE = 0
    VOLUME_CONTROL = 1 # Now requires two hands
    MOUSE_CONTROL = 2 # Mode index changed

current_mode = Mode.NONE
last_mode_change_time = time.time()
mode_change_cooldown = 1.0 # Cooldown to prevent rapid mode switching

# Volume Control State
current_volume_percent = 50 # Default value
try:
    if platform.system() == "Windows" and volume_obj:
        current_volume_percent = int(get_current_system_volume())
except Exception as e:
    print(f"Could not get initial system volume: {e}. Defaulting to {current_volume_percent}%.")

last_volume_update_time = time.time()
volume_update_interval = 0.05 # Prevent rapid volume changes (0.05 seconds)

# Removed Brightness Control State and related variables

# Mouse Control State
mouse_smooth_factor = 7 # Higher means smoother but slower mouse movement
prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2 # Initialize to center of screen
mouse_click_debounce_time = 0.3 # Time required between clicks
last_click_time = time.time()

# Gesture Debounce for discrete actions (mute, task view)
gesture_debounce_time = 0.8 # Longer debounce for actions that shouldn't be repeated quickly
last_mute_toggle_time = time.time()
last_task_view_time = time.time()


# --- Helper Functions ---

def euclidean_distance(p1, p2):
    """Calculates the Euclidean distance between two 2D points."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_landmark_coords(landmark, w, h):
    """Converts normalized landmark coordinates to pixel coordinates."""
    return int(landmark.x * w), int(landmark.y * h)

def fingers_up(hand_landmarks):
    """
    Detects which fingers are extended based on landmark positions for a single hand.
    Returns a list of booleans [thumb, index, middle, ring, pinky].
    """
    fingers = []
    # Thumb (special case: check x-direction relative to thumb's IP joint for extension)
    # This logic assumes the hand is generally upright and facing the camera.
    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x: # Likely Right hand
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)
    else: # Likely Left hand
        fingers.append(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x)

    # Other 4 fingers (check y-direction of tip relative to their PIP joint - finger knuckle)
    for tip_idx in [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]:
        pip_idx = tip_idx - 2 # PIP joint is 2 landmarks below the tip (e.g., INDEX_FINGER_PIP is 6, INDEX_FINGER_TIP is 8)
        fingers.append(hand_landmarks.landmark[tip_idx].y < hand_landmarks.landmark[pip_idx].y)
    return fingers

# --- System Control Functions ---

def set_system_volume(vol_percent):
    """Sets the system volume to a given percentage (0-100)."""
    global current_volume_percent, last_volume_update_time
    # Update internal tracking immediately for display purposes
    current_volume_percent = vol_percent

    current_time = time.time()
    if current_time - last_volume_update_time < volume_update_interval:
        return # Debounce rapid updates

    if platform.system() == "Windows" and volume_obj:
        target_vol_db = np.interp(vol_percent, [0, 100], [min_vol_db, max_vol_db])
        try:
            volume_obj.SetMasterVolumeLevel(target_vol_db, None)
        except Exception as e:
            print(f"Error setting Windows volume: {e}")
    elif platform.system() == "Darwin": # macOS
        try:
            # osascript volume is 0-100, directly use vol_percent
            subprocess.run(["osascript", "-e", f"set volume output volume {vol_percent}"], check=True, capture_output=True)
        except Exception as e:
            print(f"Error setting macOS volume: {e}")
    elif platform.system() == "Linux": # Linux (using amixer as a common fallback)
        try:
            # amixer expects percentage directly. -D pulse specifies PulseAudio control.
            subprocess.run(["amixer", "-D", "pulse", "set", "Master", f"{vol_percent}%"], check=True, capture_output=True)
        except FileNotFoundError:
            print("Error: 'amixer' command not found. Please ensure 'pulseaudio-utils' or 'alsa-utils' is installed.")
        except Exception as e:
            print(f"Error setting Linux volume: {e}")
    last_volume_update_time = current_time # Update timestamp only if action was performed

def get_current_system_volume():
    """Gets the current system volume as a percentage (0-100)."""
    if platform.system() == "Windows" and volume_obj:
        current_vol_db = volume_obj.GetMasterVolumeLevel()
        return np.interp(current_vol_db, [min_vol_db, max_vol_db], [0, 100])
    elif platform.system() == "Darwin": # macOS
        try:
            result = subprocess.run(["osascript", "-e", "output volume of (get volume settings)"], capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
        except Exception:
            return current_volume_percent # Fallback if error
    elif platform.system() == "Linux": # Linux (using amixer)
        try:
            result = subprocess.run(["amixer", "-D", "pulse", "get", "Master"], capture_output=True, text=True, check=True)
            percent_match = re.search(r"\[(\d+)%\]", result.stdout)
            return int(percent_match.group(1)) if percent_match else current_volume_percent
        except Exception:
            return current_volume_percent # Fallback if error
    return current_volume_percent # Fallback if no specific OS method or issue

def toggle_mute():
    """Toggles the system mute state."""
    global last_mute_toggle_time
    current_time = time.time()
    if current_time - last_mute_toggle_time < gesture_debounce_time:
        return

    if platform.system() == "Windows" and volume_obj:
        try:
            current_mute_state = volume_obj.GetMute()
            volume_obj.SetMute(not current_mute_state, None)
            print(f"Mute toggled to: {not current_mute_state}")
        except Exception as e:
            print(f"Error toggling Windows mute: {e}")
    elif platform.system() == "Darwin": # macOS
        try:
            # Check current mute status
            status = subprocess.run(["osascript", "-e", "output muted of (get volume settings)"], capture_output=True, text=True, check=True).stdout.strip()
            if status == "true":
                subprocess.run(["osascript", "-e", "set volume output muted false"], check=True, capture_output=True)
                print("Mute toggled to: Off")
            else:
                subprocess.run(["osascript", "-e", "set volume output muted true"], check=True, capture_output=True)
                print("Mute toggled to: On")
        except Exception as e:
            print(f"Error toggling macOS mute: {e}")
    elif platform.system() == "Linux":
        try:
            # Using pactl for PulseAudio, which is common. Fallback to amixer if pactl isn't found.
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=True, capture_output=True)
            print("Mute toggled (Linux - PulseAudio)")
        except FileNotFoundError:
            print("Pactl not found, trying amixer...")
            try:
                subprocess.run(["amixer", "-D", "pulse", "set", "Master", "toggle"], check=True, capture_output=True)
                print("Mute toggled (Linux - ALSA via amixer)")
            except Exception as e:
                print(f"Error toggling Linux mute with amixer: {e}")
        except Exception as e:
            print(f"Error toggling Linux mute: {e}")
    last_mute_toggle_time = current_time

# Removed set_system_brightness function

def activate_task_view():
    """Activates the task view/expose (Windows Key + Tab or Ctrl + Up Arrow on macOS)."""
    global last_task_view_time
    current_time = time.time()
    if current_time - last_task_view_time < gesture_debounce_time:
        return

    if platform.system() == "Windows":
        keyboard.press(Key.cmd)
        keyboard.press(Key.tab)
        keyboard.release(Key.tab)
        keyboard.release(Key.cmd)
        print("Activated Windows Task View")
    elif platform.system() == "Darwin": # macOS Mission Control
        keyboard.press(Key.ctrl)
        keyboard.press(Key.up)
        keyboard.release(Key.up)
        keyboard.release(Key.ctrl)
        print("Activated macOS Mission Control")
    elif platform.system() == "Linux": # Gnome Shell Overview (Super Key)
        # Often Super key (Windows key on keyboard) opens overview
        keyboard.press(Key.super)
        keyboard.release(Key.super)
        print("Activated Linux Desktop Overview (Super Key)")
    last_task_view_time = current_time


def main():
    global current_mode, last_mode_change_time, prev_mouse_x, prev_mouse_y, last_click_time, current_volume_percent

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Virtual Desktop Assistant ---")
    print("Available Modes:")
    print("  - Hand Poses to ENTER Modes:")
    print("    - Two Hands, Index Fingers Up: Volume Control (Index-to-Index Distance)")
    print("    - One Hand, Only Index Finger Up: Mouse Control (Pinch with Thumb for click)")
    print("  - Discrete Gestures (Resets to IDLE after action):")
    print("    - Closed Fist: Toggle Mute/Unmute")
    print("    - Open Palm (All Fingers Up): Activate Task View / Mission Control")
    print("\nTo exit a mode, simply hide your hand(s) from the camera for a moment.")
    print("Press 'q' to quit the application.")

    # Initialize mouse position to center of the hardcoded screen
    prev_mouse_x, prev_mouse_y = screen_w // 2, screen_h // 2

    while True:
        current_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame = cv2.flip(frame, 1) # Flip for natural view
        h, w, _ = frame.shape # Get frame dimensions

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        hand_detected_in_frame = False
        num_hands_detected = 0
        if results.multi_hand_landmarks:
            num_hands_detected = len(results.multi_hand_landmarks)
            hand_detected_in_frame = True

            # Dictionaries to store hands based on which side they are (left/right)
            # This helps in reliably identifying two distinct hands for gestures
            hands_data = {} # Stores {'Left': hand_landmarks, 'Right': hand_landmarks}

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # MediaPipe gives us the hand classification (left/right)
                hand_label = results.multi_handedness[idx].classification[0].label # 'Left' or 'Right'
                hands_data[hand_label] = hand_landmarks

                # Draw landmarks for all detected hands
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        # --- Mode Switching Logic ---
        # Check for mode change gestures only if cooldown is over
        if current_time - last_mode_change_time > mode_change_cooldown:
            # Two-Hand Volume Control (New Gesture)
            if num_hands_detected == 2 and 'Left' in hands_data and 'Right' in hands_data:
                left_hand_fingers = fingers_up(hands_data['Left'])
                right_hand_fingers = fingers_up(hands_data['Right'])

                # Check if both index fingers are up
                if left_hand_fingers[1] and right_hand_fingers[1]: # Index finger is at index 1
                    if current_mode != Mode.VOLUME_CONTROL:
                        current_mode = Mode.VOLUME_CONTROL
                        print("Mode: VOLUME CONTROL (Two Hands, Index to Index)")
                        last_mode_change_time = current_time
            # One-Hand Mouse Control (Same Gesture)
            elif num_hands_detected == 1:
                # Find the single hand detected
                single_hand_landmarks = None
                if 'Left' in hands_data:
                    single_hand_landmarks = hands_data['Left']
                elif 'Right' in hands_data:
                    single_hand_landmarks = hands_data['Right']

                if single_hand_landmarks:
                    fingers = fingers_up(single_hand_landmarks)
                    if fingers == [False, True, False, False, False]: # Only Index finger up
                        if current_mode != Mode.MOUSE_CONTROL:
                            current_mode = Mode.MOUSE_CONTROL
                            print("Mode: MOUSE CONTROL (One Hand, Index Finger)")
                            last_mode_change_time = current_time
            # Discrete actions (only if in IDLE mode)
            elif num_hands_detected == 1 and current_mode == Mode.NONE:
                # Find the single hand detected
                single_hand_landmarks = None
                if 'Left' in hands_data:
                    single_hand_landmarks = hands_data['Left']
                elif 'Right' in hands_data:
                    single_hand_landmarks = hands_data['Right']

                if single_hand_landmarks:
                    fingers = fingers_up(single_hand_landmarks)
                    if fingers == [False, False, False, False, False]: # Closed Fist
                        toggle_mute()
                        last_mode_change_time = current_time # Reset cooldown
                        # Mode stays NONE after discrete action
                    elif all(fingers): # Open Palm (All fingers up)
                        activate_task_view()
                        last_mode_change_time = current_time # Reset cooldown
                        # Mode stays NONE after discrete action


        # --- Execute Actions based on Current Mode ---
        if current_mode == Mode.VOLUME_CONTROL and num_hands_detected == 2 and 'Left' in hands_data and 'Right' in hands_data:
            # Get index finger tips from both hands
            left_index_tip_x, left_index_tip_y = get_landmark_coords(hands_data['Left'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)
            right_index_tip_x, right_index_tip_y = get_landmark_coords(hands_data['Right'].landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)

            length = euclidean_distance((left_index_tip_x, left_index_tip_y), (right_index_tip_x, right_index_tip_y))
            # Calibrate these thresholds based on your arm's reach and camera distance
            # You'll likely need to adjust these significantly for two hands
            min_length_thresh = 50   # Hands very close
            max_length_thresh = 500  # Hands far apart (filling most of the camera frame)
            vol_percent = int(np.interp(length, [min_length_thresh, max_length_thresh], [0, 100]))
            vol_percent = int(np.clip(vol_percent, 0, 100)) # Ensure 0-100
            set_system_volume(vol_percent)

            # Visual feedback for volume control
            cv2.circle(frame, (left_index_tip_x, left_index_tip_y), 10, (255, 0, 0), cv2.FILLED) # Blue on left index
            cv2.circle(frame, (right_index_tip_x, right_index_tip_y), 10, (0, 0, 255), cv2.FILLED) # Red on right index
            cv2.line(frame, (left_index_tip_x, left_index_tip_y), (right_index_tip_x, right_index_tip_y), (255, 255, 0), 3) # Yellow line

        elif current_mode == Mode.MOUSE_CONTROL and num_hands_detected == 1:
            # Find the single hand detected for mouse control
            single_hand_landmarks = None
            if 'Left' in hands_data: # Prioritize left hand for mouse if both could be detected, or pick one.
                single_hand_landmarks = hands_data['Left']
            elif 'Right' in hands_data:
                single_hand_landmarks = hands_data['Right']

            if single_hand_landmarks:
                # Use Index finger tip for mouse movement
                index_tip_x_px, index_tip_y_px = get_landmark_coords(single_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP], w, h)

                # Smooth mouse movement by gradually moving towards target
                target_x = np.interp(index_tip_x_px, [0, w], [0, screen_w])
                target_y = np.interp(index_tip_y_px, [0, h], [0, screen_h])

                # Apply smoothing
                smooth_x = prev_mouse_x + (target_x - prev_mouse_x) / mouse_smooth_factor
                smooth_y = prev_mouse_y + (target_y - prev_mouse_y) / mouse_smooth_factor

                mouse.position = (smooth_x, smooth_y)
                prev_mouse_x, prev_mouse_y = smooth_x, smooth_y

                # Visual feedback for mouse
                cv2.circle(frame, (index_tip_x_px, index_tip_y_px), 15, (0, 255, 255), cv2.FILLED) # Yellow for index finger

                # Detect pinch for click (Thumb and Index)
                thumb_tip_x_px, thumb_tip_y_px = get_landmark_coords(single_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP], w, h)
                pinch_length = euclidean_distance((index_tip_x_px, index_tip_y_px), (thumb_tip_x_px, thumb_tip_y_px))

                if pinch_length < 30: # Threshold for a pinch (adjust as needed)
                    if current_time - last_click_time > mouse_click_debounce_time:
                        mouse.click(Button.left, 1)
                        last_click_time = current_time
                        print("Mouse Clicked!")
                        cv2.circle(frame, (index_tip_x_px, index_tip_y_px), 20, (0, 0, 255), cv2.FILLED) # Red on click
        else: # No hand detected, or hands don't match current mode gesture
            # Reset mode to NONE after cooldown if no hand(s) detected
            # or if the current gesture doesn't match the active mode
            if current_mode != Mode.NONE and (not hand_detected_in_frame or \
               (current_mode == Mode.VOLUME_CONTROL and num_hands_detected != 2) or \
               (current_mode == Mode.MOUSE_CONTROL and num_hands_detected != 1)):
                if current_time - last_mode_change_time > mode_change_cooldown:
                    current_mode = Mode.NONE
                    print("Mode: IDLE (Hand(s) not detected or gesture changed)")


        # --- Display Current Mode and Status on Screen ---
        mode_text_color = (0, 255, 255) # Yellow
        info_text_color = (0, 255, 0)   # Green

        mode_text = "MODE: "
        if current_mode == Mode.NONE:
            mode_text += "IDLE (Show Hand(s) for Control)"
        elif current_mode == Mode.VOLUME_CONTROL:
            mode_text += f"VOLUME ({int(current_volume_percent)}%)"
            # Draw volume bar
            bar_height = 200
            bar_width = 25
            bar_x = w - bar_width - 20
            bar_y = h // 2 - bar_height // 2
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 0), 3) # Black border
            fill_height = int(np.interp(current_volume_percent, [0, 100], [0, bar_height]))
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - fill_height), (bar_x + bar_width, bar_y + bar_height), info_text_color, cv2.FILLED) # Green fill
            cv2.putText(frame, f"{int(current_volume_percent)}%", (bar_x - 10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, info_text_color, 2)

        elif current_mode == Mode.MOUSE_CONTROL:
            mode_text += "MOUSE (Pinch to Click)"
            # Mouse cursor indicator on screen (relative to webcam frame)
            # Map actual mouse position (screen_w, screen_h) back to the webcam frame (w, h) for display
            display_mouse_x = int(np.interp(mouse.position[0], [0, screen_w], [0, w]))
            display_mouse_y = int(np.interp(mouse.position[1], [0, screen_h], [0, h]))
            cv2.circle(frame, (display_mouse_x, display_mouse_y), 10, (255, 0, 255), 2) # Magenta outline for cursor

        cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_text_color, 2)

        cv2.imshow('Virtual Desktop Assistant', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

if __name__ == "__main__":
    main()