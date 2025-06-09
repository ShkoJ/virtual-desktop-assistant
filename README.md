# Virtual Desktop Assistant

## Project Overview
Control your computer's volume and mouse with intuitive hand gestures via webcam, featuring **two-hand volume adjustment**, **one-hand mouse control with pinch-to-click**, **mute toggling**, and **task view activation**. Say goodbye to your mouse and keyboard for basic tasks, and hello to a more futuristic way of interacting with your digital workspace!

---

## ✨ Features
* **Two-Handed Volume Control:** Adjust your system volume by moving your index fingers (from both hands) closer or further apart.
* **One-Handed Mouse Control:** Move your mouse cursor precisely with a single index finger.
* **Pinch-to-Click:** Perform a left mouse click by pinching your thumb and index finger together.
* **Mute/Unmute Toggle:** Quickly mute or unmute your audio with a closed fist gesture.
* **Task View / Mission Control Activation:** Bring up your OS's task switcher (Windows Task View, macOS Mission Control, Linux Desktop Overview) with an open palm gesture.
* **Real-time Visual Feedback:** See your hand landmarks and current mode/status directly on the webcam feed.
* **Cross-Platform Compatibility:** Designed to work on Windows, macOS, and Linux for core functionalities.

---

## 🚀 Getting Started

### Prerequisites
* **Python 3.x** installed (preferably 3.8 or newer)
* `pip` (Python package installer), usually comes with Python
* A working **webcam**
* (Optional, but recommended for advanced Git users) `git` installed on your system if you plan to clone the repository.

### 1. Download the Project
Since you're using the drag-and-drop method, you'll simply prepare your project folder locally with all the necessary files (`hand_detector.py`, `requirements.txt`, `README.md`, `.gitignore`, `LICENSE`, and any `media` folder with GIFs/screenshots).

### 2. Install Python Dependencies
It's highly recommended to use a **virtual environment** to manage dependencies. This keeps your project's libraries separate from your system's global Python installation.

1.  **Open your terminal or command prompt.**
2.  **Navigate to your project folder** (e.g., `cd path/to/your/virtual-desktop-assistant`).
3.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```
4.  **Activate the virtual environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
5.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### 3. System-Specific Setup & Permissions

* **Windows:**
    * No special system installations are typically needed beyond the Python packages listed in `requirements.txt`.
* **macOS:**
    * **Accessibility Permissions:** You **MUST** grant `Terminal.app` (or your IDE, e.g., VS Code, PyCharm, if you run from there) **Accessibility permissions**. Go to `System Settings > Privacy & Security > Accessibility` and add your terminal application to the list. Without this, `pynput` cannot control your mouse or keyboard.
    * Volume control uses `osascript`, which is built-in.
* **Linux (Ubuntu/Debian-based example):**
    * For volume control (`amixer`, `pactl`), ensure `alsa-utils` and/or `pulseaudio-utils` are installed:
        ```bash
        sudo apt update
        sudo apt install alsa-utils pulseaudio-utils
        ```
    * For keyboard commands (Task View), `pynput` generally works, but ensure your display server (e.g., Xorg, Wayland) is configured correctly.

### 4. Important: Set Your Screen Resolution
The mouse control functionality relies on your screen's resolution to accurately map your hand movements. **You MUST edit the Python script to match your monitor's dimensions.**

1.  Open `hand_detector.py` (or whatever you named your main script) in a text editor.
2.  Find the lines near the top:
    ```python
    screen_w, screen_h = 1920, 1080 # <<<--- SET YOUR SCREEN WIDTH AND HEIGHT HERE!
    ```
3.  Replace `1920` and `1080` with your actual screen's width and height.

---

## 🚀 Usage

1.  **Activate your virtual environment** (if you closed your terminal or opened a new one):
    * **On Windows:** `.\venv\Scripts\activate`
    * **On macOS/Linux:** `source venv/bin/activate`
2.  **Run the script from your project folder:**
    ```bash
    python hand_detector.py
    ```
3.  A webcam window will open, displaying your hand movements and the assistant's current mode.

### Gesture Guide:

* **To Enter Volume Control:** Bring **both hands** into the camera frame. Extend **only your index finger on each hand**.
    * **Control:** Move your two index fingers **closer together** to decrease volume, or **further apart** to increase volume.
* **To Enter Mouse Control:** Bring **one hand** into the camera frame. Extend **only your index finger**.
    * **Control:** Move your index finger to move the mouse cursor.
    * **Click:** Pinch your thumb and index finger together (bringing them very close) to perform a **left mouse click**.
* **Toggle Mute/Unmute:** (While in `IDLE` mode) Make a **Closed Fist** with one hand.
* **Activate Task View / Mission Control:** (While in `IDLE` mode) Make an **Open Palm** (all fingers up) with one hand.
* **To Exit Any Mode:** Simply **hide your hand(s) from the camera** for a moment. The assistant will return to `IDLE` mode after a short cooldown.
* **To Quit the Application:** Press the **`q` key** on your keyboard at any time while the webcam window is active.

---

## ⚙️ Calibration Notes
You might need to adjust some thresholds in the `hand_detector.py` script for optimal performance based on your webcam's view and your hand size:

* **Volume Control:** The sensitivity for volume (how much distance translates to how much volume change) can be adjusted by changing `min_length_thresh` and `max_length_thresh` within the `VOLUME_CONTROL` block in the `main()` function. You might need to experiment with these values (e.g., print the `length` variable to see the range you achieve).
* **Mouse Click:** The `pinch_length < 30` threshold in the `MOUSE_CONTROL` block determines how close your thumb and index finger need to be for a click. If clicks are too sensitive or not sensitive enough, adjust this `30` value.

---

## Contributing
Feel free to fork the repository, open issues, or submit pull requests. Contributions are welcome!

---

##  License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
