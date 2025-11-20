# 👋 Hand Gesture Window Control System

## 🌟 Project Overview

This project implements a real-time hand gesture recognition system using the laptop's built-in camera, leveraging **MediaPipe Hands** and **OpenCV**. It allows users to control the active window on their desktop (Maximize or Minimize) by simply performing hand gestures in front of the camera.

## ✨ Features

  * **Real-Time Detection:** Utilizes MediaPipe to detect and track 21 key landmarks on a single hand with low latency.
  * **Window Management:** Translates specific hand poses into standard Windows operating system commands.
  * **Gesture-to-Action Mapping:**
      * ✋ **Open Hand (Five Fingers Spread):** Maximizes the current active window.
      * ✊ **Closed Hand (Fist):** Minimizes the current active window.
  * **Cooldwon Mechanism:** Includes a cool-down period (`1.0s`) to prevent rapid, accidental command execution.

## 🛠️ Installation & Setup

### Prerequisites

  * Python (3.9 to 3.12 recommended. **Avoid 3.13 due to known dependency conflicts with MediaPipe/NumPy**).
  * A functional camera integrated into your laptop.

### 1\. Install Dependencies

Install all required Python libraries, ensuring compatibility by specifying the NumPy version required by MediaPipe. If using the Tsinghua mirror:

```bash
# Recommended: Ensure you are using a compatible Python version (3.12 or 3.11)
pip install opencv-python mediapipe numpy==1.26.4 pyautogui -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2\. Configure PyAutoGUI (Windows Specific)

This system uses `pyautogui` to simulate the `Win + Up` and `Win + Down` keyboard shortcuts. No extra configuration is usually required, but ensure your terminal or IDE environment is not preventing keyboard simulation.

## 🚀 Usage

1.  **Save the Script:** Ensure the project code is saved as **`gesture_control_system.py`** in your project folder (`D:\GestureCtrl`).

2.  **Run the System:** Execute the script from your terminal:

    ```bash
    python gesture_control_system.py
    ```

3.  **Activate Window:** Ensure the application you wish to control (e.g., browser, file explorer) is the **active window** on your screen.

4.  **Perform Gestures:** Face your hand towards the camera and perform the gestures listed below.

## 👋 Gesture Guide

| Gesture | Pose | Action | Command Simulated |
| :--- | :--- | :--- | :--- |
| **Maximize** | **Open Hand (Fingers Spread)** | Makes the active window full screen. | `Win` + `Up Arrow` |
| **Minimize** | **Closed Hand (Fist)** | Minimizes the active window to the taskbar. | `Win` + `Down Arrow` |

## ⚙️ Troubleshooting

| Issue | Potential Cause & Fix |
| :--- | :--- |
| `ModuleNotFoundError: No module named 'cv2'` | **Environment Mismatch:** Ensure you are running the script with the exact Python interpreter (`python.exe`) where you ran the `pip install` commands. **Python Version:** Confirm you are not using Python 3.13. |
| Gesture is **not responsive** | **Lighting:** Ensure the area around your hand is well-lit and the background is uncluttered. **Thresholds:** The detection thresholds (`0.08` and `0.05` in the code) might need slight adjustment based on your camera and hand size. |
| **Window closes unexpectedly** | This is usually due to misclassification. Try to hold the Open/Closed pose firmly until the action executes, then relax your hand outside the frame before the next gesture. |


