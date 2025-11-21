import cv2
import mediapipe as mp
import pyautogui
import time
import sys
import numpy as np
import math 
import ctypes 

# --- 检查库是否正确安装 ---
try:
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"MediaPipe Version: {mp.__version__}")
except AttributeError:
    print("🔴 错误：cv2 或 mediapipe 库未正确加载，请检查您的 Python 环境和依赖。")
    sys.exit(1)

# --- 初始化 MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,    # 只需检测一只手
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.6 
)
mp_drawing = mp.solutions.drawing_utils

# --- 初始化摄像头 ---
cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    print("🔴 错误：无法打开摄像头。")
    sys.exit(1)

# --- 配置控制参数 ---
COOLDOWN_TIME = 1.0     # 窗口/系统动作冷却时间（秒）
CLICK_COOLDOWN_TIME = 0.3 # 鼠标点击的冷却时间
last_action_time = time.time() - COOLDOWN_TIME 
last_click_time = time.time() - CLICK_COOLDOWN_TIME

# 鼠标移动相关参数
start_x, start_y = 0, 0     # 相对移动锚点
MOUSE_SENSITIVITY = 1.5     # 鼠标移动灵敏度

# 核心阈值（角度）
STRAIGHT_ANGLE_THRESHOLD = 160 
BENT_ANGLE_THRESHOLD = 150     

# 四指的关键点序列（MCP -> PIP -> TIP）
FINGER_JOINTS = [
    [5, 6, 8],    # 食指
    [9, 10, 12],  # 中指
    [13, 14, 16], # 无名指
    [17, 18, 20]  # 小指
]

# --- 核心函数：计算三点夹角 (保持不变) ---
def calculate_angle(p1, p2, p3):
    p1_coords = np.array([p1.x, p1.y, p1.z])
    p2_coords = np.array([p2.x, p2.y, p2.z])
    p3_coords = np.array([p3.x, p3.y, p3.z])
    vec1 = p1_coords - p2_coords
    vec2 = p3_coords - p2_coords
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm_product == 0: return 180.0 
    cosine_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)

# --- 核心函数：判断手指状态 (保持不变) ---
def is_finger_straight(hand_landmarks, joints, threshold):
    p_mcp = hand_landmarks.landmark[joints[0]]
    p_pip = hand_landmarks.landmark[joints[1]]
    p_tip = hand_landmarks.landmark[joints[2]]
    angle = calculate_angle(p_mcp, p_pip, p_tip)
    return angle > threshold

# --- 核心手势识别函数（V13.3 逻辑） ---
def get_hand_gesture(hand_landmarks):
    
    THUMB_CMC = mp_hands.HandLandmark.THUMB_CMC.value 
    THUMB_MP_INDEX = 2 
    THUMB_IP = mp_hands.HandLandmark.THUMB_IP.value 
    
    thumb_angle = calculate_angle(hand_landmarks.landmark[THUMB_CMC], hand_landmarks.landmark[THUMB_MP_INDEX], hand_landmarks.landmark[THUMB_IP])
    thumb_open = thumb_angle > BENT_ANGLE_THRESHOLD

    finger_states = []
    for joints in FINGER_JOINTS:
        is_open = is_finger_straight(hand_landmarks, joints, STRAIGHT_ANGLE_THRESHOLD)
        finger_states.append(is_open)

    index_open, middle_open, ring_open, pinky_open = finger_states

    four_fingers_closed = not index_open and not middle_open and not ring_open and not pinky_open
    all_fingers_open = [thumb_open, index_open, middle_open, ring_open, pinky_open]

    # --- 手势逻辑判断（优先级从高到低）---
    
    # 📐 L_SHAPE (L 形手势) -> 鼠标移动
    # 逻辑：拇指和食指伸直
    if thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
        return "L_SHAPE"
        
    # ✊ CLOSE_HAND (全掌收拢/拳头) -> 最小化
    elif not thumb_open and four_fingers_closed:
        return "CLOSE_HAND" 
    
    # ✋ OPEN_HAND (张开手掌) -> 最大化/恢复
    elif all(all_fingers_open):
        return "OPEN_HAND"
        
    # ✌️ V_SIGN (剪刀手) -> 任务视图
    elif not thumb_open and index_open and middle_open and not ring_open and not pinky_open:
        return "V_SIGN"
        
    # 👆 INDEX_FINGER (食指指向) -> 鼠标左键点击
    # 逻辑：仅食指伸直，其他四指收拢
    elif not thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
        return "INDEX_FINGER"
        
    # 🖕 MIDDLE_FINGER (中指指向) -> 锁定屏幕
    elif not thumb_open and not index_open and middle_open and not ring_open and not pinky_open:
        return "MIDDLE_FINGER"
        
    # THUMB_UP 逻辑已移除

    return "UNKNOWN"

# --- 核心函数：相对鼠标移动控制 ---
def control_mouse_by_relative_movement(hand_landmarks, frame_width, frame_height):
    """
    根据 L 形手势中食指尖的相对位移控制鼠标。
    """
    global start_x, start_y, MOUSE_SENSITIVITY
    
    # 使用食指尖作为移动锚点
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value]
    current_x = index_finger_tip.x
    current_y = index_finger_tip.y
    
    if start_x == 0 and start_y == 0:
        # 第一次检测到 L_SHAPE，设置锚点
        start_x, start_y = current_x, current_y
        return "READY" 

    dx = current_x - start_x
    dy = current_y - start_y
    
    # 将归一化位移映射到屏幕像素位移
    move_x = int(dx * frame_width * MOUSE_SENSITIVITY)
    move_y = int(dy * frame_height * MOUSE_SENSITIVITY) 

    # 只有当移动量大于微小阈值时才移动
    if abs(move_x) > 1 or abs(move_y) > 1:
        pyautogui.move(move_x, move_y)
        # 更新锚点，保持相对移动
        start_x, start_y = current_x, current_y 
        return "MOVING"
    else:
        return "STILL"


# --- 主循环 ---
print("=== V13.3 最终精简版手势控制系统 ===")
print("手势功能说明：")
print("📐 L形手势 -> **主导光标移动**")
print("👆 食指指向 -> **鼠标左键点击**")
print("✌️ 剪刀手 -> 恢复所有窗口 (Win+Tab)")
print("✋ 全掌 -> 窗口最大化 (Win+Up)")
print("✊ 拳头 -> 窗口缩小/最小化 (Win+Down)")
print("🖕 中指指向 -> 锁定屏幕 (Win API)")
print("在视频窗口中按 'q' 键退出。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 镜像翻转
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    current_time = time.time()
    
    hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
    
    if hand_landmarks_list:
        frame_height, frame_width, _ = frame.shape
        hand_landmarks = hand_landmarks_list[0] 
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        current_gesture = get_hand_gesture(hand_landmarks)

        # 默认为空操作
        display_action = ""
        action_performed = None

        # --- A. 鼠标移动/点击逻辑 (L_SHAPE 和 INDEX_FINGER) ---
        if current_gesture == "L_SHAPE":
            # L形手势：相对光标移动
            move_state = control_mouse_by_relative_movement(hand_landmarks, frame_width, frame_height)
            display_action = f"MOUSE: {move_state}"
            
        elif current_gesture == "INDEX_FINGER":
            # 食指指向：鼠标左键点击
            if current_time - last_click_time > CLICK_COOLDOWN_TIME:
                 pyautogui.click()
                 last_click_time = current_time
                 display_action = "MOUSE: LEFT CLICK"
            else:
                 display_action = "MOUSE: CLICK COOLDOWN"
            
        else:
            # 非移动手势时，重置锚点，防止光标跳跃
            start_x, start_y = 0, 0 
            
            # --- B. 窗口/系统动作逻辑 (ACTIONS) ---
            if current_time - last_action_time > COOLDOWN_TIME:
                
                # ✌️ V_SIGN (剪刀手) -> 恢复所有已打开窗口 (Task View)
                if current_gesture == "V_SIGN": 
                    pyautogui.hotkey('win', 'tab')
                    action_performed = "恢复所有窗口 (Task View)"
                    
                # ✋ OPEN_HAND (张开手掌) -> 最大化
                elif current_gesture == "OPEN_HAND":
                    pyautogui.hotkey('win', 'up')
                    action_performed = "窗口最大化"

                # ✊ CLOSE_HAND (拳头) -> 最小化
                elif current_gesture == "CLOSE_HAND": 
                    pyautogui.hotkey('win', 'down')
                    action_performed = "窗口缩小/最小化"
                
                # 🖕 MIDDLE_FINGER (中指指向) -> 锁定屏幕
                elif current_gesture == "MIDDLE_FINGER":
                    ctypes.windll.user32.LockWorkStation()
                    action_performed = "锁定屏幕 (API)"
                
                if action_performed:
                    print(f"✅ {current_gesture} -> {action_performed}")
                    last_action_time = current_time

        # 显示当前状态
        display_text = f"GESTURE: {current_gesture}"
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, display_action, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


    # --- 2. 未检测到手的提示 ---
    if not results.multi_hand_landmarks:
          cv2.putText(frame, "No Hand Detected", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          # 丢失手部时，重置光标锚点
          start_x, start_y = 0, 0

    cv2.imshow('Hand Gesture Control V13.3 (Minimal Single Mode)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 资源释放 ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("手势控制已退出。")