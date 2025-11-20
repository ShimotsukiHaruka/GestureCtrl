import cv2
import mediapipe as mp
import pyautogui
import time
import sys
import numpy as np
import math
import ctypes # <-- 导入 ctypes 用于调用 Windows API 锁屏

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
    max_num_hands=1,  # 只需检测一只手
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
COOLDOWN_TIME = 1.0  # 手势触发冷却时间（秒）
last_action_time = time.time() - COOLDOWN_TIME 
scroll_threshold = 0.05 # 滚动触发的 Y 轴归一化移动阈值
SCROLL_SPEED = 15      # 每次滚动操作的幅度

# 模式控制状态
scroll_mode_active = False 
last_scroll_y = 0.5        # 用于跟踪食指的Y坐标

# 核心阈值（角度）
STRAIGHT_ANGLE_THRESHOLD = 160 # 角度大于 160 度视为伸直（四指）
BENT_ANGLE_THRESHOLD = 150     # 拇指的角度阈值

# 四指的关键点序列（MCP -> PIP -> TIP）
FINGER_JOINTS = [
    [5, 6, 8],   # 食指
    [9, 10, 12], # 中指
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
    
    if norm_product == 0:
        return 180.0 
        
    cosine_angle = dot_product / norm_product
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# --- 核心函数：判断手指状态 (保持不变) ---
def is_finger_straight(hand_landmarks, joints, threshold):
    p_mcp = hand_landmarks.landmark[joints[0]]
    p_pip = hand_landmarks.landmark[joints[1]]
    p_tip = hand_landmarks.landmark[joints[2]]
    
    angle = calculate_angle(p_mcp, p_pip, p_tip)
    
    return angle > threshold

# --- 核心手势识别函数（V11.0 逻辑） ---
def get_hand_gesture(hand_landmarks):
    
    # 关键点索引常量
    THUMB_CMC = mp_hands.HandLandmark.THUMB_CMC.value 
    THUMB_MP_INDEX = 2 
    THUMB_IP = mp_hands.HandLandmark.THUMB_IP.value 
    
    # 1. 判断拇指状态
    thumb_angle = calculate_angle(
        hand_landmarks.landmark[THUMB_CMC],      
        hand_landmarks.landmark[THUMB_MP_INDEX], 
        hand_landmarks.landmark[THUMB_IP]        
    )
    thumb_open = thumb_angle > BENT_ANGLE_THRESHOLD

    # 2. 判断四指状态
    finger_states = []
    for joints in FINGER_JOINTS:
        is_open = is_finger_straight(hand_landmarks, joints, STRAIGHT_ANGLE_THRESHOLD)
        finger_states.append(is_open)

    index_open, middle_open, ring_open, pinky_open = finger_states

    # 组合状态列表
    all_fingers_open = [thumb_open, index_open, middle_open, ring_open, pinky_open]

    # --- 手势逻辑判断（优先级从高到低）---
    
    # ✌️ V_SIGN (剪刀手) -> 锁定屏幕
    if not thumb_open and index_open and middle_open and not ring_open and not pinky_open:
        return "V_SIGN"
    
    # 👆 POINTING (食指伸出) -> 滚动模式切换
    elif not thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
        return "SCROLL_MODE_TOGGLE"

    # ✋ OPEN_HAND (张开手掌) -> 窗口最大化 (功能恢复)
    elif all(all_fingers_open):
        return "OPEN_HAND"
        
    # ✊ CLOSED_FIST (握拳) -> 窗口缩小/最小化
    elif not index_open and not middle_open and not ring_open and not pinky_open:
         return "CLOSED_FIST"
    
    # 移除了所有其他不用的手势，如 THUMB_UP
    
    return "UNKNOWN"

# --- 滚动控制函数：基于食指 Y 轴移动 (保持不变) ---
def control_scroll_by_index(hand_landmarks):
    global last_scroll_y, SCROLL_SPEED, scroll_threshold
    
    current_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y
    
    y_diff = current_y - last_scroll_y
    scroll_action = None
    
    if y_diff > scroll_threshold: 
        pyautogui.scroll(-SCROLL_SPEED) 
        scroll_action = "DOWN"
    elif y_diff < -scroll_threshold:
        pyautogui.scroll(SCROLL_SPEED) 
        scroll_action = "UP"

    last_scroll_y = current_y
    return scroll_action
    
# --- 主循环 ---
print("=== V11.0 稳定版手势控制系统 (窗口最大化恢复) ===")
print("手势功能说明：")
print("✌️ V字手势 -> **锁定屏幕 (API)**")
print("✋ 张开手掌 -> **窗口最大化 (win+up)**")
print("✊ 握拳 (四指弯曲) -> 窗口缩小/最小化 (win+down)") 
print("👆 食指伸出 -> **模式切换：激活/退出 食指滚动模式**")
print("↔️ (滚动模式激活时) 食指上下移动 -> 页面滚动")
print("在视频窗口中按 'q' 键退出。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    current_time = time.time()
    gesture_detected = False

    hand_landmarks_list = results.multi_hand_landmarks if results.multi_hand_landmarks else []
    
    # --- 1. 单手手势检测 ---
    if hand_landmarks_list:
        # 只处理检测到的第一只手
        hand_landmarks = hand_landmarks_list[0] 
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
        current_gesture = get_hand_gesture(hand_landmarks)

        # --- 手势功能执行逻辑 ---
            
        # 模式切换控制 (食指伸出)
        if current_gesture == "SCROLL_MODE_TOGGLE":
             if current_time - last_action_time > COOLDOWN_TIME:
                scroll_mode_active = not scroll_mode_active
                print(f"🔄 模式切换 (食指): 滚动模式 {'已激活' if scroll_mode_active else '已退出'}")
                
                if scroll_mode_active:
                     last_scroll_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP.value].y
                     
                last_action_time = current_time
                gesture_detected = True
        
        # 1. 滚动模式激活时，只执行滚动操作
        if scroll_mode_active:
            scroll_action = control_scroll_by_index(hand_landmarks)
            if scroll_action:
                cv2.putText(frame, f"SCROLLING: {scroll_action.upper()}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            current_gesture = "SCROLL_ACTIVE" 

        # 2. 滚动模式未激活时，执行其他单次动作（受冷却时间限制）
        elif current_time - last_action_time > COOLDOWN_TIME and not gesture_detected:
            
            action_performed = None
            
            if current_gesture == "V_SIGN":
                # *** 锁屏功能 (V字手势) ***
                ctypes.windll.user32.LockWorkStation()
                action_performed = "锁定屏幕 (API)"
                
            elif current_gesture == "OPEN_HAND":
                # *** 窗口最大化功能 (张开手掌) ***
                pyautogui.hotkey('win', 'up')
                action_performed = "窗口最大化"

            elif current_gesture == "CLOSED_FIST":
                # *** 窗口缩小/最小化功能 (握拳) ***
                pyautogui.hotkey('win', 'down')
                action_performed = "窗口缩小/最小化"
            
            if action_performed:
                print(f"✅ {current_gesture} -> {action_performed}")
                last_action_time = current_time
                gesture_detected = True
        
        # 显示当前状态
        display_text = f"GESTURE: {current_gesture}"
        mode_text = f"MODE: {'SCROLL' if scroll_mode_active else 'ACTIONS'}"
        
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        mode_y_pos = 90 if scroll_mode_active else 60
        cv2.putText(frame, mode_text, (10, mode_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


    # 显示未检测到手的提示
    if not results.multi_hand_landmarks:
         cv2.putText(frame, "No Hand Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Robust Hand Gesture Control V11.0', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("手势控制已退出。")