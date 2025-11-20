import cv2
import mediapipe as mp
import pyautogui
import time
import sys
import numpy as np
import math 
import ctypes # <-- 用于调用 Windows API 锁屏

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
scroll_threshold = 0.03 # 滚动触发的 Y 轴归一化移动阈值
SCROLL_SPEED = 15      # 每次滚动操作的幅度

# 模式控制状态
scroll_mode_active = False 
last_scroll_y = 0.5        # 用于跟踪手腕的Y坐标

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

# --- 核心函数：计算三点夹角 ---
def calculate_angle(p1, p2, p3):
    """
    计算由三个关键点 p1, p2, p3 形成的夹角，p2 为顶点。
    """
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

# --- 核心函数：判断手指状态 ---
def is_finger_straight(hand_landmarks, joints, threshold):
    """判断手指是否伸直（夹角是否大于阈值）。"""
    p_mcp = hand_landmarks.landmark[joints[0]]
    p_pip = hand_landmarks.landmark[joints[1]]
    p_tip = hand_landmarks.landmark[joints[2]]
    
    angle = calculate_angle(p_mcp, p_pip, p_tip)
    
    return angle > threshold

# --- 核心手势识别函数（V11.8 逻辑） ---
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
    
    # 👎 THUMB_DOWN (拇指向下) -> 向下滚动/刷下一条
    # 逻辑：拇指伸直 (open)，其他四指全部收拢 (not open)
    elif thumb_open and not index_open and not middle_open and not ring_open and not pinky_open:
        return "THUMB_DOWN" 

    # 📐 L_SHAPE (L 形手势) -> 滚动模式切换
    # 逻辑：拇指伸直 (open)，食指伸直 (open)，其他三指收拢 (not open)
    elif thumb_open and index_open and not middle_open and not ring_open and not pinky_open:
        return "SCROLL_MODE_TOGGLE"

    # ✋ OPEN_HAND (张开手掌) -> 窗口最大化 (五指全开)
    elif all(all_fingers_open):
        return "OPEN_HAND"
        
    # 🖐️ THREE_FINGER_CLENCH (三指并拢) -> 窗口缩小/最小化 (新逻辑)
    # 逻辑：拇指收拢，小指收拢，食指、中指、无名指伸直
    elif not thumb_open and index_open and middle_open and ring_open and not pinky_open:
         return "THREE_FINGER_CLENCH" # <--- 最小化新手势
    
    return "UNKNOWN"

# --- 滚动控制函数：基于手腕 Y 轴移动 (手掌移动) ---
def control_scroll_by_palm(hand_landmarks):
    """根据手腕关键点 (WRIST) 的Y坐标变化来模拟鼠标滚轮操作。"""
    global last_scroll_y, SCROLL_SPEED, scroll_threshold
    
    # 使用 WRIST 关键点 (索引 0) 作为手掌的中心点
    current_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].y
    
    y_diff = current_y - last_scroll_y
    scroll_action = None
    
    # Y轴向下增大。手掌向下移动 (current_y > last_scroll_y) 意味着向下滚动页面。
    if y_diff > scroll_threshold: # Y轴增大，向下滚动页面
        pyautogui.scroll(-SCROLL_SPEED) 
        scroll_action = "DOWN"
    elif y_diff < -scroll_threshold: # Y轴减小，向上滚动页面
        pyautogui.scroll(SCROLL_SPEED) 
        scroll_action = "UP"

    # 实时更新位置，使滚动更平滑
    last_scroll_y = current_y
    return scroll_action
    
# --- 主循环 ---
print("=== V11.8 稳定版手势控制系统 (三指并拢 最小化) ===")
print("手势功能说明：")
print("✌️ V字手势 -> 锁定屏幕 (Win API)")
print("👎 拇指向下 (Thumbs Down) -> 向下滚动 (自动刷短视频)")
print("📐 L形手势 (拇指、食指伸直) -> 模式切换：激活/退出 手掌滚动模式")
print("✋ 张开手掌 -> 窗口最大化 (win+up)")
print("🖐️ 三指并拢 (食中无伸直，拇小指收拢) -> **窗口缩小/最小化 (win+down)**")
print("在视频窗口中按 'q' 键退出。")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) # 镜像翻转
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
            
        # 模式切换控制 (L形手势)
        if current_gesture == "SCROLL_MODE_TOGGLE":
            if current_time - last_action_time > COOLDOWN_TIME:
                scroll_mode_active = not scroll_mode_active
                print(f"🔄 模式切换 (L形): 滚动模式 {'已激活' if scroll_mode_active else '已退出'}")
                
                if scroll_mode_active:
                    # 激活时重置跟踪位置，使用 WRIST (索引 0)
                    last_scroll_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value].y
                    
                last_action_time = current_time
                gesture_detected = True
        
        # 1. 滚动模式激活时，只执行滚动操作
        if scroll_mode_active:
            # *** 调用手掌滚动函数 ***
            scroll_action = control_scroll_by_palm(hand_landmarks) 
            if scroll_action:
                cv2.putText(frame, f"PALM SCROLLING: {scroll_action.upper()}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            current_gesture = "SCROLL_ACTIVE" 

        # 2. 滚动模式未激活时，执行其他单次动作（受冷却时间限制）
        elif current_time - last_action_time > COOLDOWN_TIME and not gesture_detected:
            
            action_performed = None
            
            if current_gesture == "V_SIGN":
                # *** 锁屏功能 (V字手势) ***
                ctypes.windll.user32.LockWorkStation()
                action_performed = "锁定屏幕 (API)"
                
            elif current_gesture == "THUMB_DOWN": 
                # *** 自动刷短视频/向下滚动功能 (拇指向下) ***
                pyautogui.scroll(-20) 
                action_performed = "向下滚动 (刷下一条)"
            
            elif current_gesture == "OPEN_HAND":
                # *** 窗口最大化功能 (张开手掌) ***
                pyautogui.hotkey('win', 'up')
                action_performed = "窗口最大化"

            elif current_gesture == "THREE_FINGER_CLENCH": # <--- 触发最小化
                # *** 窗口缩小/最小化功能 (三指并拢) ***
                pyautogui.hotkey('win', 'down')
                action_performed = "窗口缩小/最小化"
            
            if action_performed:
                print(f"✅ {current_gesture} -> {action_performed}")
                last_action_time = current_time
                gesture_detected = True
        
        # 显示当前状态
        display_text = f"GESTURE: {current_gesture}"
        mode_text = f"MODE: {'SCROLL' if scroll_mode_active else 'ACTIONS'}"
        
        # 调整滚动模式下的文本显示位置，避免覆盖
        mode_y_pos = 90 if scroll_mode_active else 60 
        
        cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, mode_text, (10, mode_y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)


    # 显示未检测到手的提示
    if not results.multi_hand_landmarks:
          cv2.putText(frame, "No Hand Detected", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
          # 退出滚动模式，防止误触
          if scroll_mode_active:
              scroll_mode_active = False
              print("🚫 手部丢失，退出滚动模式。")

    cv2.imshow('Robust Hand Gesture Control V11.8', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 资源释放 ---
cap.release()
cv2.destroyAllWindows()
hands.close()
print("手势控制已退出。")