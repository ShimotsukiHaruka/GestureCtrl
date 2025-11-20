import cv2
import time

# --- 配置 ---
# 0 通常是默认的 Integrated Camera
CAMERA_INDEX = 0
# 尝试设置常用的高清分辨率。如果不支持，OpenCV会使用默认值。
WIDTH = 1280
HEIGHT = 720
# ----------------

def test_camera():
    """打开摄像头并测试其分辨率和实时帧率。"""
    
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("🔴 错误：无法打开摄像头。请检查是否有其他应用（如Teams/Zoom）正在占用它。")
        return

    # 尝试设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # 实际获取的宽度和高度
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ 摄像头已打开。")
    print(f"--- 目标分辨率: {WIDTH}x{HEIGHT} | 实际分辨率: {actual_width}x{actual_height} ---")
    print("🎥 实时视频窗口已弹出，按 'q' 键退出测试。")

    # 用于测量 FPS 的变量
    frame_count = 0
    start_time = time.time()
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        if not ret:
            print("🔴 无法从摄像头接收帧，退出。")
            break

        frame_count += 1
        
        # 每隔 30 帧计算一次 FPS
        if frame_count % 30 == 0:
            end_time = time.time()
            # 计算 FPS
            fps = 30 / (end_time - start_time)
            # 打印到终端
            print(f"实时 FPS: {fps:.2f}")
            
            # 重置计时器
            start_time = time.time()
            frame_count = 0
        
        # 显示视频流
        cv2.imshow('Camera Test - Press Q to Quit', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头资源并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()
    print("测试结束。")

if __name__ == "__main__":
    test_camera()