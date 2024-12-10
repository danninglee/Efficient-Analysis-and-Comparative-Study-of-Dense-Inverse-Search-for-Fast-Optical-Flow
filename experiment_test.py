import numpy as np
import cv2 as cv
import time
import os
import pandas as pd

# 绘制光流矢量场箭头可视化
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# 绘制光流矢量场彩色可视化
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: Motion direction
    hsv[..., 1] = 255  # Saturation: Full
    hsv[..., 2] = np.clip(mag * 15, 0, 255).astype(np.uint8)  # Value: Motion magnitude
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# 测试单一光流方法
def run_optical_flow(method, prev_gray, gray, flow_prev=None, **kwargs):
    start = time.time()
    if method == "DIS":
        dis = kwargs["dis_instance"]
        flow = dis.calc(prev_gray, gray, flow_prev)
    elif method == "Farneback":
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None,
                                           pyr_scale=kwargs.get("pyr_scale", 0.5),
                                           levels=kwargs.get("levels", 3),
                                           winsize=kwargs.get("winsize", 15),
                                           iterations=kwargs.get("iterations", 3),
                                           poly_n=kwargs.get("poly_n", 5),
                                           poly_sigma=kwargs.get("poly_sigma", 1.2),
                                           flags=kwargs.get("flags", 0))
    elif method == "TVL1":
        tvl1 = kwargs["tvl1_instance"]
        flow = tvl1.calc(prev_gray, gray, None)
    elif method == "DeepFlow":
        deepflow = kwargs["deepflow_instance"]
        flow = deepflow.calc(prev_gray, gray, None)
    elif method == "DenseRLOF":
        rlof = kwargs["rlof_instance"]
        prev_bgr = cv.cvtColor(prev_gray, cv.COLOR_GRAY2BGR)
        curr_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        flow = rlof.calc(prev_bgr, curr_bgr, None)
    else:
        raise ValueError(f"Unsupported method: {method}")
    elapsed_time = (time.time() - start) * 1000  # 毫秒
    return flow, elapsed_time

def save_results_to_csv(performance, output_path):
    data = []
    for method, metrics in performance.items():
        avg_fps = np.mean(metrics["fps"])
        avg_latency = np.mean(metrics["latency"])
        data.append({"Method": method, "Average FPS": avg_fps, "Average Latency (ms)": avg_latency})
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

def main():
    # 初始化摄像头
    cap = cv.VideoCapture(0)
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # 初始化光流方法实例
    dis_instance = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis_instance.setUseSpatialPropagation(False)
    tvl1_instance = cv.optflow.createOptFlow_DualTVL1()
    deepflow_instance = cv.optflow.createOptFlow_DeepFlow()
    rlof_instance = cv.optflow.createOptFlow_DenseRLOF()

    methods = ["DIS", "Farneback", "TVL1", "DeepFlow", "DenseRLOF"]
    flow_instances = {
        "DIS": {"dis_instance": dis_instance},
        "Farneback": {},
        "TVL1": {"tvl1_instance": tvl1_instance},
        "DeepFlow": {"deepflow_instance": deepflow_instance},
        "DenseRLOF": {"rlof_instance": rlof_instance}
    }

    # 实时性能记录
    performance = {method: {"fps": [], "latency": []} for method in methods}

    method_index = 0
    start_time = time.time()

    # 实时处理视频流
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 当前算法
        method = methods[method_index]
        flow, elapsed_time = run_optical_flow(method, prev_gray, gray, **flow_instances[method])
        arrows_vis = draw_flow(gray, flow)
        color_vis = draw_hsv(flow)

        # 显示实时效果
        combined_vis = np.hstack((arrows_vis, color_vis))
        cv.putText(combined_vis, f"Method: {method}", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow("Real-Time Optical Flow", combined_vis)

        # 记录性能
        fps = 1000 / elapsed_time if elapsed_time > 0 else 0
        performance[method]["fps"].append(fps)
        performance[method]["latency"].append(elapsed_time)

        # 切换算法测试，每 5 秒切换
        if time.time() - start_time > 5:
            method_index = (method_index + 1) % len(methods)
            start_time = time.time()
            print(f"Switching to method: {methods[method_index]}")

        # 按下 Esc 退出
        if cv.waitKey(1) & 0xFF == 27:
            break

        prev_gray = gray

    cap.release()
    cv.destroyAllWindows()

    # 保存结果到 CSV 文件
    save_results_to_csv(performance, "data/real_time_result.csv")

if __name__ == "__main__":
    main()
