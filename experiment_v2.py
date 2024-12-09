import numpy as np
import cv2 as cv
import time
import os
import matplotlib.pyplot as plt

# 绘制光流矢量场
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# 计算端点误差 (EPE)
def calc_epe(flow_gt, flow_pred):
    diff = flow_gt - flow_pred  # 计算光流矢量差
    epe_map = np.sqrt(np.sum(diff**2, axis=2))  # 每像素的欧几里得距离
    mean_epe = np.mean(epe_map)  # 平均端点误差
    return mean_epe

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
    elapsed_time = (time.time() - start) * 1000  # 计算耗时
    return flow, elapsed_time

# 可视化实验结果
def visualize_results(results):
    methods = list(results.keys())
    times = [results[m]["time"] for m in methods]
    epes = [results[m]["epe"] for m in methods]

    # 绘制性能对比图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(methods, times, color='skyblue')
    plt.title("Average Processing Time (ms)")
    plt.ylabel("Time (ms)")
    plt.xlabel("Method")

    plt.subplot(1, 2, 2)
    plt.bar(methods, epes, color='lightgreen')
    plt.title("Average EPE (Endpoint Error)")
    plt.ylabel("EPE")
    plt.xlabel("Method")

    plt.tight_layout()
    plt.show()

# 主实验代码
def main():
    video_path = "test_random.mp4"  # 视频文件路径
    output_dir = "output_videos_random"  # 输出文件夹路径
    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(video_path)
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

    # 初始化结果存储
    results = {method: {"time": [], "epe": []} for method in methods}
    flow_gt = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)  # 伪光流 Ground Truth

    writers = {}
    for method in methods:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv.CAP_PROP_FPS)
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        writers[method] = cv.VideoWriter(
            os.path.join(output_dir, f"{method}_output.mp4"), fourcc, fps, (width, height)
        )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for method in methods:
            flow, elapsed_time = run_optical_flow(method, prev_gray, gray, **flow_instances[method])
            vis = draw_flow(gray, flow)
            # 保存可视化结果
            writers[method].write(vis)

            # 记录时间和 EPE
            results[method]["time"].append(elapsed_time)
            results[method]["epe"].append(calc_epe(flow_gt, flow))

        prev_gray = gray

    cap.release()
    for writer in writers.values():
        writer.release()

    # 计算平均值并可视化结果
    for method in results:
        results[method]["time"] = np.mean(results[method]["time"])
        results[method]["epe"] = np.mean(results[method]["epe"])

    visualize_results(results)
    print(f"All videos saved in: {output_dir}")

if __name__ == "__main__":
    main()
