import numpy as np
import cv2 as cv
import time
import os

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

# 测试单一光流方法
def run_optical_flow(method, prev_gray, gray, flow_prev=None, **kwargs):
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
    elif method == "Lucas-Kanade":
        p0 = cv.goodFeaturesToTrack(prev_gray, mask=None, **kwargs["gftt_params"])
        if p0 is not None:
            p1, st, err = cv.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **kwargs["lk_params"])
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            for i, (new, old, valid) in enumerate(zip(p1, p0, st)):
                if valid:
                    x0, y0 = old.ravel()
                    x1, y1 = new.ravel()
                    flow[int(y0), int(x0)] = [x1 - x0, y1 - y0]
        else:
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
    elif method == "DenseRLOF":
        rlof = kwargs["rlof_instance"]
        # 将灰度图转换为 BGR 格式
        prev_bgr = cv.cvtColor(prev_gray, cv.COLOR_GRAY2BGR)
        curr_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        flow = rlof.calc(prev_bgr, curr_bgr, None)
    else:
        raise ValueError(f"Unsupported method: {method}")
    return flow


# 主实验代码
def main():
    video_path = "test_vertical.mp4"  # 视频文件路径
    output_dir = "output_videos"  # 输出文件夹路径
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

    methods = ["DIS", "Farneback", "TVL1", "DeepFlow", "Lucas-Kanade", "DenseRLOF"]
    flow_instances = {
        "DIS": {"dis_instance": dis_instance},
        "Farneback": {},
        "TVL1": {"tvl1_instance": tvl1_instance},
        "DeepFlow": {"deepflow_instance": deepflow_instance},
        "Lucas-Kanade": {
            "gftt_params": {"maxCorners": 500, "qualityLevel": 0.01, "minDistance": 7, "blockSize": 7},
            "lk_params": {"winSize": (15, 15), "maxLevel": 2, 
                          "criteria": (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)}
        },
        "DenseRLOF": {"rlof_instance": rlof_instance}
    }

    # 初始化视频写入器
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
            flow = run_optical_flow(method, prev_gray, gray, **flow_instances[method])
            vis = draw_flow(gray, flow)
            writers[method].write(vis)

        prev_gray = gray  # 更新上一帧

    # 释放资源
    cap.release()
    for writer in writers.values():
        writer.release()

    print(f"All videos saved in: {output_dir}")

if __name__ == "__main__":
    main()
