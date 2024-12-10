import numpy as np
import cv2 as cv
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

# Draw optical flow vector field arrow visualization
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    cv.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

# Draw color visualization of the optical flow vector field
def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[..., 0], flow[..., 1]
    mag, ang = cv.cartToPolar(fx, fy)
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Hue: Motion direction
    hsv[..., 1] = 255  # Saturation: Full
    hsv[..., 2] = np.clip(mag * 15, 0, 255).astype(np.uint8)  # Value: Motion magnitude
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

# Calculate endpoint error (EPE)
def calc_epe(flow_gt, flow_pred):
    diff = flow_gt - flow_pred
    epe_map = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.mean(epe_map)

# Test single optical flow method
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
    elapsed_time = (time.time() - start) * 1000
    return flow, elapsed_time

def save_results_to_csv(results, video_name):
    data = []
    for method, metrics in results.items():
        data.append({
            "Method": method,
            "Average Runtime (ms)": metrics["time"],
            # "Average EPE": metrics["epe"]
        })
    df = pd.DataFrame(data)
    csv_path = f"data/{video_name}_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

def main():
    video_path = "test_random.mp4"  # Video file path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"output/{video_name}_output"  # Output folder path
    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

    # Initialize the optical flow method instance
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

    # Initialize result storage
    results = {method: {"time": [], "epe": []} for method in methods}
    # flow_gt = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

    writers = {method: {
        "arrows": cv.VideoWriter(
            os.path.join(output_dir, f"{method}_arrows.mp4"),
            cv.VideoWriter_fourcc(*'mp4v'), cap.get(cv.CAP_PROP_FPS),
            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        ),
        "color": cv.VideoWriter(
            os.path.join(output_dir, f"{method}_color.mp4"),
            cv.VideoWriter_fourcc(*'mp4v'), cap.get(cv.CAP_PROP_FPS),
            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
        )
    } for method in methods}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        for method in methods:
            flow, elapsed_time = run_optical_flow(method, prev_gray, gray, **flow_instances[method])
            arrows_vis = draw_flow(gray, flow)
            color_vis = draw_hsv(flow)
            
            writers[method]["arrows"].write(arrows_vis)
            writers[method]["color"].write(color_vis)

            results[method]["time"].append(elapsed_time)
            # results[method]["epe"].append(calc_epe(flow_gt, flow))

        prev_gray = gray

    cap.release()
    for writer_set in writers.values():
        writer_set["arrows"].release()
        writer_set["color"].release()

    for method in results:
        results[method]["time"] = np.mean(results[method]["time"])
        # results[method]["epe"] = np.mean(results[method]["epe"])

    save_results_to_csv(results, video_name)
    print(f"All videos saved in: {output_dir}")

if __name__ == "__main__":
    main()
