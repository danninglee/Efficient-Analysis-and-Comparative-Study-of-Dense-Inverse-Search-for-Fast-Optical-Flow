import numpy as np
import cv2 as cv
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import flowiz as fz


def calc_epe(flow_gt, flow_pred):
    """
    Calculates the end point error
    """
    diff = flow_gt - flow_pred
    epe_map = np.sqrt(np.sum(diff ** 2, axis=2))
    return np.mean(epe_map)


class ImageFlow:

    @staticmethod
    def is_valid_image(file_name):
        try:
            with Image.open(file_name) as img:
                img.verify()
                return True
        except (IOError, SyntaxError):
            return False

    @staticmethod
    def is_valid_flow(file_path):
        return file_path.endswith(".flo")

    @staticmethod
    def load_training(training_dir_path):
        image_names = []
        for image_name in os.listdir(training_dir_path):
            image_names.append(image_name)

        image_names.sort()
        images = []
        for image_name in image_names:
            image_path = os.path.join(training_dir_path, image_name)
            if not ImageFlow.is_valid_image(image_path):
                continue
            images.append(cv.imread(image_path))
        return images

    @staticmethod
    def load_flows(flow_dir_path):
        """
            Return color coded flows as a numpy array under a directory, flow images are returns as BGR
        """
        flow_names = []
        for flow_name in os.listdir(flow_dir_path):
            if not ImageFlow.is_valid_flow(flow_name):
                continue
            flow_names.append(os.path.join(flow_dir_path, flow_name))

        flow_names.sort()

        # images = []
        # for flow_name in flow_names:
        #     image_path = os.path.join(flow_dir_path, flow_name)
        #     if not ImageFlow.is_valid_flow(image_path):
        #         continue
        #     # img = np.array(fz.convert_from_file(image_path))
        #     img = fz.convert_from_file(image_path, mode='UV')
        #     images.append(img)
        return flow_names

    def __init__(self, training_images_path, flow_images_path):
        training_images = self.load_training(training_images_path)
        flow_images = self.load_flows(flow_images_path)
        self.training_images = training_images
        self.flow_images = flow_images

    def get_img_flow_cv(self, index):
        assert index < len(self.training_images) and index < len(
            self.training_images)
        # return self.training_images[index], cv.cvtColor(
        #     np.array(
        #         fz.convert_from_flow(self.flow_images[index], mode='RGB')
        #     ), cv.COLOR_RGB2BGR
        # )
        flow = cv.cvtColor(np.array(fz.convert_from_file(
            self.flow_images[index])), cv.COLOR_RGB2BGR)
        return self.training_images[index], flow

    def get_flow_arrow_img(self, index, step=16):
        img = self.training_images[index].copy()
        uv = self.get_flow(index)
        # print(img.shape)
        # print(uv.shape)
        for y in range(0, img.shape[0], step):
            for x in range(0, img.shape[1], step):
                u, v = uv[y, x, 0], uv[y, x, 1]
                x_1, y_1 = int(x), int(y)
                x_2, y_2 = x + u,  y + v
                x_2, y_2 = int(x_2), int(y_2)

                x_2 = max(0, min(x_2, img.shape[1] - 1))
                y_2 = max(0, min(y_2, img.shape[0] - 1))

                # print(x_1, y_1, x_2, y_2)
                cv.polylines(
                    img, [np.array([[x_1, y_1], [x_2, y_2]])], True, (255, 255, 255))
                cv.circle(img, [x_2, y_2], 1, (255, 255, 255), -1)
        return img

    def get_flow(self, index):
        flow_name = self.flow_images[index]
        uv = np.array(fz.convert_from_file(flow_name, mode='UV'))
        uv = uv - (255 / 2)
        return uv

    def __len__(self):
        return len(self.flow_images)


def run_optical_flow(method, prev_gray, gray, flow_prev=None, **kwargs):
    if method == "DIS":
        dis = kwargs["dis_instance"]
        flow = dis.calc(prev_gray, gray, flow_prev)
    elif method == "Farneback":
        flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None,
                                           pyr_scale=kwargs.get(
                                               "pyr_scale", 0.5),
                                           levels=kwargs.get("levels", 3),
                                           winsize=kwargs.get("winsize", 15),
                                           iterations=kwargs.get(
                                               "iterations", 3),
                                           poly_n=kwargs.get("poly_n", 5),
                                           poly_sigma=kwargs.get(
                                               "poly_sigma", 1.2),
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
    return flow

    # arrow_img = img_flow.get_flow_arrow_img(index)
    # cv.imshow("training", training_img)
    # cv.imshow("ground truth", flow)
    # cv.imshow("arrow", arrow_img)


def main() -> None:
    img_flow = ImageFlow("cave_4_train", "cave_4_gt")

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
    results = {method: {"epe": []} for method in methods}

    for i in range(1, len(img_flow)):
        frame_im1, _ = img_flow.get_img_flow_cv(
            i - 1)
        frame_i, _ = img_flow.get_img_flow_cv(
            i)
        ground_truth_uv = img_flow.get_flow(i)

        frame_im1 = cv.cvtColor(frame_im1, cv.COLOR_BGR2GRAY)
        frame_i = cv.cvtColor(frame_i, cv.COLOR_BGR2GRAY)

        method_uv = dis_instance.calc(frame_im1, frame_i, None)

        for method in methods:
            flow = run_optical_flow(
                method, frame_im1, frame_i, **flow_instances[method])

            epe = calc_epe(ground_truth_uv, flow)
            results[method]["epe"].append(epe)

    for method, obj in results:
        epes = obj["epe"]
        print(f"{method} : {np.mean(epes)}")


if __name__ == "__main__":
    main()
