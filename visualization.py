import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

def get_middle_frame(cap):
    """Get the frame closest to the middle of the video."""
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))  # Get the total number of video frames
    middle_frame_index = total_frames // 2  # Index of the intermediate frame
    
    # Set the video capture to the middle frame
    cap.set(cv.CAP_PROP_POS_FRAMES, middle_frame_index)
    
    ret, frame = cap.read()
    if ret:
        return frame
    else:
        raise ValueError("Failed to read the middle frame.")

def create_combined_image(arrow_frame, color_frame, method):
    """Create a combined image of arrow and color flow visualizations for a method."""
    # Stack the arrow and color visualizations vertically
    combined = np.vstack([arrow_frame, color_frame])
    return combined

def save_combined_image(output_dir, methods, arrows, colors):
    """Save a 2x5 grid image for all methods."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, method in enumerate(methods):
        # For each method, we plot the arrow and color flow
        axes[0, i].imshow(arrows[method])
        axes[0, i].axis('off')
        axes[0, i].set_title(f'{method} Arrow')

        axes[1, i].imshow(colors[method])
        axes[1, i].axis('off')
        axes[1, i].set_title(f'{method} Color')

    # Save the combined image
    plt.tight_layout()
    plt.savefig(output_dir, bbox_inches='tight')
    plt.close()

def process_video_files(output_dir, methods):
    """Process each method's output to create a 2x5 grid image."""
    arrows = {}
    colors = {}

    # Iterate over all methods and process their output videos
    for method in methods:
        # Construct paths to arrow and color videos
        arrow_video_path = os.path.join(output_dir, f"{method}_arrows.mp4")
        color_video_path = os.path.join(output_dir, f"{method}_color.mp4")

        # Open the videos for each method
        cap_arrow = cv.VideoCapture(arrow_video_path)
        cap_color = cv.VideoCapture(color_video_path)

        # Get the middle frame for both arrow and color videos
        arrow_frame = get_middle_frame(cap_arrow)
        color_frame = get_middle_frame(cap_color)

        # Resize frames to a consistent size
        height, width = arrow_frame.shape[:2]
        arrow_frame_resized = cv.resize(arrow_frame, (width, height))
        color_frame_resized = cv.resize(color_frame, (width, height))

        # Store the frames in a dictionary
        arrows[method] = arrow_frame_resized
        colors[method] = color_frame_resized

        cap_arrow.release()
        cap_color.release()

    # Define output path for the combined image
    combined_image_path = os.path.join(output_dir, "combined_output.png")
    save_combined_image(combined_image_path, methods, arrows, colors)

def main():
    output_dir = "output/test_vertical_output"
    methods = ["DIS", "Farneback", "TVL1", "DeepFlow", "DenseRLOF"]

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process the video files and generate the combined image
    process_video_files(output_dir, methods)

    print("Combined image saved to:", os.path.join(output_dir, "combined_output.png"))

if __name__ == "__main__":
    main()
