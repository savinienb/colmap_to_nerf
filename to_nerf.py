import argparse
import json
import logging
import os
import numpy as np

from colmap import load_colmap_dataset

def convert_colmap_to_nerfstudio(colmap_dataset, colmap_dir, output_dir: str):
    """
    Convert a COLMAP dataset to the transforms.json format that Nerfstudio expects.
    This will produce a 'transforms.json' that has frames with 'fl_x', 'fl_y', 'cx',
    'cy', etc. rather than a NOTG-style transforms.
    """

    # Grab data from your loaded COLMAP structure.
    c2ws = colmap_dataset["cameras"].poses                  # List of 3x4 camera-to-world extrinsics
    intrinsics = colmap_dataset["cameras"].intrinsics       # List of [fx, fy, cx, cy]
    distortion = colmap_dataset["cameras"].distortion_parameters  # [k1, k2, p1, p2, (k3, k4, ...)]
    image_paths = colmap_dataset["image_paths"]
    w, h = colmap_dataset["cameras"].image_sizes[0]             # e.g. [width, height], if available

    frames = []

    for i in range(len(image_paths)):
        fx, fy, cx, cy = intrinsics[i]
        # You may only have k1, k2, p1, p2. Sometimes k3, k4, etc. exist too, so handle with caution
        # Or just set them to zero if you only have fewer parameters.
        if len(distortion[i]) >= 6:
            k1, k2, p1, p2, k3, k4 = distortion[i]
        else:
            # if you only have [k1, k2, p1, p2] or something
            k1, k2, p1, p2 = distortion[i][0:4]
            k3, k4 = 0.0, 0.0

        # Convert the 3x4 to a 4x4
        c2w_3x4 = c2ws[i]
        c2w_4x4 = np.eye(4, dtype=np.float32)
        c2w_4x4[:3, :4] = c2w_3x4

        # Make sure the file_path in the JSON is *relative* to the transforms.json location
        rel_img_path = os.path.relpath(image_paths[i], output_dir)
        pointcloud_path = os.path.join(colmap_dir , 'sparse/0/points3D_vishull_merged.ply')
        # Construct the frame dict that Nerfstudio's parser expects
        frame = {
            "file_path": rel_img_path,
            "fl_x": float(fx),
            "fl_y": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": float(k1),
            "k2": float(k2),
            "p1": float(p1),
            "p2": float(p2),
            # If you want to store k3, k4, you can either rename them or just store them
            "k3": float(k3),
            "k4": float(k4),
            "w": int(w),  # Make sure you have the correct width
            "h": int(h),  # and height
            "transform_matrix": c2w_4x4.tolist()
        }
        frames.append(frame)

    # Top-level dictionary for Nerfstudio style transforms
    transforms = {
        "camera_model": "OPENCV",  # or "SIMPLE_RADIAL" / "OPENCV_FISHEYE" depending on your data
        "frames": frames,
        "ply_file_path": pointcloud_path
    }

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, "transforms.json")
    with open(outpath, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Finished writing Nerfstudio-friendly transforms.json to '{outpath}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COLMAP dataset to Nerfstudio's transforms.json format.")
    parser.add_argument(
        "--colmap_dir",
        type=str,
        default="/home/sav/DATA/JT01TESTEST_1D6D584C9554",
        help="Path to the COLMAP dataset directory."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/sav/DATA/JT01TESTEST_1D6D584C9554",
        help="Where to write out the transforms.json Nerfstudio expects."
    )
    args = parser.parse_args()

    # Load the COLMAP dataset (adapt to your own loader)
    colmap_dataset = load_colmap_dataset(args.colmap_dir)

    # Convert and save
    convert_colmap_to_nerfstudio(colmap_dataset,args.colmap_dir, args.output_dir)

