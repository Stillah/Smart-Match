import json
import os

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter
from sklearn.cluster import DBSCAN


def locate_table(binary_img: np.ndarray):
    """Find bounding box of the table using vertical/horizontal projections."""
    h, w = binary_img.shape
    hor_proj = np.sum(binary_img, axis=1) // 255
    hor_thresh = np.percentile(hor_proj[hor_proj > 0], 20) if np.any(hor_proj > 0) else 1
    rows_with_content = np.where(hor_proj > hor_thresh)[0]
    if len(rows_with_content) == 0:
        return 0, 0, w, h
    y1, y2 = rows_with_content[0], rows_with_content[-1]
    ver_proj = np.sum(binary_img[y1:y2, :], axis=0) // 255
    ver_thresh = np.percentile(ver_proj[ver_proj > 0], 20) if np.any(ver_proj > 0) else 1
    cols_with_content = np.where(ver_proj > ver_thresh)[0]
    if len(cols_with_content) == 0:
        x1, x2 = 0, w
    else:
        x1, x2 = cols_with_content[0], cols_with_content[-1]
    pad = 10
    x1 = max(0, x1 - pad)
    x2 = min(w, x2 + pad)
    y1 = max(0, y1 - pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def detect_vertical_lines(binary_img: np.ndarray, min_length_ratio: float = 0.8):
    """Detect long vertical lines using morphological operations and Hough transform."""
    h, _ = binary_img.shape
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(1, h // 10)))
    vertical_lines = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, vertical_kernel)
    edges = cv2.Canny(vertical_lines, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=int(min_length_ratio * h),
        maxLineGap=10,
    )
    x_coords = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if 80 <= abs(angle) <= 100:
                x_mid = (x1 + x2) // 2
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length >= min_length_ratio * h:
                    x_coords.append(x_mid)
    return x_coords


def merge_close_lines(x_coords: list, distance_threshold: int = 20):
    """Cluster x-coordinates that are too close and return the mean of each cluster."""
    if len(x_coords) == 0:
        return []
    x_coords = np.array(x_coords).reshape(-1, 1)
    clustering = DBSCAN(eps=distance_threshold, min_samples=1).fit(x_coords)
    merged = []
    for label in set(clustering.labels_):
        cluster_points = x_coords[clustering.labels_ == label]
        merged.append(int(np.mean(cluster_points)))
    return sorted(merged)


def vertical_projection_analysis(
    binary_img: np.ndarray,
    table_bbox: tuple,
    smoothing_window: int = 15,
    valley_prominence: int = 80,
):
    """Compute vertical projection within the table region, smooth it, and find valley positions."""
    x1, y1, x2, y2 = table_bbox
    table_region = binary_img[y1:y2, x1:x2]
    proj = np.sum(table_region, axis=0) / 255
    if len(proj) > smoothing_window:
        proj_smooth = savgol_filter(proj, smoothing_window, 3)
    else:
        proj_smooth = proj
    inverted = -proj_smooth
    valleys, _ = find_peaks(inverted, prominence=valley_prominence, distance=10)
    valley_positions = [x1 + v for v in valleys]
    return valley_positions, proj_smooth


def determine_column_boundaries(
    line_positions: list,
    valley_positions: list,
    table_bbox: tuple,
    min_column_width: int = 30,
):
    """Combine line positions and valley positions to get final column boundaries."""
    x1, _, x2, _ = table_bbox
    candidates = sorted(set(line_positions + valley_positions + [x1, x2]))
    filtered = []
    for pos in candidates:
        if not filtered or pos - filtered[-1] >= min_column_width:
            filtered.append(pos)
    return filtered


def detect_columns(image_path: str, output_dir: str = "./segments"):
    """
    Detect table columns in the given binarized grayscale image and split it into
    segments according to column boundaries.
    """
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")

    binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if binary is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    x1, y1, x2, y2 = locate_table(binary)
    table_bbox = (x1, y1, x2, y2)
    table_binary = binary[y1:y2, x1:x2].copy()

    line_x = detect_vertical_lines(table_binary, min_length_ratio=0.8)
    global_line_x = [x + x1 for x in line_x]
    merged_line_x = merge_close_lines(global_line_x, distance_threshold=20)
    valley_positions, _ = vertical_projection_analysis(binary, table_bbox)
    boundaries = determine_column_boundaries(merged_line_x, valley_positions, table_bbox)

    segment_infos = []
    for i in range(len(boundaries) - 1):
        bx_start = boundaries[i]
        bx_end = boundaries[i + 1]
        segment = binary[y1:y2, bx_start:bx_end]
        if segment.size == 0:
            continue
        segment_filename = f"segment_{len(segment_infos) + 1:03d}.png"
        segment_path = os.path.join(output_dir, segment_filename)
        cv2.imwrite(segment_path, segment)
        segment_infos.append(
            {
                "index": len(segment_infos) + 1,
                "filename": segment_filename,
                "path": segment_path,
                "bbox": {
                    "x1": int(bx_start),
                    "y1": int(y1),
                    "x2": int(bx_end),
                    "y2": int(y2),
                },
            }
        )

    manifest = [
        {
            "index": info["index"],
            "filename": info["filename"],
            "bbox": info["bbox"],
        }
        for info in segment_infos
    ]
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    return segment_infos


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python detection.py <image_path> [output_dir]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "./segments"
    infos = detect_columns(img_path, out_dir)
    print(f"Saved {len(infos)} segment(s) to {out_dir}")
    for info in infos:
        print(f"  {info['path']}")
