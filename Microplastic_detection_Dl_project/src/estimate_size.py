"""Estimate particle size from bounding boxes using a pixel-to-micron scale."""
import argparse
from ultralytics import YOLO


def estimate_sizes(weights, source, um_per_pixel, conf=0.25):
    model = YOLO(weights)
    rows = []
    for r in model.predict(source=source, conf=conf, verbose=False):
        for i, (x1, y1, x2, y2) in enumerate(r.boxes.xyxy.tolist()):
            w_um = (x2 - x1) * um_per_pixel
            h_um = (y2 - y1) * um_per_pixel
            rows.append({"image": r.path, "particle": i,
                         "width_um": round(w_um, 2), "height_um": round(h_um, 2),
                         "max_dim_um": round(max(w_um, h_um), 2)})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--um-per-pixel", type=float, required=True,
                   help="calibration: micrometres per pixel of your imaging setup")
    a = p.parse_args()
    for row in estimate_sizes(a.weights, a.source, a.um_per_pixel):
        print(row)
