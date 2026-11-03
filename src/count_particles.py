"""Count detected microplastic particles per image."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def count_particles(weights, source, conf=0.25):
    model = YOLO(weights)
    counts = {}
    for r in model.predict(source=source, conf=conf, verbose=False):
        counts[Path(r.path).name] = len(r.boxes)
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--source", required=True)
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()
    for name, n in count_particles(a.weights, a.source, a.conf).items():
        print(f"{name}: {n} particles")
