"""Run inference with a trained model on an image or folder."""
import argparse
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="models/best.pt")
    p.add_argument("--source", required=True, help="image, folder or video")
    p.add_argument("--conf", type=float, default=0.25)
    a = p.parse_args()

    model = YOLO(a.weights)
    model.predict(source=a.source, conf=a.conf, save=True, save_txt=True,
                  project="results", name="sample_images", exist_ok=True)


if __name__ == "__main__":
    main()
