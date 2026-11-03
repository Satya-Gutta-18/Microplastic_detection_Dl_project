# Microplastic Detection using YOLOv8

AI-based detection, counting and size estimation of microplastics in water-sample images using YOLOv8.

## Project Structure
```
dataset/    images/ and labels/ split into train/val/test (YOLO format)
models/     trained weights (best.pt)
runs/       YOLOv8 training outputs
src/        train, detect, count, size estimation, report scripts
results/    sample_images/ and reports/
config/     data.yaml
```

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/train.py --epochs 100
python src/detect.py --source dataset/images/test
python src/count_particles.py --source dataset/images/test
python src/estimate_size.py --source dataset/images/test --um-per-pixel 2.5
python src/generate_report.py --source dataset/images/test --um-per-pixel 2.5
```
`--um-per-pixel` is the calibration of your microscope/camera setup (replace 2.5 with your value).

## Dataset
Place images in `dataset/images/{train,val,test}` and YOLO-format `.txt` labels in the matching `dataset/labels/` folder. Class: `microplastic`.

## License
MIT
