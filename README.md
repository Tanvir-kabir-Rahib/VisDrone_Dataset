# VisDrone Human and Car Detection

Computer vision pipeline for the Antlings AI/ML technical assessment. The GitHub repository intentionally contains only the files required to run the project in Google Colab plus this README. Dataset files, model weights, and generated outputs are downloaded or saved outside GitHub.

## What Is Included

- Dataset analysis and sample visualizations: `scripts/analyze_dataset.py`
- YOLO fine-tuning entry point: `scripts/train_yolo.py`
- Human/car detection and counting: `scripts/detect_count.py`
- Model evaluation wrapper: `scripts/evaluate_yolo.py`
- Colab training workflow: `notebooks/VisDrone_Colab_Training.ipynb`

## What Not To Upload To GitHub

These are ignored because Colab downloads or creates them:

- `VisDrone2019-DET-*`
- `data/`
- `outputs/`
- `runs/`
- `*.pt`
- `kaggle.json`

For final submission, keep the GitHub repo for code and documentation. Put trained weights, prediction images, evaluation outputs, and the demo asset/video in Google Drive.

## Dataset

This workspace expects the VisDrone folders at the repository root:

```text
VisDrone2019-DET-train/
VisDrone2019-DET-val/
VisDrone2019-DET-test-dev/
VisDrone2019-DET-test-challenge/
```

Human count = `pedestrian` + `people`. Car count = `car`.

Do not upload the VisDrone dataset to GitHub. Download it from Kaggle when you run locally, in Jupyter, or in Colab:

```bash
python3 -m pip install kaggle
kaggle datasets download -d banuprasadb/visdrone-dataset -p data --unzip
```

Then pass the extracted dataset root to scripts with `--dataset-root`. The dataset root is the folder that contains `VisDrone2019-DET-train`.

## Setup

```bash
python3 -m pip install -r requirements.txt
```

The local machine used to prepare this project already had Pillow, NumPy, Matplotlib, and PyYAML, but not PyTorch/Ultralytics. Dataset analysis and label-based visual demos run without the training stack.

## Run on Google Colab

For training, use Colab/GPU instead of local CPU. Open `notebooks/VisDrone_Colab_Training.ipynb`, set the runtime to GPU, upload `kaggle.json` when prompted, and run the notebook top to bottom.

The notebook handles:

- GitHub clone
- Kaggle dataset download
- Google Drive output saving
- GPU check
- YOLO training with `--device 0`
- Evaluation and prediction visualization

Colab outputs are saved under:

```text
/content/drive/MyDrive/visdrone_outputs/
```

## Run Dataset Analysis

```bash
python3 scripts/analyze_dataset.py --dataset-root .
```

Outputs:

- `outputs/dataset/dataset_summary.json`
- `outputs/dataset/class_distribution.png`
- `outputs/dataset/target_distribution.png`
- `outputs/dataset/image_size_spread.png`
- `outputs/dataset/sample_grid.jpg`

## Train YOLO

```bash
python3 scripts/train_yolo.py --model yolo11n.pt --epochs 50 --imgsz 640 --batch 16
```

Fast Colab baseline:

```bash
python3 scripts/train_yolo.py \
  --dataset-root /content/data/VisDrone_Dataset \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --cache disk \
  --freeze 10
```

For a stronger but slower run on a GPU:

```bash
python3 scripts/train_yolo.py --model yolo11s.pt --epochs 80 --imgsz 960 --batch 8 --device 0 --workers 8 --cache disk
```

The script trains all 10 VisDrone classes and filters human/car classes during inference.

## Detect and Count

With trained weights:

```bash
python3 scripts/detect_count.py \
  --weights outputs/training/visdrone-yolo/weights/best.pt \
  --split val \
  --limit 12
```

Label-based local demo when weights are not available:

```bash
python3 scripts/detect_count.py --use-labels --split val --limit 8
```

Outputs are saved under `outputs/predictions/`, including processed images and `count_summary.csv`.

## Evaluate

```bash
python3 scripts/evaluate_yolo.py \
  --weights outputs/training/visdrone-yolo/weights/best.pt \
  --split val
```

## Demo Asset

```bash
python3 scripts/make_demo_slideshow.py
```

This creates `outputs/demo/visdrone_demo.gif` from the generated charts and counting outputs.

## Notes

The generated label-demo images are marked as ground-truth demos. They verify parsing, visualization, and counting logic, but they are not model predictions. After training, rerun `scripts/detect_count.py` with `--weights` to produce prediction outputs for the final submission/demo video.
