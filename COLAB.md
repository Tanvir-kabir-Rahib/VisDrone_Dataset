# Google Colab Training Guide

Use the notebook at `notebooks/VisDrone_Colab_Training.ipynb` for the easiest Colab workflow.

This is the recommended workflow for submission: GitHub stores the project code, while Google Drive stores large training outputs, predictions, weights, and demo assets.

## Before You Start

1. Push this project to a public or private GitHub repository.
2. Open Google Colab.
3. Set runtime to GPU: `Runtime > Change runtime type > T4 GPU`.
4. Open the notebook and set `REPO_URL` to your GitHub repository URL.
5. Upload `kaggle.json` when the notebook asks for it.

Do not upload the VisDrone dataset folders to GitHub. The notebook downloads the dataset directly from Kaggle using:

```bash
kaggle datasets download -d banuprasadb/visdrone-dataset -p /content/data --unzip
```

The notebook saves generated files to:

```text
/content/drive/MyDrive/visdrone_outputs/
```

## Recommended Colab Settings

For the fastest useful Colab baseline, start with:

```bash
--model yolo11n.pt --epochs 50 --imgsz 640 --batch 16 --device 0 --workers 8 --cache disk --freeze 10
```

For better accuracy after the fast baseline works, try:

```bash
--model yolo11s.pt --epochs 80 --imgsz 960 --batch 8 --device 0 --workers 8 --cache disk
```

For a stronger GPU, try the larger image preset:

```bash
--model yolo11s.pt --epochs 80 --imgsz 1280 --batch 4 --device 0 --workers 8 --cache disk
```

In the training table, confirm that `GPU_mem` is not `0G`. If it shows `0G`, Colab is not using a GPU.

## Manual Colab Commands

After cloning the repo and downloading the dataset, replace `/content/data/VisDrone_Dataset`
with the dataset root printed by the notebook if Kaggle extracts it somewhere else. Then run:

```bash
python scripts/train_yolo.py \
  --dataset-root /content/data/VisDrone_Dataset \
  --model yolo11n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --cache disk \
  --freeze 10 \
  --project /content/drive/MyDrive/visdrone_outputs/training \
  --name visdrone-yolo11n-640-fast
```

Then evaluate and create detection/counting outputs:

```bash
python scripts/evaluate_yolo.py \
  --dataset-root /content/data/VisDrone_Dataset \
  --weights /content/drive/MyDrive/visdrone_outputs/training/visdrone-yolo11s-960/weights/best.pt \
  --split val \
  --imgsz 960

python scripts/detect_count.py \
  --dataset-root /content/data/VisDrone_Dataset \
  --weights /content/drive/MyDrive/visdrone_outputs/training/visdrone-yolo11s-960/weights/best.pt \
  --split val \
  --limit 12 \
  --output-dir /content/drive/MyDrive/visdrone_outputs/predictions
```
