# VisDrone Human and Car Detection Assessment Report

## Task-01: Dataset Understanding and Preprocessing

The workspace contains the VisDrone2019-DET dataset in YOLO format:

- `VisDrone2019-DET-train`: 6,471 images and labels
- `VisDrone2019-DET-val`: 548 images and labels
- `VisDrone2019-DET-test-dev`: 1,610 images and labels
- `VisDrone2019-DET-test-challenge`: 1,580 images without public labels

The original classes are `pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, and `motor`. For the assessment, `pedestrian` and `people` are counted as humans, while `car` is counted as car.

Observed annotation counts:

| Split | Total boxes | Human boxes | Car boxes |
| --- | ---: | ---: | ---: |
| train | 343,205 | 106,396 | 144,867 |
| val | 38,759 | 13,969 | 14,064 |
| test-dev | 75,102 | 27,382 | 28,074 |

Preprocessing and augmentation strategy:

- Keep annotations in normalized YOLO format.
- Resize during training with multi-scale YOLO augmentation.
- Use mosaic, HSV jitter, horizontal flip, and random scale/translation to handle altitude, viewpoint, lighting, and density variation.
- Avoid aggressive cropping because small aerial objects can disappear easily.
- Use a larger input size such as `960` or `1280` because many humans are tiny in drone imagery.

Challenges noticed:

- Small objects dominate, especially pedestrians.
- Occlusion and crowd density make human counting noisy.
- Aerial perspective creates large scale variation within a single image.
- Similar vehicle categories such as `car`, `van`, `truck`, and `bus` can be confused.
- Test challenge images do not include public labels, so test-dev/val are used for local evaluation.

Generated visual outputs:

- `outputs/dataset/class_distribution.png`
- `outputs/dataset/target_distribution.png`
- `outputs/dataset/image_size_spread.png`
- `outputs/dataset/sample_grid.jpg`

## Task-02: Model Training

The training script fine-tunes an Ultralytics YOLO model on all 10 VisDrone classes:

```bash
python3 -m pip install -r requirements.txt
python3 scripts/train_yolo.py --model yolo11n.pt --epochs 50 --imgsz 960 --batch 8
```

Training all VisDrone classes is useful because vehicle classes share visual context, while the inference script filters the assessment targets. The human count is computed from `pedestrian` and `people`; car detections use class `car`.

Recommended stronger runs:

```bash
python3 scripts/train_yolo.py --model yolo11s.pt --epochs 80 --imgsz 1280 --batch 4
python3 scripts/train_yolo.py --model yolo11m.pt --epochs 100 --imgsz 1280 --batch 2
```

## Task-03: Detection and Counting

The detection/counting script supports two modes:

- YOLO prediction mode with trained weights.
- Ground-truth label demo mode when weights are unavailable locally.

Prediction mode:

```bash
python3 scripts/detect_count.py --weights outputs/training/visdrone-yolo/weights/best.pt --split val --limit 12
```

Local label-demo mode:

```bash
python3 scripts/detect_count.py --use-labels --split val --limit 8
```

Each output image displays bounding boxes and a top banner with total humans and cars. Counts are saved to:

- `outputs/predictions/count_summary.csv`
- `outputs/predictions/count_summary.json`

## Task-04: Optional Tracking

Tracking is wired through Ultralytics trackers:

```bash
python3 scripts/track_yolo.py \
  --weights outputs/training/visdrone-yolo/weights/best.pt \
  --source path/to/video.mp4 \
  --tracker bytetrack.yaml
```

ByteTrack is a good default because it is simple and performs well for detector-first pipelines.

## Task-05: Evaluation and Visualization

Evaluation with trained weights:

```bash
python3 scripts/evaluate_yolo.py --weights outputs/training/visdrone-yolo/weights/best.pt --split val
```

Expected metrics to report after training:

- Precision
- Recall
- mAP50
- mAP50-95
- Inference speed/FPS

Strengths:

- Uses the dataset's native YOLO format.
- Cleanly separates dataset analysis, training, inference, counting, evaluation, and tracking.
- Counts humans from both VisDrone human categories.
- Keeps outputs reproducible under `outputs/`.

Limitations:

- This machine does not currently have PyTorch/Ultralytics installed, so local training was not executed here.
- Label-demo outputs are visual sanity checks, not model predictions.
- True video tracking requires an input video or ordered frame sequence.

## Demo Video Outline

1. Show dataset structure and generated dataset visualizations.
2. Explain class mapping: `pedestrian` + `people` = humans, `car` = car.
3. Show training command and training output folder.
4. Run detection/counting on validation images.
5. Show count summary CSV/JSON and several processed images.
6. Optionally show tracker command/output if a video is available.
