# Demo Video Storyboard

Use this structure for the required 3-5 minute demo video.

1. Show the repository and dataset folders.
2. Open `outputs/dataset/dataset_summary.json` and the generated plots.
3. Explain the class mapping: `pedestrian` and `people` are humans; `car` is car.
4. Run or show the training command in `scripts/train_yolo.py`.
5. Run detection/counting with trained weights, or show the label-demo fallback if weights are unavailable.
6. Open `outputs/predictions/count_summary.csv`.
7. Show several processed images with count banners.
8. Mention optional tracking via `scripts/track_yolo.py`.

Prepared visual asset:

```bash
python3 scripts/make_demo_slideshow.py
```

This creates `outputs/demo/visdrone_demo.gif`, which can be used as visual support while recording narration.
