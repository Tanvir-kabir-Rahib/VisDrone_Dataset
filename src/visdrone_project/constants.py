"""Shared class metadata for the VisDrone DET dataset."""

CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

HUMAN_CLASS_IDS = {0, 1}
CAR_CLASS_IDS = {3}
TARGET_CLASS_IDS = HUMAN_CLASS_IDS | CAR_CLASS_IDS

TARGET_LABELS = {
    0: "human",
    1: "human",
    3: "car",
}

DRAW_COLORS = {
    "human": (0, 166, 81),
    "car": (220, 80, 40),
    "other": (90, 120, 160),
}
