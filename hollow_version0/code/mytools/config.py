"""Configuration for the packaged hollow-powder inference app.

All paths are relative to the package root:

    /root/code/empty_heart/ans

The command-line entrypoint can still override the model paths and thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ANS_ROOT = Path(__file__).resolve().parents[2]

PARTICLE_MODEL = str(ANS_ROOT / "model" / "particle" / "particle.pt")
HOLLOW_MODEL = str(ANS_ROOT / "model" / "hollow" / "hollow.pt")

EVAL_PARTICLE_SCRIPT = str(ANS_ROOT / "code" / "eval_particle" / "infer_full_plus_tiles.py")
HOLLOW_FILTER_SCRIPT = str(ANS_ROOT / "code" / "hollow_pipeline" / "filter_black_ratio.py")


@dataclass(frozen=True)
class InferenceConfig:
    model_min_size: int
    model_max_size: int
    anchor_sizes: str
    detections_per_img: int
    full_score_threshold: float
    tile_score_threshold: float
    mask_threshold: float = 0.5
    tile_cols: int = 5
    tile_rows: int = 4
    tile_w: int = 768
    tile_h: int = 576
    image_boundary_margin: int = 4
    tile_boundary_margin: int = 4
    tile_max_intersection_ratio: float = 0.05
    tile_max_iou: float = 0.2
    tile_model_min_size: int | None = None
    tile_model_max_size: int | None = None


PARTICLE_INFERENCE = InferenceConfig(
    model_min_size=512,
    model_max_size=1024,
    anchor_sizes="8,16,32,64,128",
    detections_per_img=900,
    full_score_threshold=0.5,
    tile_score_threshold=0.2,
    tile_cols=5,
    tile_rows=4,
    tile_w=640,
    tile_h=480,
    image_boundary_margin=2,
)

HOLLOW_INFERENCE = InferenceConfig(
    model_min_size=1536,
    model_max_size=2048,
    anchor_sizes="16,32,64,128,256",
    detections_per_img=700,
    full_score_threshold=0.2,
    tile_score_threshold=0.3,
    tile_cols=5,
    tile_rows=4,
    tile_w=768,
    tile_h=576,
)

HOLLOW_RATIO_THRESHOLD = 0.25
