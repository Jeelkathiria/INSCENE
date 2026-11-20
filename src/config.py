"""
Configuration settings for AI-based In-Scene Advertising.
Project: A Score-Based System for Detecting Optimal Planar Advertisement Spaces in Video
"""

import os
from dataclasses import dataclass, field
from typing import List

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAMPLES_DIR = os.path.join(DATA_DIR, "samples")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

# Direct Ultralytics cache and settings to local workspace data folder to avoid permission errors & disk retries
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(DATA_DIR, ".ultralytics"))

# Create data directories if they don't exist
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, ".ultralytics"), exist_ok=True)


@dataclass
class ScoreWeights:
    """Weights for the multi-factor ScoreEngine."""
    duration: float = 0.25      # Screen-time / continuous visibility
    area: float = 0.20          # Surface scale relative to frame
    occlusion: float = 0.25     # Penalty for foreground occluders
    stability: float = 0.15     # Motion smoothness / low jitter
    geometry: float = 0.15      # Planarity, internal angles, rectangularity


@dataclass
class DetectorConfig:
    """Parameters for candidate surface and occluder detection."""
    model_name: str = "yolov8s-worldv2.pt"
    fallback_model: str = "yolov8n.pt"
    conf_threshold: float = 0.10        # Sensitive enough for distant/angled ad planes and street banners
    iou_threshold: float = 0.45
    inference_imgsz: int = 512          # Optimized balanced resolution for high throughput & sharp boundaries
    frame_stride: int = 5               # Process keyframe every 5th frame (~6 Hz at 30fps) with continuous track coasting

    # Open-vocabulary prompts for candidate planar surfaces & transit ad vehicles
    candidate_classes: List[str] = field(default_factory=lambda: [
        "billboard",
        "advertising board",
        "signboard",
        "poster",
        "display screen",
        "banner",
        "wall billboard",
        "building billboard",
        "large billboard",
        "digital billboard",
        "overhead banner",
        "hanging banner",
        "street banner",
        "facade banner",
        "plain wall",
        "upper wall",
        "wall without window",
        "brick wall",
        "gable",
        "side of bus",
        "exterior city bus",
        "advertisement poster",
        "framed poster",
        "transit ad poster",
        "train poster",
        "subway poster",
        "metro ad poster",
        "carriage poster",
        "interior billboard",
        "truck",
        "side of truck",
        "skyscraper facade",
        "high-rise facade",
        "building facade",
        "commercial building facade",
        "office building facade",
        "flat building wall",
        "building wall",
        "wall",
        "storefront",
        "shop sign",
        "blank wall",
        "concrete bridge wall",
        "overhead bridge girder",
        "bridge girder",
        "bridge wall",
        "whiteboard",
        # Hero handheld props & beverage containers for high-attention focal product placement
        "ceramic mug",
        "coffee mug",
        "drinking mug",
        "drinking cup",
        "coffee cup",
        "tea mug",
        "mug",
        "cup",
        "beverage can",
        "drink can",
        "soda can",
        "water bottle",
        "beer bottle",
        "bottle",
        "drinking glass",
        "tumbler"
    ])

    # Open-vocabulary prompts for foreground occluders
    occluder_classes: List[str] = field(default_factory=lambda: [
        "person",
        "pedestrian",
        "car",
        "automobile",
        "pole",
        "light pole",
        "metal pole",
        "utility pole",
        "vertical pole",
        "lamp post",
        "street lamp",
        "lamppost",
        "pillar",
        "column",
        "traffic sign",
        "tree branch",
        "scaffolding pole",
        "umbrella",
        "window",
        "handrail",
        "grab handle",
        "train seat"
    ])

    # Fallback standard COCO mappings when using base YOLO models
    coco_candidate_classes: List[str] = field(default_factory=lambda: [
        "tv", "laptop", "refrigerator", "book", "cup", "bottle", "wine glass"
    ])
    coco_occluder_classes: List[str] = field(default_factory=lambda: [
        "person", "car", "motorcycle", "bus", "truck", "traffic light", "stop sign", "chair"
    ])


@dataclass
class GeometryConfig:
    """Thresholds for 4-point quadrilateral verification."""
    min_area_fraction: float = 0.005    # Min candidate area (0.5% of total frame area)
    max_area_fraction: float = 0.75     # Max candidate area (75% of total frame area)
    min_internal_angle_deg: float = 30.0  # Reject hyper-acute perspective angles
    max_internal_angle_deg: float = 150.0
    epsilon_approx_ratio: float = 0.03  # Douglas-Peucker approximation factor


@dataclass
class TrackerConfig:
    """Tracking and tracklet management settings."""
    max_distance_threshold: float = 90.0  # Tightened pixel threshold to prevent jumping across adjacent signs
    max_lost_frames: int = 15             # Keep track alive for up to 15 frames (~0.5s) through dropouts
    max_coast_frames: int = 10            # Smoothly bridge brief dropouts without letting boxes float through space
    max_continuous_gap_sec: float = 1.5   # Split tracklet if an unobserved gap exceeds 1.5s
    min_tracklet_duration_sec: float = 0.5  # Filter out very short flashes (< 0.5 sec)
    target_optimal_duration_sec: float = 5.0 # Max score achieved around this duration
    max_reid_distance: float = 40.0       # Spatial radius (px) to re-identify and revive recently lost surfaces
    max_reid_frames: int = 45             # Memory window (~1.5s) to revive a temporarily lost tracklet


@dataclass
class SystemConfig:
    """Master configuration class."""
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    target_fps: int = 30

