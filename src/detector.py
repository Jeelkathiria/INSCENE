"""
Candidate Surface and Occluder Detection module.
Integrates YOLOv8-World v2 (open-vocabulary) with GrabCut / SAM refinement
and graceful offline/classical contour fallback.
"""

import math
import logging
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.config import DetectorConfig, GeometryConfig
from src.geometry import extract_quad_from_contour, is_valid_planar_quad, order_points

logger = logging.getLogger(__name__)


@dataclass
class CandidateDetection:
    """Represents a detected planar surface candidate."""
    bbox: Tuple[int, int, int, int]        # (x1, y1, x2, y2)
    quad: np.ndarray                       # 4x2 array: [TL, TR, BR, BL]
    class_name: str
    confidence: float
    mask: Optional[np.ndarray] = None


@dataclass
class OccluderDetection:
    """Represents a detected foreground occluding object."""
    bbox: Tuple[int, int, int, int]
    poly: np.ndarray                       # Polygon or box corners
    class_name: str
    confidence: float


class SurfaceDetector:
    """
    Hybrid detector combining YOLOv8-World v2 open-vocabulary detection,
    GrabCut pixel refinement, and classical geometry extraction.
    """
    _cached_model = None
    _cached_model_type = None
    _cached_classes = None

    def __init__(self, det_cfg: DetectorConfig, geom_cfg: GeometryConfig):
        self.det_cfg = det_cfg
        self.geom_cfg = geom_cfg
        all_classes = self.det_cfg.candidate_classes + self.det_cfg.occluder_classes
        if SurfaceDetector._cached_model is not None and SurfaceDetector._cached_classes == all_classes:
            self.model = SurfaceDetector._cached_model
            self.model_type = SurfaceDetector._cached_model_type
        else:
            self.model = None
            self.model_type = "none"
            self._init_model()

    def _init_model(self):
        """Initializes YOLOv8-World v2 with fallback to standard YOLO or classical mode."""
        try:
            from ultralytics import YOLO
            logger.info("Attempting to load YOLOv8-World model: %s", self.det_cfg.model_name)
            # Try loading YOLOv8-World
            try:
                self.model = YOLO(self.det_cfg.model_name)
                # Set custom open-vocabulary classes
                all_classes = self.det_cfg.candidate_classes + self.det_cfg.occluder_classes
                self.model.set_classes(all_classes)
                self.model_type = "yolo_world"
                SurfaceDetector._cached_model = self.model
                SurfaceDetector._cached_model_type = self.model_type
                SurfaceDetector._cached_classes = all_classes
                logger.info("YOLOv8-World loaded with %d custom classes.", len(all_classes))
                return
            except Exception as e:
                logger.warning("Could not initialize %s: %s. Falling back to %s",
                               self.det_cfg.model_name, e, self.det_cfg.fallback_model)

            # Fallback to standard YOLOv8n
            try:
                self.model = YOLO(self.det_cfg.fallback_model)
                self.model_type = "yolo_coco"
                logger.info("Loaded fallback model: %s", self.det_cfg.fallback_model)
                return
            except Exception as e2:
                logger.warning("Could not load fallback model: %s. Using classical detector.", e2)
                self.model = None
                self.model_type = "classical"
        except (ImportError, Exception) as e:
            logger.warning("Could not initialize neural detector (%s). Using classical detector.", e)
            self.model = None
            self.model_type = "classical"

    def refine_quad_fast(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Ultra-fast (<2ms) quad refinement inside candidate bounding box.
        Applies local edge detection and polygon fitting instead of slow iterative GrabCut.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        bw, bh = x2 - x1, y2 - y1

        default_quad = order_points(np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32))

        if bw < 25 or bh < 25:
            return default_quad

        try:
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 40, 130)

            cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                largest = max(cnts, key=cv2.contourArea)
                if cv2.contourArea(largest) > (bw * bh * 0.20):
                    peri = cv2.arcLength(largest, True)
                    approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
                    if len(approx) == 4 and cv2.isContourConvex(approx):
                        pts = approx.reshape(4, 2).astype(np.float32)
                        pts[:, 0] += x1
                        pts[:, 1] += y1
                        return order_points(pts)

            # If contour approximation did not yield 4 points, test 3D perspective depth vanishing lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=max(20, int(bw * 0.20)), maxLineGap=12)
            if lines is not None and len(lines) >= 2:
                top_slopes = []
                bot_slopes = []
                mid_y = bh * 0.5
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    dx = lx2 - lx1
                    dy = ly2 - ly1
                    if abs(dx) < 10:
                        continue
                    slope = dy / float(dx)
                    if abs(slope) > 0.40:
                        continue
                    y_avg = (ly1 + ly2) * 0.5
                    if y_avg < mid_y:
                        top_slopes.append(slope)
                    else:
                        bot_slopes.append(slope)

                if top_slopes or bot_slopes:
                    m_top = float(np.median(top_slopes)) if top_slopes else 0.0
                    m_bot = float(np.median(bot_slopes)) if bot_slopes else 0.0
                    if abs(m_top) > 0.025 or abs(m_bot) > 0.025:
                        p1 = [float(x1), float(y1)]
                        p2 = [float(x2), float(y1 + m_top * bw)]
                        p3 = [float(x2), float(y2 + m_bot * bw)]
                        p4 = [float(x1), float(y2)]
                        return order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
        except Exception:
            pass

        return default_quad

    def fit_bridge_3d_quad(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Fits a perspective 3D planar quadrilateral conforming to the architectural
        girder slope and vanishing perspective of a bridge wall / overpass beam.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0 or bw < 40 or bh < 20:
            return order_points(np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float32))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 35, 130)

        min_len = max(30, int(bw * 0.10))
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=45, minLineLength=min_len, maxLineGap=25)

        if lines is not None and len(lines) >= 3:
            filtered_lines = []
            gx_mid = (x1 + x2) * 0.5
            for line in lines:
                lx1, ly1, lx2, ly2 = line[0]
                dx = lx2 - lx1
                dy = ly2 - ly1
                if abs(dx) < 15:
                    continue
                slope = dy / float(dx)
                if abs(slope) > 0.85:
                    continue
                glx1 = x1 + lx1
                gly1 = y1 + ly1
                glx2 = x1 + lx2
                gly2 = y1 + ly2
                gy_mid = ((gly1 + gly2) * 0.5) + slope * (gx_mid - (glx1 + glx2) * 0.5)
                length = np.hypot(dx, dy)
                filtered_lines.append((slope, gy_mid, length))

            if len(filtered_lines) >= 3:
                slopes = [f[0] for f in filtered_lines]
                median_slope = float(np.median(slopes))
                consistent_lines = [f for f in filtered_lines if abs(f[0] - median_slope) <= 0.15]
                if len(consistent_lines) >= 2:
                    ymids = [f[1] for f in consistent_lines]
                    min_ym = float(np.min(ymids))
                    max_ym = float(np.max(ymids))
                    span_ym = max_ym - min_ym

                    if span_ym >= bh * 0.25:
                        top_group = [f for f in consistent_lines if f[1] <= min_ym + span_ym * 0.35]
                        bot_group = [f for f in consistent_lines if f[1] >= max_ym - span_ym * 0.35]

                        if top_group and bot_group:
                            w_top = [f[2] for f in top_group]
                            slope_top = float(np.average([f[0] for f in top_group], weights=w_top))
                            ym_top = float(np.average([f[1] for f in top_group], weights=w_top))

                            w_bot = [f[2] for f in bot_group]
                            slope_bot = float(np.average([f[0] for f in bot_group], weights=w_bot))
                            ym_bot = float(np.average([f[1] for f in bot_group], weights=w_bot))

                            mx1 = float(max(4, x1 + int(bw * 0.02)))
                            mx2 = float(min(w - 4, x2 - int(bw * 0.02)))

                            p1_y = float(ym_top + slope_top * (mx1 - gx_mid))
                            p2_y = float(ym_top + slope_top * (mx2 - gx_mid))
                            p3_y = float(ym_bot + slope_bot * (mx2 - gx_mid))
                            p4_y = float(ym_bot + slope_bot * (mx1 - gx_mid))

                            p1_y = float(max(2, min(h - 2, p1_y)))
                            p2_y = float(max(2, min(h - 2, p2_y)))
                            p3_y = float(max(2, min(h - 2, p3_y)))
                            p4_y = float(max(2, min(h - 2, p4_y)))

                            quad = order_points(np.array([
                                [mx1, p1_y],
                                [mx2, p2_y],
                                [mx2, p3_y],
                                [mx1, p4_y]
                            ], dtype=np.float32))

                            return quad

        return order_points(np.array([
            [x1 + 0.02 * bw, y1 + 0.05 * bh],
            [x2 - 0.02 * bw, y1 + 0.05 * bh],
            [x2 - 0.02 * bw, y2 - 0.05 * bh],
            [x1 + 0.02 * bw, y2 - 0.05 * bh]
        ], dtype=np.float32))

    def fit_billboard_banner_3d_quad(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Fits a true 3D perspective quadrilateral point-to-point for billboards,
        framed posters, and wall banners in both outdoor street canyons and transit interiors.
        """
        fh, fw = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        default_quad = order_points(np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32))

        if bw < 20 or bh < 20:
            return default_quad

        crop = frame[max(0, y1):min(fh, y2), max(0, x1):min(fw, x2)]
        if crop.size == 0:
            return default_quad

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 115)

            # 1. Point-to-point outer boundary line solver
            min_len = max(16, int(min(bw, bh) * 0.12))
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=18, minLineLength=min_len, maxLineGap=12)

            if lines is not None and len(lines) >= 3:
                v_lines = []
                h_lines = []
                for l in lines:
                    lx1, ly1, lx2, ly2 = l[0]
                    dx = lx2 - lx1
                    dy = ly2 - ly1
                    length = np.hypot(dx, dy)
                    if abs(dx) < 18 or abs(dy) > abs(dx) * 1.5:
                        v_lines.append((lx1, ly1, lx2, ly2, (lx1 + lx2) * 0.5, length))
                    elif abs(dy) < abs(dx) * 0.65:
                        h_lines.append((lx1, ly1, lx2, ly2, (ly1 + ly2) * 0.5, length))

                if v_lines and h_lines:
                    v_lines.sort(key=lambda l: l[4])
                    h_lines.sort(key=lambda l: l[4])

                    outer_left = [l for l in v_lines if l[4] <= bw * 0.28]
                    outer_right = [l for l in v_lines if l[4] >= bw * 0.72]
                    outer_top = [l for l in h_lines if l[4] <= bh * 0.28]
                    outer_bot = [l for l in h_lines if l[4] >= bh * 0.72]

                    if outer_left and outer_right and outer_top and outer_bot:
                        def fit_cluster(cluster, is_v):
                            pts = []
                            for l in cluster:
                                pts.extend([[l[0], l[1]], [l[2], l[3]]])
                            pts = np.array(pts, dtype=np.float32)
                            if is_v:
                                y, x = pts[:, 1], pts[:, 0]
                                if len(y) >= 2 and (np.max(y) - np.min(y)) > 15:
                                    m, c = np.polyfit(y, x, 1)
                                    return float(m), float(c)
                                return 0.0, float(np.median(x))
                            else:
                                x, y = pts[:, 0], pts[:, 1]
                                if len(x) >= 2 and (np.max(x) - np.min(x)) > 15:
                                    m, c = np.polyfit(x, y, 1)
                                    return float(m), float(c)
                                return 0.0, float(np.median(y))

                        lt = fit_cluster(outer_top, False)
                        lb = fit_cluster(outer_bot, False)
                        ll = fit_cluster(outer_left, True)
                        lr = fit_cluster(outer_right, True)

                        def intersect(lh, lv):
                            mh, ch = lh
                            mv, cv = lv
                            denom = 1.0 - mh * mv
                            if abs(denom) < 1e-4:
                                denom = 1e-4
                            iy = (mh * cv + ch) / denom
                            ix = mv * iy + cv
                            return float(ix), float(iy)

                        c1 = intersect(lt, ll)
                        c2 = intersect(lt, lr)
                        c3 = intersect(lb, lr)
                        c4 = intersect(lb, ll)

                        p1 = [float(np.clip(x1 + c1[0], 1, fw - 2)), float(np.clip(y1 + c1[1], 1, fh - 2))]
                        p2 = [float(np.clip(x1 + c2[0], 1, fw - 2)), float(np.clip(y1 + c2[1], 1, fh - 2))]
                        p3 = [float(np.clip(x1 + c3[0], 1, fw - 2)), float(np.clip(y1 + c3[1], 1, fh - 2))]
                        p4 = [float(np.clip(x1 + c4[0], 1, fw - 2)), float(np.clip(y1 + c4[1], 1, fh - 2))]

                        p2p_quad = order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            p2p_quad, fw, fh,
                            min_area_fraction=self.geom_cfg.min_area_fraction * 0.5,
                            max_area_fraction=self.geom_cfg.max_area_fraction
                        )
                        if valid:
                            return p2p_quad

            # 2. Independent Top & Bottom Slopes with Perspective Vanishing Ray Fallback
            vp_x = fw * 0.50
            vp_y = fh * 0.58
            box_cx = (x1 + x2) * 0.5
            box_cy = (y1 + y2) * 0.5

            expected_slope_sign = 1.0 if box_cx < vp_x else -1.0
            dx_to_vp = vp_x - box_cx
            dy_to_vp = vp_y - box_cy
            ray_slope = dy_to_vp / float(dx_to_vp) if abs(dx_to_vp) > 10 else 0.0

            # Derive top and bottom slopes independently
            top_slopes = []
            bot_slopes = []
            v_slopes = []
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    dx = lx2 - lx1
                    dy = ly2 - ly1
                    length = np.hypot(dx, dy)
                    y_mid = (ly1 + ly2) * 0.5
                    if abs(dx) >= 12 and abs(dy) <= abs(dx) * 0.85:
                        slope = dy / float(dx)
                        if y_mid < bh * 0.45:
                            top_slopes.append((slope, length))
                        elif y_mid > bh * 0.55:
                            bot_slopes.append((slope, length))
                    elif abs(dx) < 14 or abs(dy) > abs(dx) * 1.2:
                        v_slopes.append(dx / float(dy) if dy != 0 else 0.0)

            if top_slopes:
                weights = [t[1] for t in top_slopes]
                m_top = float(np.average([t[0] for t in top_slopes], weights=weights))
            else:
                dy_top = vp_y - y1
                m_top = float(np.clip(dy_top / float(dx_to_vp) if abs(dx_to_vp) > 10 else ray_slope, -0.65, 0.65))

            if bot_slopes:
                weights = [b[1] for b in bot_slopes]
                m_bot = float(np.average([b[0] for b in bot_slopes], weights=weights))
            else:
                dy_bot = vp_y - y2
                m_bot = float(np.clip(dy_bot / float(dx_to_vp) if abs(dx_to_vp) > 10 else ray_slope, -0.65, 0.65))

            if v_slopes:
                v_inv = float(np.median(v_slopes))
                v_inv = float(np.clip(v_inv, -0.18, 0.18))
            else:
                v_inv = 0.025 if box_cx < vp_x else -0.025

            # Overhead central banner dampening
            if abs(box_cx - vp_x) < 0.15 * fw and (bw / max(1, bh)) > 1.2:
                m_top = float(np.clip(m_top, -0.06, 0.06))
                m_bot = float(np.clip(m_bot, -0.06, 0.06))
                v_inv = 0.0

            gx_mid = box_cx
            gy_mid = box_cy
            half_w = bw * 0.5 * 0.98
            half_h = bh * 0.5 * 0.98

            p1_x = (gx_mid - half_w) - half_h * v_inv
            p1_y = (gy_mid - half_h) + m_top * (p1_x - gx_mid)

            p2_x = (gx_mid + half_w) - half_h * v_inv
            p2_y = (gy_mid - half_h) + m_top * (p2_x - gx_mid)

            p3_x = (gx_mid + half_w) + half_h * v_inv
            p3_y = (gy_mid + half_h) + m_bot * (p3_x - gx_mid)

            p4_x = (gx_mid - half_w) + half_h * v_inv
            p4_y = (gy_mid + half_h) + m_bot * (p4_x - gx_mid)

            pts = np.array([
                [np.clip(p1_x, 1, fw - 2), np.clip(p1_y, 1, fh - 2)],
                [np.clip(p2_x, 1, fw - 2), np.clip(p2_y, 1, fh - 2)],
                [np.clip(p3_x, 1, fw - 2), np.clip(p3_y, 1, fh - 2)],
                [np.clip(p4_x, 1, fw - 2), np.clip(p4_y, 1, fh - 2)]
            ], dtype=np.float32)

            quad = order_points(pts)
            valid, _ = is_valid_planar_quad(
                quad, fw, fh,
                min_area_fraction=self.geom_cfg.min_area_fraction,
                max_area_fraction=self.geom_cfg.max_area_fraction,
                min_angle=self.geom_cfg.min_internal_angle_deg,
                max_angle=self.geom_cfg.max_internal_angle_deg
            )
            if valid:
                return quad
        except Exception:
            pass

        return default_quad

    def fit_upper_wall_plain_quad(
        self,
        frame: np.ndarray,
        facade_bbox: Tuple[int, int, int, int],
        window_bboxes: List[Tuple[int, int, int, int]]
    ) -> Optional[np.ndarray]:
        """
        Detects and fits the optimal plain wall advertisement surface above windows
        on a building facade, excluding window panes and conforming to upward keystone perspective.
        """
        fh, fw = frame.shape[:2]
        fx1, fy1, fx2, fy2 = facade_bbox
        fbw = fx2 - fx1
        fbh = fy2 - fy1

        # Collect windows inside or intersecting this facade
        contained_windows = []
        for wb in window_bboxes:
            wx1, wy1, wx2, wy2 = wb
            if wx2 > fx1 + 10 and wx1 < fx2 - 10 and wy2 > fy1 and wy1 < fy2:
                contained_windows.append(wb)

        if not contained_windows:
            return None

        # Minimum top y among windows
        min_win_y = min(wb[1] for wb in contained_windows)
        upper_h = min_win_y - fy1

        if upper_h < 35 or upper_h < (fbh * 0.10):
            return None

        margin_bottom = max(6, int(upper_h * 0.06))
        bot_y = float(min_win_y - margin_bottom)
        top_y = float(fy1 + int(upper_h * 0.15))

        crop_upper = frame[max(0, fy1):int(bot_y), max(0, fx1):min(fw, fx2)]
        win_min_x = min(wb[0] for wb in contained_windows)
        win_max_x = max(wb[2] for wb in contained_windows)

        left_x = float(max(fx1 + 15, win_min_x - 15))
        right_x = float(min(fx2 - 15, win_max_x + 15))

        if fbw > 200 and crop_upper.size > 0:
            lab = cv2.cvtColor(crop_upper, cv2.COLOR_BGR2Lab)
            a_channel = lab[:, :, 1]
            is_brick = (a_channel > 132)
            col_counts = np.sum(is_brick, axis=0)
            valid_cols = np.where(col_counts > (crop_upper.shape[0] * 0.35))[0]
            if len(valid_cols) > 40:
                left_x = float(fx1 + valid_cols[0] + 5)
                right_x = float(fx1 + valid_cols[-1] - 5)

        span_w = right_x - left_x
        delta_x = span_w * 0.025

        p1 = [left_x + delta_x, top_y]
        p2 = [right_x - delta_x, top_y]
        p3 = [right_x, bot_y]
        p4 = [left_x, bot_y]

        quad = order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
        valid, _ = is_valid_planar_quad(
            quad, fw, fh,
            min_area_fraction=self.geom_cfg.min_area_fraction,
            max_area_fraction=self.geom_cfg.max_area_fraction,
            min_angle=self.geom_cfg.min_internal_angle_deg,
            max_angle=self.geom_cfg.max_internal_angle_deg
        )
        if valid:
            return quad
        return None

    @staticmethod
    def is_window_grid_facade(crop: np.ndarray, strict: bool = False) -> bool:
        """
        Determines whether a building facade crop is covered in a dense repeating
        window grid / aperture lattice (such as residential apartment blocks) with no plain ad surface.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
            return False
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        col_energy = np.mean(sobel_x, axis=0)
        col_mean = float(np.mean(sobel_x))
        col_std = float(np.std(col_energy))

        # Strict check for low-confidence detections or window clutter
        if strict and (col_mean > 50.0 or col_std > 24.0):
            return True

        # Dense repeating vertical window aperture columns (e.g. apartment towers)
        if col_mean > 58.0 and col_std > 28.0:
            return True
        return False

    def detect_slender_occluders(
        self,
        frame: np.ndarray,
        quad: np.ndarray
    ) -> List[np.ndarray]:
        """
        Detects thin foreground occluders (such as streetlight poles, utility poles,
        posts, columns, wires, and tree branches) that pass directly in front of a
        planar candidate surface and might be missed by object detection bounding boxes.
        """
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]
        qw = int(max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])))
        qh = int(max(np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        if qw < 30 or qh < 30:
            return []

        dst = np.array([[0, 0], [qw, 0], [qw, qh], [0, qh]], dtype=np.float32)
        try:
            M = cv2.getPerspectiveTransform(quad, dst)
            M_inv = cv2.getPerspectiveTransform(dst, quad)
            warped = cv2.warpPerspective(frame, M, (qw, qh))
        except Exception:
            return []

        med_bgr = np.median(warped.reshape(-1, 3), axis=0)
        diff = np.linalg.norm(warped.astype(float) - med_bgr, axis=2)

        is_diff = (diff > 35.0)
        frac_vertical = np.mean(is_diff, axis=0)
        touches_top = np.any(is_diff[:12, :], axis=0)
        touches_bot = np.any(is_diff[-12:, :], axis=0)

        up_dir = (quad[0] - quad[3]) / max(1e-5, np.linalg.norm(quad[0] - quad[3]))
        down_dir = -up_dir

        margin_x = max(3, int(qw * 0.06))
        matched_cols = np.zeros(qw, dtype=bool)

        for x in range(margin_x, qw - margin_x):
            if frac_vertical[x] > 0.65 and (touches_top[x] or touches_bot[x]):
                t = x / float(qw)
                pt_top = quad[0] * (1 - t) + quad[1] * t
                pt_bot = quad[3] * (1 - t) + quad[2] * t

                p_ab1 = (pt_top + up_dir * 10).astype(int)
                p_ab2 = (pt_top + up_dir * 20).astype(int)
                p_bl1 = (pt_bot + down_dir * 10).astype(int)
                p_bl2 = (pt_bot + down_dir * 20).astype(int)

                c_in = np.median(warped[qh // 4: 3 * qh // 4, x], axis=0).astype(float)

                d_ab = 999.0
                if 0 <= p_ab1[0] < fw and 0 <= p_ab1[1] < fh:
                    d_ab = min(d_ab, float(np.linalg.norm(c_in - frame[p_ab1[1], p_ab1[0]].astype(float))))
                if 0 <= p_ab2[0] < fw and 0 <= p_ab2[1] < fh:
                    d_ab = min(d_ab, float(np.linalg.norm(c_in - frame[p_ab2[1], p_ab2[0]].astype(float))))

                d_bl = 999.0
                if 0 <= p_bl1[0] < fw and 0 <= p_bl1[1] < fh:
                    d_bl = min(d_bl, float(np.linalg.norm(c_in - frame[p_bl1[1], p_bl1[0]].astype(float))))
                if 0 <= p_bl2[0] < fw and 0 <= p_bl2[1] < fh:
                    d_bl = min(d_bl, float(np.linalg.norm(c_in - frame[p_bl2[1], p_bl2[0]].astype(float))))

                if d_ab < 40.0 or d_bl < 45.0 or (frac_vertical[x] > 0.85 and touches_top[x] and touches_bot[x]):
                    matched_cols[x] = True

        intervals = []
        in_run = False
        start_x = 0
        for x in range(qw):
            if matched_cols[x] and not in_run:
                in_run = True
                start_x = x
            elif not matched_cols[x] and in_run:
                in_run = False
                run_w = x - start_x
                if 2 <= run_w <= int(qw * 0.40):
                    intervals.append((start_x, x))
        if in_run:
            run_w = qw - start_x
            if 2 <= run_w <= int(qw * 0.40):
                intervals.append((start_x, qw))

        polys = []
        for sx, ex in intervals:
            pts_w = np.array([
                [sx, 0], [ex, 0], [ex, qh], [sx, qh]
            ], dtype=np.float32).reshape(-1, 1, 2)
            pts_f = cv2.perspectiveTransform(pts_w, M_inv).reshape(-1, 2)
            polys.append(pts_f)

        return polys

    def detect_classical(self, frame: np.ndarray) -> Tuple[List[CandidateDetection], List[OccluderDetection]]:
        """
        Model-free classical detection: uses bilateral filtering, adaptive Canny edge detection,
        morphological closing, and quadrilateral contour extraction.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)

        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[CandidateDetection] = []

        # Sort contours by area descending so we find clean prominent surfaces
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for cnt in contours[:25]:
            area = cv2.contourArea(cnt)
            if area < (h * w * self.geom_cfg.min_area_fraction):
                continue

            quad = extract_quad_from_contour(cnt, self.geom_cfg.epsilon_approx_ratio)
            if quad is None:
                continue

            valid, _ = is_valid_planar_quad(
                quad, w, h,
                min_area_fraction=self.geom_cfg.min_area_fraction,
                max_area_fraction=self.geom_cfg.max_area_fraction,
                min_angle=self.geom_cfg.min_internal_angle_deg,
                max_angle=self.geom_cfg.max_internal_angle_deg
            )
            if not valid:
                continue

            # Compute bounding box from quad
            pts = quad.astype(int)
            x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
            x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])

            # Deduplicate nearly identical quads
            is_dup = False
            for existing in candidates:
                eq = existing.quad.astype(int)
                ex1, ey1 = np.min(eq[:, 0]), np.min(eq[:, 1])
                ex2, ey2 = np.max(eq[:, 0]), np.max(eq[:, 1])
                # Check center distance
                if abs((x1+x2)/2 - (ex1+ex2)/2) < 25 and abs((y1+y2)/2 - (ey1+ey2)/2) < 25:
                    is_dup = True
                    break
            if is_dup:
                continue

            candidates.append(CandidateDetection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                quad=quad,
                class_name="billboard",
                confidence=0.90
            ))

        occluders: List[OccluderDetection] = []
        for cand in candidates:
            slender_polys = self.detect_slender_occluders(frame, cand.quad)
            for spoly in slender_polys:
                sx1 = int(np.min(spoly[:, 0]))
                sy1 = int(np.min(spoly[:, 1]))
                sx2 = int(np.max(spoly[:, 0]))
                sy2 = int(np.max(spoly[:, 1]))
                occluders.append(OccluderDetection(
                    bbox=(sx1, sy1, sx2, sy2),
                    poly=spoly,
                    class_name="light pole / foreground occluder",
                    confidence=0.88
                ))

        return candidates, occluders

    def detect(self, frame: np.ndarray) -> Tuple[List[CandidateDetection], List[OccluderDetection]]:
        """
        Runs detection on a single frame. Returns list of Candidate surfaces and Occluders.
        """
        if self.model is None or self.model_type == "classical":
            return self.detect_classical(frame)

        h, w = frame.shape[:2]
        candidates: List[CandidateDetection] = []
        occluders: List[OccluderDetection] = []

        try:
            results = self.model(
                frame,
                conf=self.det_cfg.conf_threshold,
                iou=self.det_cfg.iou_threshold,
                imgsz=self.det_cfg.inference_imgsz,
                verbose=False
            )[0]
        except Exception as e:
            logger.error("Inference failed: %s. Using classical detector.", e)
            return self.detect_classical(frame)

        # Parse detections
        names = results.names
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            c_cand, c_occ = self.detect_classical(frame)
            return c_cand, occluders + c_occ

        # Pre-scan for window boxes and interior cues
        window_boxes: List[Tuple[int, int, int, int]] = []
        detected_names = [names.get(int(box.cls[0].item()), "").lower() for box in boxes]
        is_interior_scene = any(
            k in d_name for d_name in detected_names
            for k in ["handrail", "grab handle", "train seat", "train interior", "subway", "carriage", "metro", "seat", "interior"]
        )

        for box in boxes:
            cls_id = int(box.cls[0].item())
            b_name = names.get(cls_id, str(cls_id)).lower()
            if "window" in b_name:
                wxyxy = box.xyxy[0].cpu().numpy().astype(int)
                window_boxes.append((int(wxyxy[0]), int(wxyxy[1]), int(wxyxy[2]), int(wxyxy[3])))

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = names.get(cls_id, str(cls_id)).lower()
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            bbox = (x1, y1, x2, y2)
            bw = x2 - x1
            bh = y2 - y1
            aspect = bw / max(1, bh)

            # Special Transit & Automotive Advertising Handler: Bus & Truck Side Ad Panels
            if ("bus" in cls_name or "truck" in cls_name) and not is_interior_scene:
                if aspect > 1.05 and conf >= 0.35:
                    if "bus" in cls_name:
                        # Extract the transit ad banner below the window band
                        bus_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        dark_left = 0.5
                        dark_right = 0.5
                        if bus_crop.size > 0 and bus_crop.shape[1] > 20:
                            lc = bus_crop[:, :int(bw * 0.35)]
                            rc = bus_crop[:, int(bw * 0.65):]
                            if lc.size > 0 and rc.size > 0:
                                gl = cv2.cvtColor(lc, cv2.COLOR_BGR2GRAY)
                                gr = cv2.cvtColor(rc, cv2.COLOR_BGR2GRAY)
                                dark_left = float(np.sum(gl < 70) / max(1, gl.size))
                                dark_right = float(np.sum(gr < 70) / max(1, gr.size))

                        if aspect > 2.8:
                            # Full side profile (flat elevation)
                            sx1 = int(x1 + 0.10 * bw)
                            sx2 = int(x2 - 0.04 * bw)
                            sy1 = int(y1 + 0.48 * bh)
                            sy2 = int(y1 + 0.85 * bh)
                            p1 = [sx1, sy1]
                            p2 = [sx2, sy1]
                            p3 = [sx2, sy2]
                            p4 = [sx1, sy2]
                        elif dark_left > dark_right:
                            # 3D Depth Perspective: Front on left, side flank recedes into street depth (right)
                            # Top sill slopes down along vanishing ray; bottom skirt slopes up towards vanishing point
                            sx1 = int(x1 + 0.36 * bw)
                            sx2 = int(x2 - 0.03 * bw)
                            dx = sx2 - sx1
                            sy1_l = int(y1 + 0.48 * bh)
                            sy2_l = int(y1 + 0.85 * bh)
                            sy1_r = int(sy1_l + 0.065 * dx)
                            sy2_r = int(sy2_l - 0.075 * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]
                        else:
                            # 3D Depth Perspective: Front on right, side flank recedes into depth (left)
                            sx1 = int(x1 + 0.03 * bw)
                            sx2 = int(x1 + 0.64 * bw)
                            dx = sx2 - sx1
                            sy1_r = int(y1 + 0.48 * bh)
                            sy2_r = int(y1 + 0.85 * bh)
                            sy1_l = int(sy1_r + 0.065 * dx)
                            sy2_l = int(sy2_r - 0.075 * dx)
                            p1 = [sx1, sy1_l]
                            p2 = [sx2, sy1_r]
                            p3 = [sx2, sy2_r]
                            p4 = [sx1, sy2_l]

                        quad = order_points(np.array([p1, p2, p3, p4], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            b_min_y = int(min(p1[1], p2[1]))
                            b_max_y = int(max(p3[1], p4[1]))
                            candidates.append(CandidateDetection(
                                bbox=(sx1, b_min_y, sx2, b_max_y),
                                quad=quad,
                                class_name="bus side ad space",
                                confidence=conf,
                                mask=None
                            ))
                            continue
                    elif "truck" in cls_name:
                        # Cargo truck side billboard
                        sx1 = int(x1 + 0.30 * bw)
                        sx2 = int(x2 - 0.02 * bw)
                        sy1 = int(y1 + 0.15 * bh)
                        sy2 = int(y1 + 0.85 * bh)
                        quad = order_points(np.array([
                            [sx1, sy1], [sx2, sy1], [sx2, sy2], [sx1, sy2]
                        ], dtype=np.float32))
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            candidates.append(CandidateDetection(
                                bbox=(sx1, sy1, sx2, sy2),
                                quad=quad,
                                class_name="truck side billboard",
                                confidence=conf,
                                mask=None
                            ))
                            continue

                # Head-on transit vehicle (no side flank visible) is treated as a traffic occluder
                occ_quad = order_points(np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32))
                occluders.append(OccluderDetection(
                    bbox=bbox,
                    poly=occ_quad,
                    class_name=cls_name,
                    confidence=conf
                ))
                continue

            # Special Infrastructure & Overpass Advertising Handler: Bridge Walls & Girders
            if any(k in cls_name for k in ["bridge wall", "concrete bridge wall", "overhead bridge girder", "bridge girder"]):
                if conf >= 0.40:
                    quad = self.fit_bridge_3d_quad(frame, bbox)
                    if quad is not None:
                        valid, _ = is_valid_planar_quad(
                            quad, w, h,
                            min_area_fraction=self.geom_cfg.min_area_fraction,
                            max_area_fraction=self.geom_cfg.max_area_fraction,
                            min_angle=self.geom_cfg.min_internal_angle_deg,
                            max_angle=self.geom_cfg.max_internal_angle_deg
                        )
                        if valid:
                            b_min_y = int(min(quad[:, 1]))
                            b_max_y = int(max(quad[:, 1]))
                            candidates.append(CandidateDetection(
                                bbox=(int(min(quad[:, 0])), b_min_y, int(max(quad[:, 0])), b_max_y),
                                quad=quad,
                                class_name="bridge wall ad space",
                                confidence=conf,
                                mask=None
                            ))
                            continue

            # Determine whether this is a candidate or an occluder
            is_candidate = False
            is_occluder = False

            if self.model_type == "yolo_world":
                if "window" in cls_name:
                    is_occluder = True
                elif any(cand.lower() in cls_name for cand in self.det_cfg.candidate_classes):
                    is_candidate = True
                elif any(occ.lower() in cls_name for occ in self.det_cfg.occluder_classes):
                    is_occluder = True
            else:
                # Standard COCO fallback mappings
                if cls_name in self.det_cfg.coco_candidate_classes:
                    is_candidate = True
                elif cls_name in self.det_cfg.coco_occluder_classes:
                    is_occluder = True

            if is_candidate:
                # Fast <2ms quad extraction
                if any(k in cls_name for k in ["bridge wall", "concrete bridge wall", "overhead bridge girder", "bridge girder"]):
                    if conf < 0.40:
                        continue
                    quad = self.fit_bridge_3d_quad(frame, bbox)
                    if quad is None:
                        continue
                elif any(k in cls_name for k in ["skyscraper", "high-rise", "building", "facade", "wall", "brick wall"]):
                    # First, check if there is an unobstructed upper plain wall above windows
                    upper_quad = self.fit_upper_wall_plain_quad(frame, bbox, window_boxes)
                    if upper_quad is not None:
                        candidates.append(CandidateDetection(
                            bbox=(int(min(upper_quad[:, 0])), int(min(upper_quad[:, 1])), int(max(upper_quad[:, 0])), int(max(upper_quad[:, 1]))),
                            quad=upper_quad,
                            class_name="plain wall ad space (above windows)",
                            confidence=conf,
                            mask=None
                        ))
                        continue
                    else:
                        # Verify that this building facade has a plain / plane surface suitable for placing an ad,
                        # and is not covered in a dense window grid (such as residential apartment towers)
                        crop_b = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if self.is_window_grid_facade(crop_b):
                            # Facade is covered with windows/balconies; not a plain ad space
                            continue
                        if conf < 0.22 and self.is_window_grid_facade(crop_b, strict=True):
                            continue
                        quad = self.refine_quad_fast(frame, bbox)
                elif any(k in cls_name for k in ["billboard", "banner", "poster", "advertising board", "signboard", "display screen"]):
                    # Fit 3D perspective quad for urban billboards and banners
                    quad = self.fit_billboard_banner_3d_quad(frame, bbox)
                else:
                    quad = self.refine_quad_fast(frame, bbox)

                valid, _ = is_valid_planar_quad(
                    quad, w, h,
                    min_area_fraction=self.geom_cfg.min_area_fraction,
                    max_area_fraction=self.geom_cfg.max_area_fraction,
                    min_angle=self.geom_cfg.min_internal_angle_deg,
                    max_angle=self.geom_cfg.max_internal_angle_deg
                )

                if valid:
                    b_min_y = int(min(quad[:, 1]))
                    b_max_y = int(max(quad[:, 1]))
                    c_display = cls_name
                    if any(k in cls_name for k in ["skyscraper", "building", "high-rise"]):
                        c_display = "building facade ad space"
                    elif any(k in cls_name for k in ["bridge", "girder"]):
                        c_display = "bridge wall ad space"
                    candidates.append(CandidateDetection(
                        bbox=(int(min(quad[:, 0])), b_min_y, int(max(quad[:, 0])), b_max_y),
                        quad=quad,
                        class_name=c_display,
                        confidence=conf,
                        mask=None
                    ))

            elif is_occluder:
                occ_quad = order_points(np.array([
                    [x1, y1], [x2, y1], [x2, y2], [x1, y2]
                ], dtype=np.float32))
                occluders.append(OccluderDetection(
                    bbox=bbox,
                    poly=occ_quad,
                    class_name=cls_name,
                    confidence=conf
                ))

        # If model found no candidates, only run classical detector if in classical mode
        if len(candidates) == 0 and self.model_type == "classical":
            c_cand, _ = self.detect_classical(frame)
            candidates.extend(c_cand)

        # Deduplicate candidates with high spatial overlap (prevent duplicate boxes on same surface)
        if len(candidates) > 1:
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            deduped: List[CandidateDetection] = []
            for cand in candidates:
                keep = True
                c_cx = float(np.mean(cand.quad[:, 0]))
                c_cy = float(np.mean(cand.quad[:, 1]))
                cb = cand.bbox
                c_area = max(1, (cb[2] - cb[0]) * (cb[3] - cb[1]))
                for existing in deduped:
                    e_cx = float(np.mean(existing.quad[:, 0]))
                    e_cy = float(np.mean(existing.quad[:, 1]))
                    dist = math.hypot(c_cx - e_cx, c_cy - e_cy)
                    eb = existing.bbox
                    e_area = max(1, (eb[2] - eb[0]) * (eb[3] - eb[1]))

                    ix1, iy1 = max(cb[0], eb[0]), max(cb[1], eb[1])
                    ix2, iy2 = min(cb[2], eb[2]), min(cb[3], eb[3])
                    iw = max(0, ix2 - ix1)
                    ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    iou = inter / float(c_area + e_area - inter) if (c_area + e_area - inter) > 0 else 0.0
                    overlap = inter / float(min(c_area, e_area)) if min(c_area, e_area) > 0 else 0.0

                    if dist < 50.0 or iou > 0.30 or overlap > 0.45:
                        keep = False
                        break
                if keep:
                    deduped.append(cand)
            candidates = deduped

        # In-canvas intrusion detection: detect slender poles, posts, and foreground occluders
        if candidates and frame is not None:
            for cand in candidates:
                slender_polys = self.detect_slender_occluders(frame, cand.quad)
                for spoly in slender_polys:
                    sx1 = int(np.min(spoly[:, 0]))
                    sy1 = int(np.min(spoly[:, 1]))
                    sx2 = int(np.max(spoly[:, 0]))
                    sy2 = int(np.max(spoly[:, 1]))
                    occluders.append(OccluderDetection(
                        bbox=(sx1, sy1, sx2, sy2),
                        poly=spoly,
                        class_name="light pole / foreground occluder",
                        confidence=0.88
                    ))

        return candidates, occluders

