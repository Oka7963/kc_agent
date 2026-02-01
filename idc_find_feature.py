import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import cv2
import mss
import numpy as np
import win32gui
import win32con


from logger import setup_logger

logger = setup_logger("find_feature")

'''
    template: or crops, or key examples for matching
    Window: basically its poi app
    roi: to minimize the search space, we predefine the rough region of interest


    Document:
        1. This is where we load the screen shot and find the key to press.
        2. At this point, we know the action is required
        3. The input for FeatureMatcher is the window title and the template image
            a. window title: the title of the window that we want to find the template in
            b. template path: the path to the crops which is the key examples for matching
            c. roi: the rough region of interest which defind in the workflow TODO
        4. The output is the match result and should contains as follow
            a. found or not
            b. matching score
            c. key bbox in window coords (xywh)
            d. key bbox in screen coords (xywh)
            e. rough region of interest in scene coords that we should search the key
            f. matches method

'''


@dataclass
class MatchResult:
    """Container for feature matching results."""
    found: bool
    score: float
    bbox_xywh: Tuple[int, int, int, int]          # in client-area coords: where the crops should point to
    bbox_screen_xywh: Tuple[int, int, int, int]   # in screen coords: where the mouse should point to
    roi: Tuple[int, int, int, int]             # rough region of interest in scene coords that we should search the key
    method: str                                         # matches method: how do we find the template


class WindowInfo(NamedTuple):
    """Container for window information."""
    hwnd: int
    title: str
    client_rect: Tuple[int, int, int, int]  # left, top, right, bottom
    screen_rect: Tuple[int, int, int, int]  # left, top, right, bottom


class FeatureMatcher:
    """Feature matching for finding template images within window captures."""

    def __init__(self, window_title: str, template_path: Union[str, Path]):
        """Initialize the feature matcher with a window title and template image.

        Args:
            window_title: Substring to match in the window title (case-insensitive)
            template_path: Path to the template image file
        """
        self._set_dpi_aware()
        self.window = self._find_window_by_title(window_title)
        if not self.window:
            raise ValueError(f"Window with title containing '{window_title}' not found")

        self.template = self._load_template(template_path)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    @staticmethod
    def _set_dpi_aware() -> None:
        """Make coordinates match physical pixels under DPI scaling."""
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    @staticmethod
    def _find_window_by_title(title_substr: str) -> Optional[WindowInfo]:
        """Find a window by title substring."""
        title_substr = title_substr.lower()
        result = {"info": None}

        def enum_cb(hwnd: int, _) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            title = win32gui.GetWindowText(hwnd) or ""
            if title_substr in title.lower():
                client_rect = win32gui.GetClientRect(hwnd)  # (left, top, right, bottom)
                (sx, sy) = win32gui.ClientToScreen(hwnd, (0, 0))
                screen_rect = (sx, sy, sx + client_rect[2], sy + client_rect[3])
                result["info"] = WindowInfo(
                    hwnd=hwnd,
                    title=title,
                    client_rect=client_rect,
                    screen_rect=screen_rect
                )

        win32gui.EnumWindows(enum_cb, None)
        return result["info"]

    @staticmethod
    def _load_template(path: Union[str, Path]) -> np.ndarray:
        """Load template image from file."""
        path = Path(path) if isinstance(path, str) else path
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load template image: {path}")
        return img

    def capture_window(self) -> np.ndarray:
        """Capture the current window's client area."""
        left, top, right, bottom = self.window.screen_rect
        w = max(1, right - left)
        h = max(1, bottom - top)

        region = {"left": left, "top": top, "width": w, "height": h}
        with mss.mss() as sct:
            shot = sct.grab(region)  # BGRA
            img = np.array(shot, dtype=np.uint8)

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def match_template_ORB(
        self,
        scene_bgr: Optional[np.ndarray] = None,
        min_matches: int = 20,
        ratio: float = 0.75,
        ransac_threshold: float = 5.0
    ) -> MatchResult:
        """
        Note: you can treat as main function entry for identify scene
        Find the template in the given scene or current window capture.

        Args:
            scene_bgr: BGR image to search in (captures window if None)
            min_matches: Minimum good matches to attempt homography (H matrix)
            ratio: Lowe's ratio test threshold
            ransac_threshold: RANSAC reprojection threshold in pixels

        Returns:
            MatchResult object (found=False if no match)
        """
        method_name = "ORB+BFMatcher+kNN+RANSAC"

        if scene_bgr is None:
            # capture window since no scene is provided
            logger.info("Capturing window...")
            scene_bgr = self.capture_window()

        # Convert to grayscale for feature detection
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        templ_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Detect and compute ORB features
        kp1, des1 = self.orb.detectAndCompute(templ_gray, None)
        kp2, des2 = self.orb.detectAndCompute(scene_gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return MatchResult(
                found=False,
                score=0.0,
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        # Match features using k-NN
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test (Lowe's)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < min_matches:
            return MatchResult(
                found=False,
                score=float(len(good)) / len(kp1) if len(kp1) > 0 else 0.0,
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        # Estimate homography -> use src_pts to map to dst_pts to find the points in the scene (H matrix)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        if H is None or mask is None:
            return MatchResult(
                found=False,
                score=float(len(good)) / len(kp1) if len(kp1) > 0 else 0.0,
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        inliers = int(mask.sum())
        if inliers < max(8, min_matches // 2):
            return MatchResult(
                found=False,
                score=float(inliers) / len(good),
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        # Project template corners to scene
        h_t, w_t = templ_gray.shape[:2]
        templ_corners = np.float32([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]]).reshape(-1, 1, 2)
        proj_corners = cv2.perspectiveTransform(templ_corners, H).reshape(4, 2)

        # Calculate bounding box
        xs = proj_corners[:, 0]
        ys = proj_corners[:, 1]
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())
        w, h = max(1, x2 - x1), max(1, y2 - y1)

        # Calculate screen coordinates
        screen_x = self.window.screen_rect[0] + x1
        screen_y = self.window.screen_rect[1] + y1

        # Calculate ROI (rough region of interest - could be expanded based on template size)
        roi_x = max(0, x1 - w // 2)
        roi_y = max(0, y1 - h // 2)
        roi_w = min(scene_bgr.shape[1] - roi_x, w * 2)
        roi_h = min(scene_bgr.shape[0] - roi_y, h * 2)

        return MatchResult(
            found=True,
            score=float(inliers) / len(good),
            bbox_xywh=(x1, y1, w, h),
            bbox_screen_xywh=(screen_x, screen_y, w, h),
            roi=(roi_x, roi_y, roi_w, roi_h),
            method=method_name
        )

    def match_template_tm(
        self,
        scene_bgr: Optional[np.ndarray] = None,
        threshold: float = 0.8
    ) -> MatchResult:
        """
        Template matching using OpenCV's template matching methods.

        Args:
            scene_bgr: BGR image to search in (captures window if None)
            threshold: Minimum match score threshold (0.0 to 1.0)

        Returns:
            MatchResult object (found=False if no match)
        """
        method_name = "TemplateMatching"

        if scene_bgr is None:
            # capture window since no scene is provided
            logger.info("Capturing window...")
            scene_bgr = self.capture_window()

        # Convert to grayscale for template matching
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        templ_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Get template dimensions
        h_t, w_t = templ_gray.shape[:2]

        # Check if template is larger than scene
        if h_t > scene_gray.shape[0] or w_t > scene_gray.shape[1]:
            return MatchResult(
                found=False,
                score=0.0,
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        # Perform template matching
        result = cv2.matchTemplate(scene_gray, templ_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Use TM_CCOEFF_NORMED, so higher values are better
        score = max_val

        if score < threshold:
            return MatchResult(
                found=False,
                score=score,
                bbox_xywh=(0, 0, 0, 0),
                bbox_screen_xywh=(0, 0, 0, 0),
                roi=(0, 0, 0, 0),
                method=method_name
            )

        # Get top-left corner of match
        x, y = max_loc

        # Calculate screen coordinates
        screen_x = self.window.screen_rect[0] + x
        screen_y = self.window.screen_rect[1] + y

        # Calculate ROI (rough region of interest - could be expanded based on template size)
        roi_x = max(0, x - w_t // 2)
        roi_y = max(0, y - h_t // 2)
        roi_w = min(scene_bgr.shape[1] - roi_x, w_t * 2)
        roi_h = min(scene_bgr.shape[0] - roi_y, h_t * 2)

        return MatchResult(
            found=True,
            score=score,
            bbox_xywh=(x, y, w_t, h_t),
            bbox_screen_xywh=(screen_x, screen_y, w_t, h_t),
            roi=(roi_x, roi_y, roi_w, roi_h),
            method=method_name
        )


    def visualize_match(
        self,
        result: MatchResult,
        scene_bgr: Optional[np.ndarray] = None,
        show: bool = True
    ) -> np.ndarray:
        """Visualize the matching result on the scene image.

        Args:
            result: MatchResult from match_template_ORB or match_template_tm
            scene_bgr: Optional scene image (captures window if None)
            show: Whether to show the result using cv2.imshow

        Returns:
            Visualization image with match overlay
        """
        if scene_bgr is None:
            logger.info("capture window")
            scene_bgr = self.capture_window()

        vis = scene_bgr.copy()

        if result.found:
            x, y, w, h = result.bbox_xywh

            # Draw rectangle for the matched region
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Draw ROI (region of interest) if it's valid
            if result.roi != (0, 0, 0, 0):
                roi_x, roi_y, roi_w, roi_h = result.roi
                cv2.rectangle(vis, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 1)
                cv2.putText(vis, "ROI", (roi_x, roi_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Draw info text
            text = f"{result.method}: Score={result.score:.3f}"
            cv2.putText(vis, text, (x, max(0, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(vis, (center_x, center_y), 3, (0, 0, 255), -1)

            # Draw screen coordinates
            screen_text = f"Screen: ({result.bbox_screen_xywh[0]}, {result.bbox_screen_xywh[1]})"
            cv2.putText(vis, screen_text, (x, max(0, y - 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if show:
            cv2.imshow("Feature Match", vis)
            cv2.waitKey(1)

        return vis

if __name__ == "__main__":
    title = "kc_simulator"
    # template = ".\\crops\\battle_0\\battle_start.png"
    template = ".\\crops\\battle_finish\\battle_finish_next.png"
    matcher = FeatureMatcher(title, template)
    min_matches=20
    ratio=0.75
    ransac_threshold=5.0
    def process_frame() -> Dict:
        """Process a single frame and return results as a dict."""
        # You can choose which method to use:
        # result = matcher.match_template_ORB(min_matches=min_matches, ratio=ratio, ransac_threshold=ransac_threshold)
        result = matcher.match_template_tm(threshold=0.8)

        # Always return a result since both methods now always return MatchResult
        if result.found:
            # debug visualize match
            matcher.visualize_match(result)

        # Prepare match result dictionary
        out = {
            "found": result.found,
            "title_query": title,
            "window": {
                "hwnd": matcher.window.hwnd,
                "client_rect": matcher.window.client_rect,
                "screen_rect": matcher.window.screen_rect
            },
            "match": {
                "method": result.method,
                "score": result.score,
                "bbox_client_xywh": result.bbox_xywh,
                "bbox_screen_xywh": result.bbox_screen_xywh,
                "roi": result.roi
            }
        }

        print("find log:", json.dumps(out, ensure_ascii=False, indent=2))
        logger.debug(json.dumps(out, ensure_ascii=False, indent=2))

        return out

    result = process_frame()

    cv2.waitKey(0)

