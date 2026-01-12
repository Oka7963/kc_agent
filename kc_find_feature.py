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

@dataclass
class MatchResult:
    """Container for feature matching results."""
    found: bool
    score: float
    bbox_xywh: Tuple[int, int, int, int]          # in client-area coords
    bbox_screen_xywh: Tuple[int, int, int, int]   # in screen coords
    inliers: int
    matches: int
    proj_corners: Optional[np.ndarray] = None     # 4x2 points in scene coords


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

    def find_template(
        self,
        scene_bgr: Optional[np.ndarray] = None,
        min_matches: int = 20,
        ratio: float = 0.75,
        ransac_threshold: float = 5.0
    ) -> Optional[MatchResult]:
        """Find the template in the given scene or current window capture.

        Args:
            scene_bgr: Optional BGR image to search in (captures window if None)
            min_matches: Minimum good matches to attempt homography
            ratio: Lowe's ratio test threshold
            ransac_threshold: RANSAC reprojection threshold in pixels

        Returns:
            MatchResult if found, None otherwise
        """
        if scene_bgr is None:
            logging.info("Capturing window...")
            scene_bgr = self.capture_window()

        # Convert to grayscale for feature detection
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
        templ_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Detect and compute ORB features
        kp1, des1 = self.orb.detectAndCompute(templ_gray, None)
        kp2, des2 = self.orb.detectAndCompute(scene_gray, None)

        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return None

        # Match features using k-NN
        matches = self.bf.knnMatch(des1, des2, k=2)

        # Apply ratio test (Lowe's)
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append(m)

        if len(good) < min_matches:
            return None

        # Estimate homography
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        if H is None or mask is None:
            return None

        inliers = int(mask.sum())
        if inliers < max(8, min_matches // 2):
            return None

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

        return MatchResult(
            found=True,
            score=float(inliers) / len(good),
            bbox_xywh=(x1, y1, w, h),
            bbox_screen_xywh=(screen_x, screen_y, w, h),
            inliers=inliers,
            matches=len(good),
            proj_corners=proj_corners
        )

    def visualize_match(
        self,
        result: MatchResult,
        scene_bgr: Optional[np.ndarray] = None,
        show: bool = True
    ) -> np.ndarray:
        """Visualize the matching result on the scene image.

        Args:
            result: MatchResult from find_template
            scene_bgr: Optional scene image (captures window if None)
            show: Whether to show the result using cv2.imshow

        Returns:
            Visualization image with match overlay
        """
        if scene_bgr is None:
            logger.info("capture window")
            scene_bgr = self.capture_window()

        vis = scene_bgr.copy()

        if result.found and result.proj_corners is not None:
            # Draw projected corners
            pts = result.proj_corners.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Draw info text
            x, y = pts[0, 0]
            text = f"Inliers: {result.inliers}/{result.matches} Score: {result.score:.2f}"
            cv2.putText(vis, text, (x, max(0, y - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if show:
            cv2.imshow("Feature Match", vis)
            cv2.waitKey(1)

        return vis

def main():
    title = "kc_simulator"
    template = ".\crops\battle_0\battle_start.png"
    matcher = FeatureMatcher(title, template)

def _main():
    """Command-line interface for feature matching."""
    parser = argparse.ArgumentParser(description="Feature-based template matching for window content.")
    parser.add_argument("--title", required=True, help="Window title substring (case-insensitive).")
    parser.add_argument("--template", required=True, help="Template image path (PNG recommended).")
    parser.add_argument("--min-matches", type=int, default=20,
                       help="Minimum good matches to attempt homography.")
    parser.add_argument("--ratio", type=float, default=0.75,
                       help="Lowe ratio test threshold.")
    parser.add_argument("--ransac", type=float, default=5.0,
                       help="RANSAC reprojection threshold (pixels).")
    parser.add_argument("--show", action="store_true",
                       help="Show debug window with visualization.")
    parser.add_argument("--loop", action="store_true",
                       help="Run continuously until interrupted.")
    parser.add_argument("--interval", type=float, default=0.2,
                       help="Loop interval in seconds.")
    args = parser.parse_args()

    try:
        # Initialize the feature matcher
        matcher = FeatureMatcher(args.title, args.template)

        def process_frame() -> Dict:
            """Process a single frame and return results as a dict."""
            result = matcher.find_template(
                min_matches=args.min_matches,
                ratio=args.ratio,
                ransac_threshold=args.ransac
            )

            # Prepare output dictionary
            out = {
                "found": result.found if result else False,
                "title_query": args.title,
                "window": {
                    "hwnd": matcher.window.hwnd,
                    "client_rect": matcher.window.client_rect,
                    "screen_rect": matcher.window.screen_rect
                },
                "method": "ORB+BFMatcher+RANSAC-homography",
                "min_matches": args.min_matches,
                "ratio": args.ratio,
                "ransac_reproj_thresh": args.ransac,
            }

            if result and result.found:
                out.update({
                    "score": result.score,
                    "inliers": result.inliers,
                    "good_matches": result.matches,
                    "bbox_client_xywh": list(result.bbox_xywh),
                    "bbox_screen_xywh": list(result.bbox_screen_xywh),
                    "proj_corners_client_xy": result.proj_corners.tolist() if result.proj_corners is not None else []
                })

                if args.show:
                    matcher.visualize_match(result)

            return out

        # Single run mode
        if not args.loop:
            result = process_frame()
            print(json.dumps(result, ensure_ascii=False, indent=2))

            if args.show:
                cv2.waitKey(0)
            return 0 if result["found"] else 1

        # Continuous mode
        try:
            while True:
                result = process_frame()
                print(json.dumps(result, ensure_ascii=False, indent=2))
                time.sleep(max(0.01, args.interval))
        except KeyboardInterrupt:
            pass

        return 0

    except Exception as e:
        print(f"ERROR: {str(e)}", file=sys.stderr)
        return 1
    finally:
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())

#python kc_find_feature.py --title "kc_simulator" --template .\crops\battle_0\battle_start.png --show
