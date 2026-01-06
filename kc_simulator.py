import cv2
import time

WINDOW_NAME = "switch_image"
BASE_IMAGE = "screenshots\\port.png"

# 視覺效果
BOX_COLOR = (0, 0, 255)      # 紅色 (BGR)
BOX_THICKNESS = 2
HIT_HIGHLIGHT_MS = 250
HIT_THICKNESS = 4

# 你可以用 (x, y, w, h) 來定義區域；命中後切換到 target
# 每張圖片最多 3 個區域
REGIONS_MAP = {
    "screenshots\\port.png": [
        {"name": "TL_50x50_to_A", "rect": (0, 0, 50, 50), "target": "screenshots\\next_battle_selection.png"},
        # {"name": "TR_50x50_to_B", "rect": (590, 0, 50, 50), "target": "b.png"},
        # {"name": "BL_50x50_to_C", "rect": (0, 430, 50, 50), "target": "c.png"},
    ],
    "screenshots\\next_battle_selection.png": [
        {"name": "TL_50x50_back_to_base", "rect": (0, 0, 50, 50), "target": "screenshots\\port.png"},
        # 也可以再加第2、3個區域
    ],
    # "b.png": [...最多3個...]
}

def load_bgr(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img

def enforce_region_limit(regions_map: dict, limit: int = 3):
    for k, regions in regions_map.items():
        if len(regions) > limit:
            raise ValueError(f'Image "{k}" has {len(regions)} regions, exceeds limit={limit}')

def rect_hit(x: int, y: int, rect):
    rx, ry, rw, rh = rect
    return (rx <= x < rx + rw) and (ry <= y < ry + rh)

def rect_to_xyxy(rect):
    rx, ry, rw, rh = rect
    return rx, ry, rx + rw, ry + rh

class ImageSwitcher:
    def __init__(self, base_path: str, regions_map: dict):
        enforce_region_limit(regions_map, limit=3)

        self.regions_map = regions_map
        self.images = {}
        self.current_path = base_path

        # preload base
        self.images[base_path] = load_bgr(base_path)
        self.current = self.images[base_path]

        # 命中高亮用
        self.last_hit_rect_xyxy = None
        self.last_hit_time = 0.0

    def get_regions_for_current(self):
        return self.regions_map.get(self.current_path, [])

    def set_image(self, path: str):
        if path not in self.images:
            self.images[path] = load_bgr(path)
        self.current_path = path
        self.current = self.images[path]

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        for r in self.get_regions_for_current():
            if rect_hit(x, y, r["rect"]):
                print(f'Hit region: {r["name"]}, switch to {r["target"]}')
                self.last_hit_rect_xyxy = rect_to_xyxy(r["rect"])
                self.last_hit_time = time.time()
                self.set_image(r["target"])
                return

    def render(self):
        vis = self.current.copy()

        # 畫目前這張圖的可點擊區域紅框
        for r in self.get_regions_for_current():
            x1, y1, x2, y2 = rect_to_xyxy(r["rect"])

            # 畫框時稍微保護一下座標（避免超出尺寸）
            h, w = vis.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w, x2))
            y2 = max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), BOX_COLOR, BOX_THICKNESS)

        # 命中後短暫高亮（更粗的紅框）
        if self.last_hit_rect_xyxy is not None:
            elapsed_ms = (time.time() - self.last_hit_time) * 1000.0
            if elapsed_ms <= HIT_HIGHLIGHT_MS:
                x1, y1, x2, y2 = self.last_hit_rect_xyxy
                h, w = vis.shape[:2]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), BOX_COLOR, HIT_THICKNESS)
            else:
                self.last_hit_rect_xyxy = None

        return vis

def main():
    switcher = ImageSwitcher(BASE_IMAGE, REGIONS_MAP)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, switcher.on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, switcher.render())
        key = cv2.waitKey(16) & 0xFF

        # r: 回到 base
        if key == ord('r'):
            switcher.set_image(BASE_IMAGE)

        # q / ESC: 離開
        if key in (ord('q'), 27):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
