import sys
import json
import os
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from typing import *
import argparse
import glob

MAX_REGIONS_PER_IMAGE = 6
MIN_RECT_SIZE = 4  # px


@dataclass
class Region:
    rid: str
    name: str
    rect: QtCore.QRect  # image pixel coords
    target: str
    rect_item: QtWidgets.QGraphicsRectItem | None = None


class ImageCanvas(QtWidgets.QGraphicsView):
    """
    Drag to create rectangles on top of the current image.
    Scene coords are image pixel coords (pixmap placed at (0,0) with 1:1 size).
    """
    rectCreated = QtCore.Signal(QtCore.QRect)  # image coords
    itemSelected = QtCore.Signal(object)       # QGraphicsItem or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QtWidgets.QGraphicsScene(self))
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        self._pixmap_item: QtWidgets.QGraphicsPixmapItem | None = None
        self._dragging = False
        self._start = QtCore.QPointF()
        self._rubber: QtWidgets.QGraphicsRectItem | None = None

        # nicer interaction
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)

    def set_image(self, qimg: QtGui.QImage):
        self.scene().clear()
        self.resetTransform()
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_item.setPos(0, 0)
        self._pixmap_item.setZValue(0)

        # set scene rect exactly to image
        self.scene().setSceneRect(QtCore.QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.fitInView(self.scene().sceneRect(), QtCore.Qt.KeepAspectRatio)

    def image_rect(self) -> QtCore.QRect:
        if not self._pixmap_item:
            return QtCore.QRect()
        pm = self._pixmap_item.pixmap()
        return QtCore.QRect(0, 0, pm.width(), pm.height())

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if event.angleDelta().y() == 0:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self._pixmap_item:
            self._dragging = True
            self._start = self.mapToScene(event.pos())
            if self._rubber:
                self.scene().removeItem(self._rubber)
                self._rubber = None

            pen = QtGui.QPen(QtGui.QColor(0, 255, 0))
            pen.setWidth(2)
            self._rubber = self.scene().addRect(QtCore.QRectF(self._start, self._start), pen)
            self._rubber.setZValue(10)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._dragging and self._rubber:
            cur = self.mapToScene(event.pos())
            rectf = QtCore.QRectF(self._start, cur).normalized()
            self._rubber.setRect(rectf)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.LeftButton and self._dragging:
            self._dragging = False
            if not self._rubber:
                return

            rectf = self._rubber.rect().normalized()
            self.scene().removeItem(self._rubber)
            self._rubber = None

            # clamp to image bounds
            img_rect = self.image_rect()
            x1 = int(max(img_rect.left(), min(img_rect.right(), rectf.left())))
            y1 = int(max(img_rect.top(), min(img_rect.bottom(), rectf.top())))
            x2 = int(max(img_rect.left(), min(img_rect.right(), rectf.right())))
            y2 = int(max(img_rect.top(), min(img_rect.bottom(), rectf.bottom())))

            w = abs(x2 - x1)
            h = abs(y2 - y1)
            if w < MIN_RECT_SIZE or h < MIN_RECT_SIZE:
                return

            rect = QtCore.QRect(min(x1, x2), min(y1, y2), w, h)
            self.rectCreated.emit(rect)
            event.accept()
            return

        # selection notification (click existing rect)
        super().mouseReleaseEvent(event)
        item = self.itemAt(event.pos())
        self.itemSelected.emit(item)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, images: list[str], targets: list[str]):
        super().__init__()
        self.setWindowTitle("Region Annotator (Qt)")

        self.images = [str(Path(p)) for p in images]
        self.targets = [str(Path(p)) for p in targets]
        if not self.images:
            raise ValueError("No images provided.")
        if not self.targets:
            raise ValueError("No targets provided.")

        # per-image regions
        self.regions_map: dict[str, list[Region]] = {img: [] for img in self.images}
        self._region_seq = 0

        # current image
        self.cur_idx = 0
        self.cur_qimage: QtGui.QImage | None = None

        # UI
        self.canvas = ImageCanvas()
        self.list_regions = QtWidgets.QListWidget()
        self.list_regions.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.combo_target = QtWidgets.QComboBox()
        self.combo_target.addItems([Path(t).name for t in self.targets])

        self.edit_name = QtWidgets.QLineEdit()
        self.btn_delete = QtWidgets.QPushButton("Delete selected region")
        self.btn_prev = QtWidgets.QPushButton("Prev image")
        self.btn_next = QtWidgets.QPushButton("Next image")
        self.btn_save = QtWidgets.QPushButton("Save JSON + Crops")
        self.chk_basename = QtWidgets.QCheckBox("Save paths as basename only")
        self.chk_basename.setChecked(True)

        # status
        self.lbl_info = QtWidgets.QLabel()
        self.lbl_info.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)

        # layout
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)

        right_layout.addWidget(QtWidgets.QLabel("Regions (max 3 per image)"))
        right_layout.addWidget(self.list_regions, 1)

        form = QtWidgets.QFormLayout()
        form.addRow("Name", self.edit_name)
        form.addRow("Target", self.combo_target)
        right_layout.addLayout(form)

        right_layout.addWidget(self.btn_delete)
        right_layout.addSpacing(12)
        right_layout.addWidget(self.btn_prev)
        right_layout.addWidget(self.btn_next)
        right_layout.addSpacing(12)
        right_layout.addWidget(self.chk_basename)
        right_layout.addWidget(self.btn_save)
        right_layout.addSpacing(12)
        right_layout.addWidget(QtWidgets.QLabel("Info"))
        right_layout.addWidget(self.lbl_info)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # signals
        self.canvas.rectCreated.connect(self.on_rect_created)
        self.canvas.itemSelected.connect(self.on_canvas_item_selected)
        self.list_regions.currentItemChanged.connect(self.on_list_selection_changed)
        self.combo_target.currentIndexChanged.connect(self.on_target_changed)
        self.edit_name.editingFinished.connect(self.on_name_changed)
        self.btn_delete.clicked.connect(self.on_delete_selected)
        self.btn_prev.clicked.connect(lambda: self.switch_image(-1))
        self.btn_next.clicked.connect(lambda: self.switch_image(+1))
        self.btn_save.clicked.connect(self.on_save)

        # disable edits initially
        self.set_region_edit_enabled(False)

        # load first image
        self.load_current_image()

        # shortcuts
        QtGui.QShortcut(QtGui.QKeySequence("["), self, activated=lambda: self.switch_image(-1))
        QtGui.QShortcut(QtGui.QKeySequence("]"), self, activated=lambda: self.switch_image(+1))
        QtGui.QShortcut(QtGui.QKeySequence("Delete"), self, activated=self.on_delete_selected)

    def set_region_edit_enabled(self, enabled: bool):
        self.combo_target.setEnabled(enabled)
        self.edit_name.setEnabled(enabled)
        self.btn_delete.setEnabled(enabled)

    def current_image_path(self) -> str:
        return self.images[self.cur_idx]

    def current_regions(self) -> list[Region]:
        return self.regions_map[self.current_image_path()]

    def load_qimage(self, path: str) -> QtGui.QImage:
        qimg = QtGui.QImage(path)
        if qimg.isNull():
            raise FileNotFoundError(f"Cannot load image: {path}")
        return qimg

    def load_current_image(self):
        img_path = self.current_image_path()
        self.cur_qimage = self.load_qimage(img_path)
        self.canvas.set_image(self.cur_qimage)
        self.rebuild_overlays()
        self.rebuild_list()
        self.update_info()

    def rebuild_overlays(self):
        # draw rect items for current image regions
        for r in self.current_regions():
            self.add_rect_item_for_region(r)

    def add_rect_item_for_region(self, region: Region):
        pen = QtGui.QPen(QtGui.QColor(255, 0, 0))
        pen.setWidth(2)
        item = self.canvas.scene().addRect(QtCore.QRectF(region.rect), pen)
        item.setZValue(20)
        item.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        item.setData(0, region.rid)  # store region id
        region.rect_item = item

    def rebuild_list(self):
        self.list_regions.blockSignals(True)
        self.list_regions.clear()
        for r in self.current_regions():
            it = QtWidgets.QListWidgetItem(
                f"{r.name}  | rect=({r.rect.x()},{r.rect.y()},{r.rect.width()},{r.rect.height()})  | -> {Path(r.target).name}"
            )
            it.setData(QtCore.Qt.UserRole, r.rid)
            self.list_regions.addItem(it)
        self.list_regions.blockSignals(False)

        self.set_region_edit_enabled(False)

    def update_info(self):
        img_path = Path(self.current_image_path())
        count = len(self.current_regions())
        self.lbl_info.setText(
            f"Image: {img_path.name} ({self.cur_idx+1}/{len(self.images)})\n"
            f"Regions: {count}/{MAX_REGIONS_PER_IMAGE}\n"
            f"Drag LMB on image to create a region.\n"
            f"Shortcuts: [ / ] switch image, Delete removes selected region."
        )

    def find_region_by_id(self, rid: str) -> Region | None:
        for r in self.current_regions():
            if r.rid == rid:
                return r
        return None

    def selected_region(self) -> Region | None:
        it = self.list_regions.currentItem()
        if not it:
            return None
        rid = it.data(QtCore.Qt.UserRole)
        return self.find_region_by_id(rid)

    def on_rect_created(self, rect: QtCore.QRect):
        regions = self.current_regions()
        if len(regions) >= MAX_REGIONS_PER_IMAGE:
            QtWidgets.QMessageBox.warning(self, "Limit reached", f"Each image can have at most {MAX_REGIONS_PER_IMAGE} regions.")
            return

        self._region_seq += 1
        rid = f"{Path(self.current_image_path()).stem}_{self._region_seq}"
        name = f"r{len(regions)+1}_{rect.x()}_{rect.y()}_{rect.width()}_{rect.height()}"
        target = self.targets[0]

        r = Region(rid=rid, name=name, rect=rect, target=target)
        regions.append(r)

        self.add_rect_item_for_region(r)
        self.rebuild_list()

        # auto-select the newly added one
        for i in range(self.list_regions.count()):
            if self.list_regions.item(i).data(QtCore.Qt.UserRole) == rid:
                self.list_regions.setCurrentRow(i)
                break

        self.update_info()

    def on_canvas_item_selected(self, item):
        # if user clicked on a rectangle, sync selection to list
        if item is None:
            return
        rid = item.data(0)
        if not rid:
            return
        for i in range(self.list_regions.count()):
            if self.list_regions.item(i).data(QtCore.Qt.UserRole) == rid:
                self.list_regions.setCurrentRow(i)
                break

    def on_list_selection_changed(self, cur, prev):
        r = self.selected_region()
        if not r:
            self.set_region_edit_enabled(False)
            return

        # mark rect selected
        if r.rect_item:
            r.rect_item.setSelected(True)

        # update edit widgets
        self.set_region_edit_enabled(True)
        self.edit_name.setText(r.name)

        # find target index by full path string match
        try:
            idx = self.targets.index(r.target)
        except ValueError:
            idx = 0
            r.target = self.targets[0]
        self.combo_target.blockSignals(True)
        self.combo_target.setCurrentIndex(idx)
        self.combo_target.blockSignals(False)

    def on_target_changed(self, idx: int):
        r = self.selected_region()
        if not r:
            return
        if 0 <= idx < len(self.targets):
            r.target = self.targets[idx]
            self.rebuild_list()

    def on_name_changed(self):
        r = self.selected_region()
        if not r:
            return
        new_name = self.edit_name.text().strip()
        if new_name:
            r.name = new_name
            self.rebuild_list()

    def on_delete_selected(self):
        r = self.selected_region()
        if not r:
            return

        # remove graphics item
        if r.rect_item:
            self.canvas.scene().removeItem(r.rect_item)
            r.rect_item = None

        # remove from data
        regions = self.current_regions()
        regions[:] = [x for x in regions if x.rid != r.rid]

        self.rebuild_list()
        self.update_info()

    def switch_image(self, step: int):
        self.cur_idx = (self.cur_idx + step) % len(self.images)
        self.load_current_image()

    def build_regions_map_for_save(self):
        """
        Output JSON structure:
        {
          "base.png": [ {"name":..., "rect":[x,y,w,h], "target":"a.png"}, ... ],
          "a.png":    [ ... ]
        }
        """
        use_basename = self.chk_basename.isChecked()

        def norm(p: str) -> str:
            return Path(p).name if use_basename else str(Path(p))

        out = {}
        for img_path, regions in self.regions_map.items():
            out[norm(img_path)] = []
            for r in regions:
                out[norm(img_path)].append({
                    "name": r.name,
                    "rect": [r.rect.x(), r.rect.y(), r.rect.width(), r.rect.height()],
                    "target": norm(r.target),
                })
        return out

    def export_crops(self, crop_dir: Path):
        crop_dir.mkdir(parents=True, exist_ok=True)
        for img_path, regions in self.regions_map.items():
            qimg = self.load_qimage(img_path)
            stem_dir = crop_dir / Path(img_path).stem
            stem_dir.mkdir(parents=True, exist_ok=True)

            for r in regions:
                rect = r.rect.intersected(QtCore.QRect(0, 0, qimg.width(), qimg.height()))
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
                crop = qimg.copy(rect)
                out_path = stem_dir / f"{r.name}.png"
                crop.save(str(out_path), "PNG")

    def on_save(self):
        # choose json path
        json_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save regions_map.json", "regions_map.json", "JSON (*.json)"
        )
        if not json_path:
            return

        crop_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select crop output directory", "crops")
        if not crop_dir:
            return

        # save json
        data = self.build_regions_map_for_save()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # save crops
        self.export_crops(Path(crop_dir))

        QtWidgets.QMessageBox.information(self, "Saved", f"Saved:\n- {json_path}\n- crops -> {crop_dir}")


def find_image_files(folder):
    """Find all image files in the specified folder."""
    extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                # Use relative path to the screenshots folder
                rel_path = os.path.relpath(os.path.join(root, file), os.path.dirname(os.path.abspath(__file__)))
                # Convert to Windows-style path if needed
                rel_path = rel_path.replace('\\', '\\\\')
                image_files.append(rel_path)
    return sorted(image_files)

def main():
    ap = argparse.ArgumentParser(description='Image Region Annotator')
    ap.add_argument('--screenshots', default='screenshots', 
                   help='Path to the screenshots folder (default: screenshots)')
    args = ap.parse_args()
    
    # Find all images in the screenshots folder
    screenshots_dir = args.screenshots
    if not os.path.exists(screenshots_dir):
        print(f"Error: Screenshots directory '{screenshots_dir}' not found")
        return 1
        
    image_files = find_image_files(screenshots_dir)
    
    if not image_files:
        print(f"No image files found in {screenshots_dir}")
        return 1
        
    print(f"Found {len(image_files)} images in {screenshots_dir}")
    
    app = QtWidgets.QApplication(sys.argv)
    # Use the same list for both images and targets
    w = MainWindow(image_files, image_files)
    w.resize(1200, 800)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
#python flow_setting.py --screenshots screenshots
