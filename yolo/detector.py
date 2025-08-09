from ultralytics import YOLO
from PIL import Image
import sys, os
from typing import Optional, List


class Detector:
  def __init__(self, model_path = "yolov8n.pt", device = 'cpu', half = True, imgsz = 416, conf = 0.15, iou = 0.5, max_det = 50):
    self.model = YOLO(model_path)
    self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device)
    if half: self.kw["half"] = True
  def predict(self, img : Image.Image):
    return self.model.predict(source = img, **self.kw)[0]
  @property
  def names(self) -> dict:
    return self.model.names  # {id: "label"}

  def predict(self, img: Image.Image, classes: Optional[List[int]] = None):
    return self.model.predict(source=img, classes=classes, **self.kw)[0]