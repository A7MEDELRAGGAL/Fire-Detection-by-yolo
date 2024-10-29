import torch
import cv2
from ultralytics import YOLO

model = YOLO("best.pt")
#model.predict(source=0, save=True,conf=0.5,show=True)
model.predict(source=r'tests\RPReplay_Final1730157467.mov', save=False, conf=0.5, show=True)