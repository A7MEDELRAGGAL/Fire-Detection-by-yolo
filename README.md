# Fire-Detection-by-yolo
YOLOv8-based fire detection project for real-time fire alerts and monitoring. Utilizes a custom-trained model for detecting fire in video feeds or images, making it suitable for safety and monitoring applications.



<h1 align="center"><span>YOLOv5/YOLOv8 for Fire Detection</span></h1>

Fire detection task aims to identify fire or flame in a video and put a bounding box around it. This repo includes a demo on how to build a fire detector using YOLOv8. 

<p align="center">
  <img src="runs/detect/RPReplay_Final1730157467.gif" />
</p>

## First
1. Installing the packages
``` shell
# Installing the packages
!pip install ultralytics
!pip install opencv-python
!pip install ultralytics
!pip install roboflow
```

2. Import YOLO model utilities from Ultralytics and Roboflow for model use, and Image for displaying images in Jupyter
``` shell
from roboflow import Roboflow
from ultralytics import YOLO
from IPython.display import Image
```

3. Connect to Roboflow with API key, access the "continuous_fire" project, and download version 6 of the dataset for YOLOv8
``` shell
git clone https://github.com/WongKinYiu/yolov9.git
cd yolov9
pip install -r requirements.txt
```

## Connect to Roboflow with API key, access the "continuous_fire" project, and download version 6 of the dataset for YOLOv8
``` shell
from roboflow import Roboflow
rf = Roboflow(api_key="Hm8mdFJnlkZnoorR3Uy8")
project = rf.workspace("-jwzpw").project("continuous_fire")
dataset = project.version(6).download("yolov8")
```

- Train YOLOv8 model for fire detection with specified parameters (image size, epochs, batch size, and model name)
```
!yolo task=detect mode=train model=yolov8n.pt data=/content/continuous_fire-6/data.yaml imgsz=640 epochs=10 batch=10 name=fire_detection_model
```

- Validate the trained YOLOv8 model for fire detection using the best weights and specified dataset
```
!yolo task=detect mode=val model=/content/runs/detect/fire_detection_model/weights/best.pt data=/content/continuous_fire-6/data.yaml
```

## predict

- Predict from your computer camera
``` shell
#model.predict(source=0, save=True,conf=0.5,show=True)
```


- Predict from your computer source

``` shell
model.predict(source=r'WhatsApp.mp4', save=False, conf=0.5, show=True)    /Replace WhatsApp.mp4 with the path of your file
```

You can download the pretrained yolov9-c.pt model from [google drive](https://drive.google.com/file/d/1nV5C3dbc_Q3CoczHaERTojr78-SFPdMI/view?usp=sharing) for fire detection. Note that this model was trained on the fire dataset for 50 epochs. Refer to [link](https://github.com/WongKinYiu/yolov9/issues/162) to fix for detect.py runtime error when running yolov9.

## ⏱️ Results
The following charts were produced after training YOLOv5s with input size 640x640 on the fire dataset for 10 epochs.

| P Curve | PR Curve | R Curve |
| :-: | :-: | :-: |
| ![](results/P_curve.png) | ![](results/PR_curve.png) | ![](results/R_curve.png) |

#### Prediction Results
The fire detection results were fairly good even though the model was trained only for a few epochs. However, I observed that the trained model tends to predict red emergency light on top of police car as fire. It might be due to the fact that the training dataset contains only a few hundreds of negative samples. We may fix such problem and further improve the performance of the model by adding images with non-labeled fire objects as negative samples. The [authors](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results) who created YOLOv5 recommend using about 0-10% background images to help reduce false positives. 

| Ground Truth | Prediction | 
| :-: | :-: |
| ![](results/val_batch2_labels_1.jpg) | ![](results/val_batch2_pred_1.jpg) |
| ![](results/val_batch2_labels_2.jpg) | ![](results/val_batch2_pred_2.jpg) | 

#### Feature Visualization
It is desirable for AI engineers to know what happens under the hood of object detection models. Visualizing features in deep learning models can help us a little bit understand how they make predictions. In YOLOv5, we can visualize features using ```--visualize``` argument as follows:

```
python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.2 --source ../datasets/fire/val/images/0.jpg --visualize
```

| Input | Feature Maps | 
| :-: | :-: |
| ![](results/004dec94c5de631f.jpg) | ![](results/stage23_C3_features.png) |

## 🔗 Reference
I borrowed and modified [YOLOv5-Custom-Training.ipynb](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) script for training YOLOv5 model on the fire dataset. For more information on training YOLOv5, please refer to its homepage.
* https://github.com/robmarkcole/fire-detection-from-images
* https://github.com/ultralytics/yolov5
* https://github.com/AlexeyAB/darknet
