# object_detection_yolov8n_kitti
This project explores object detection in autonomous driving using the [KITTI](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) (Karlsruhe Institute of Technology and Toyota Technological Institute) dataset . It is 5.79 GB with 7,481 labeled images in total. The dataset has 9 classes: Car, Cyclist, Pedestrian, Truck, Van, Tram, People_sitting, Misc, and DontCare. This work used [YOLOv8n](https://github.com/ultralytics/ultralytics) (You Only Look Once Version 8 Nano). To process the data, we first converted the data labels into the YOLO format, which is a format that can be recognized by the YOLO model that we are using for object detection. We then splitted the dataset into 80% training, 10% validation, and 10% test sets.

## Installation
Use the following packages:
* PyTorch
* Ultralytics (For the YOLOv8n model)


