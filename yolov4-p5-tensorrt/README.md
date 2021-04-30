# README

## yolov4-p5-tensorrt:
* This project is based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large). The project has been tested on TensorRT 7.0 CUDA 10.2 CUDNN 7.6.5, and costs about 60ms to inference an image on GeForce GTX 1660 Ti.

## Excute:

(1) Generate yolov4-p5.wts from pytorch implementation

```
  git clone -b yolov4-large https://github.com/WongKinYiu/ScaledYOLOv4.git
```
Install and test ScaledYOLOv4 before generate the yolov4-p5.wts.

// Download [yolov4-p5.pt](https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing) and copy it into ScaledYOLOv4/weights.

// 权重下载链接：https://pan.baidu.com/s/16iJlraEu5gUlq55FhQK7IA 提取码：h854
```
  git clone https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT.git

  cd ScaledYOLOv4

  cp ../Scaled-YOLOv4-TensorRT/yolov4-p5-tensorrt/gen_wts.py .

  python gen_wts.py weights/yolov4-p5.weights
```
// A file named 'yolov4-p5.wts' will be generated.

```
  cp yolov4-p5.wts ../Scaled-YOLOv4-TensorRT/yolov4-p5-tensorrt
```

(2) Build and run

```
  cd Scaled-YOLOv4-TensorRT/yolov4-p5-tensorrt

  mkdir build

  cd build

  cmake ..

  make
```
// Serialize the model and generate yolov4-p5.engine

// Set depth_multiple: 1.0, width_multiple: 1.0 according to yolov4-p5.yaml
```
  ./yolov4-p5 -s 1.0 1.0
```

// Deserialize and generate the detection results _dog.jpg and so on.
```
  ./yolov4-p5 -d ../samples
```
