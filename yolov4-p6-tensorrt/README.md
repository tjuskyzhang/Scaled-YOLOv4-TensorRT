# README

## yolov4-p6-tensorrt:
* This project is based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-large). The project has been tested on TensorRT 7.0 CUDA 10.2 CUDNN 7.6.5, and costs about 120ms to inference an image on GeForce GTX 1660 Ti.

## Excute:

(1) Generate yolov4-p6.wts from pytorch implementation

```
  git clone -b yolov4-large https://github.com/WongKinYiu/ScaledYOLOv4.git
```
Install and test ScaledYOLOv4 before generate the yolov4-p6.wts.

// Download [yolov4-p6.pt](https://drive.google.com/file/d/1aB7May8oPYzBqbgwYSZHuATPXyxh9xnf/view?usp=sharing) and copy it into ScaledYOLOv4/weights.

// 权重下载链接：https://pan.baidu.com/s/1BHQ3lrd-GdNmQhsffZW59A 提取码：on6r
```
  git clone https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT.git

  cd ScaledYOLOv4

  cp ../Scaled-YOLOv4-TensorRT/yolov4-p6-tensorrt/gen_wts.py .

  python gen_wts.py weights/yolov4-p6.weights
```
// A file named 'yolov4-p6.wts' will be generated.

```
  cp yolov4-p6.wts ../Scaled-YOLOv4-TensorRT/yolov4-p6-tensorrt
```

(2) Build and run

```
  cd Scaled-YOLOv4-TensorRT/yolov4-p6-tensorrt

  mkdir build

  cd build

  cmake ..

  make
```
// Serialize the model and generate yolov4-p6.engine

// Set depth_multiple: 1.0, width_multiple: 1.0 according to yolov4-p6.yaml
```
  ./yolov4-p6 -s 1.0 1.0
```

// Deserialize and generate the detection results _dog.jpg and so on.
```
  ./yolov4-p6 -d ../samples
```
