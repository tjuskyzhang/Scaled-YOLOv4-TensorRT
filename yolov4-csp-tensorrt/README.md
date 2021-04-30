# README

## yolov4-csp-tensorrt:
* This project is based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4/tree/yolov4-csp). The project has been tested on TensorRT 7.0 CUDA 10.2 CUDNN 7.6.5, and costs about 20ms(50fps) to inference an image on GeForce GTX 1660 Ti.

## Excute:

(1) Generate yolov4-csp.wts from pytorch implementation

```
  git clone -b yolov4-csp https://github.com/WongKinYiu/ScaledYOLOv4.git
```
Install and test ScaledYOLOv4 before generate the yolov4-csp.wts.

// Download [yolov4-csp.weights](https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing) and copy it into ScaledYOLOv4/weights.

// 权重下载链接：https://pan.baidu.com/s/1OevCSTqOPMO2tgJ3qDQpOA 提取码：rit7 

```
  git clone -b https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT.git

  cd ScaledYOLOv4

  cp ../Scaled-YOLOv4-TensorRT/yolov4-csp-tensorrt/gen_wts.py .

  python gen_wts.py weights/yolov4-csp.weights
```
// A file named 'yolov4-csp.wts' will be generated.

```
  cp yolov4-csp.wts ../Scaled-YOLOv4-TensorRT/yolov4-csp-tensorrt
```

(2) Build and run

```
  cd Scaled-YOLOv4-TensorRT/yolov4-csp-tensorrt

  mkdir build

  cd build

  cmake ..

  make
```
// Serialize the model and generate yolov4-csp.engine
```
  ./yolov4-csp -s
```

// Deserialize and generate the detection results _dog.jpg and so on.
```
  ./yolov4-csp -d ../samples
```
