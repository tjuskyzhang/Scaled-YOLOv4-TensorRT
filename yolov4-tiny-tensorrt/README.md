# README

## yolov4-tiny-tensorrt:
* This project is based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4/tree/master). The project has been tested on TensorRT 7.0 CUDA 10.2 CUDNN 7.6.5, and costs about 2ms(500fps) to inference an image on GeForce GTX 1660 Ti.

* The project also has been tested on TensorRT 7.1.0(Developer Preview) CUDA 10.2 CUDNN 8.0.0(Developer Preview), and costs about 10-11ms(90-100fps) to inference an image on TX2 (by using the MAX-N mode and jetson_clocks).

## Excute:

(1) Generate yolov4-tiny.wts from pytorch implementation

```
  git clone -b u3_preview https://github.com/WongKinYiu/PyTorch_YOLOv4.git
```

// Download [yolov4-tiny.pt](https://drive.google.com/file/d/1aQKcCvTAl1uOWzzHVE9Z8Ixgikc3AuYQ/view?usp=sharing) and copy it into PyTorch_YOLOv4/weights.

// 权重下载链接：https://pan.baidu.com/s/1lEXCyDJyjL9B0WR-MKzAeg 提取码：ml0o 

```
  git clone https://github.com/tjuskyzhang/Scaled-YOLOv4-TensorRT.git

  cd PyTorch_YOLOv4

  cp ../Scaled-YOLOv4-TensorRT/yolov4-tiny-tensorrt/gen_wts.py .

  python gen_wts.py weights/yolov4-tiny.pt
```
// A file named 'yolov4-tiny.wts' will be generated.

```
  cp yolov4-tiny.wts ../Scaled-YOLOv4-TensorRT/yolov4-tiny-tensorrt
```

(2) Build and run

```
  cd Scaled-YOLOv4-TensorRT/yolov4-tiny-tensorrt

  mkdir build

  cd build

  cmake ..

  make
```
// Serialize the model and generate yolov4-tiny.engine
```
  ./yolov4-tiny -s
```

// Deserialize and generate the detection results _dog.jpg and so on.

```
  ./yolov4-tiny -d ../samples
```
