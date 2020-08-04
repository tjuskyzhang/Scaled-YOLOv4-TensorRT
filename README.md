# yolov4-tiny-tensorrt
yolov4-tiny-tensorrt


This project is based on https://github.com/wang-xinyu/tensorrtx/tree/trt4 and https://github.com/WongKinYiu/PyTorch_YOLOv4

This project has been tested on cuda 9.0, cudnn 7.5, tensorrt 5.1.5


(1) Generate yolov4-tiny.wts from pytorch implementation

git clone https://github.com/WongKinYiu/PyTorch_YOLOv4.git

// download yolov4-tiny.pt and copy it into PyTorch_YOLOv4/weights

// 链接：https://pan.baidu.com/s/1lEXCyDJyjL9B0WR-MKzAeg 

// 提取码：ml0o 

git clone https://github.com/tjuskyzhang/yolov4-tiny-tensorrt.git

cd PyTorch_YOLOv4

cp ../yolov4-tiny-tensorrt/gen_wts.py .

python gen_wts.py weights/yolov4-tiny.pt

// a file 'yolov4-tiny.wts' will be generated

cp yolov4-tiny.wts ../yolov4-tiny-tensorrt

(2) Build and run

cd yolov4-tiny-tensorrt

mkdir build

cd build

cmake ..

make

./yolov4 -s

// serialize the model and generate yolov4-tiny.engine

./yolov4-tiny -d ../samples

// deserialize and generate the detection results _dog.jpg
