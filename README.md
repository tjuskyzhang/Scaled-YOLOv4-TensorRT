# yolov4-tiny-tensorrt
yolov4-tiny-tensorrt

This project is based on https://github.com/wang-xinyu/tensorrtx/tree/trt4 and https://github.com/WongKinYiu/PyTorch_YOLOv4

(1) Generate yolov4-tiny.wts from pytorch implementation
git clone https://github.com/WongKinYiu/PyTorch_YOLOv4.git
// download yolov4-tiny.pt and copy it into PyTorch_YOLOv4/weights

git clone https://github.com/tjuskyzhang/yolov4-tiny-tensorrt.git

cd PyTorch_YOLOv4

cp ../yolov4-tiny-tensorrt/gen_wts.py .

python gen_wts.py weights/yolov4-tiny.pt

// a file 'yolov4-tiny.wts' will be generated

