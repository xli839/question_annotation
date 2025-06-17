from ultralytics import YOLO

model_name = '/hpc2hdd/home/xli839/lxy/work/yolo-main/pre_model/yolov8s.pt'
data_name ='/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/data_multi.yaml'

# Load a model
model = YOLO(model_name)  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data=data_name, epochs=100, imgsz=640, device=[0])