from ultralytics import YOLO
import cv2
import os
import json
import time  # 添加time模块

# 配置路径
model_path = '/hpc2hdd/home/xli839/lxy/work/yolo-main/runs/detect/train/weights/best.pt'
# image_folder = '/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/images/val'
image_folder = '/hpc2hdd/home/xli839/lxy/work/data/data/英语'
output_json = '/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/output_boxes_2.json'
output_images_folder = '/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/annotated_val'  # 新增输出图片文件夹

# 创建输出图片文件夹
os.makedirs(output_images_folder, exist_ok=True)

# 加载模型
model = YOLO(model_path)

# 用于保存结果
results_dict = {}
total_preprocess_time = 0
total_inference_time = 0
image_count = 0

# 遍历图片进行预测
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    img_path = os.path.join(image_folder, img_name)
    # 测量预处理时间
    preprocess_start = time.time()
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    preprocess_time = time.time() - preprocess_start
    total_preprocess_time += preprocess_time
    
    # 测量推理时间
    inference_start = time.time()
    result = model(img_path)[0]  # 获取第一个结果（单张图）
    inference_time = time.time() - inference_start
    total_inference_time += inference_time
    
    image_count += 1
    

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    boxes = []
    for box in result.boxes:
        # 取出 YOLO 格式 (cx, cy, w, h)
        cx, cy, w, h = box.xywh[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])

        # 转为 (x1, y1, x2, y2)
        x1 = max(0, cx - w / 2)
        y1 = max(0, cy - h / 2)
        x2 = min(img_w, cx + w / 2)
        y2 = min(img_h, cy + h / 2)

        # 在图片上绘制检测框
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 添加类别和置信度标签
        label = f'Class {cls_id}: {conf:.2f}'
        cv2.putText(img, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        boxes.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    # 保存带有检测框的图片
    output_img_path = os.path.join(output_images_folder, img_name)
    cv2.imwrite(output_img_path, img)

    results_dict[img_name] = boxes

# 在保存JSON之前打印性能统计信息
print("\n性能统计：")
print(f"处理的图片总数: {image_count}")
print(f"平均预处理时间: {total_preprocess_time/image_count:.4f} 秒")
print(f"平均推理时间: {total_inference_time/image_count:.4f} 秒")
print(f"平均总处理时间: {(total_preprocess_time + total_inference_time)/image_count:.4f} 秒")


# 保存到 JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(results_dict, f, indent=4)