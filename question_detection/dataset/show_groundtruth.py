import os
import cv2
import numpy as np
from pathlib import Path

def draw_annotations(image_path, label_path, output_dir, class_names=None):
    """
    在图片上绘制标注框并保存
    
    Args:
        image_path: 图片路径
        label_path: 标注文件路径
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"无法读取图片: {image_path}")
        return
    
    height, width = img.shape[:2]
    
    # 读取标注文件
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
    except:
        print(f"无法读取标注文件: {label_path}")
        return
    
    # 为每个类别生成不同的颜色
    colors = {}
    for i in range(100):  # 假设最多100个类别
        colors[i] = tuple(np.random.randint(0, 255, 3).tolist())
    
    # 绘制每个标注框
    for line in lines:
        data = line.strip().split()
        if len(data) != 5:
            continue
            
        class_id = int(data[0])
        x_center, y_center, w, h = map(float, data[1:])
        
        # 转换归一化坐标到像素坐标
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # 绘制矩形框
        color = colors[class_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # 添加类别标签
        label = class_names[class_id] if class_names else str(class_id)
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1-label_h-4), (x1+label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图片
    output_path = os.path.join(output_dir, Path(image_path).name)
    cv2.imwrite(output_path, img)
    print(f"已保存标注图片: {output_path}")

def process_validation_set(dataset_dir, output_dir, class_names=None):
    """
    处理整个验证集
    
    Args:
        dataset_dir: 数据集根目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    # 验证集图片和标注目录
    images_dir = os.path.join(dataset_dir, 'images', 'val')
    labels_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    # 确保目录存在
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print("验证集目录不存在")
        return
    
    # 处理每张图片
    for img_file in os.listdir(images_dir):
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(images_dir, img_file)
        label_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + '.txt')
        
        if os.path.exists(label_path):
            draw_annotations(img_path, label_path, output_dir, class_names)

# 使用示例
if __name__ == "__main__":
    # 设置数据集目录和输出目录
    dataset_dir = "/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset"  # 数据集根目录
    output_dir = "yolo-main/dataset/annotated_val"  # 输出目录
    
    
    class_names = ["q"] 
    
    # 处理验证集
    process_validation_set(dataset_dir, output_dir, class_names)
