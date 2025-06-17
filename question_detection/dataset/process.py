import os
import shutil
import glob
import random
import xml.etree.ElementTree as ET
from PIL import Image

# 参数
SOURCE_DIRS = ['/hpc2hdd/home/xli839/lxy/work/data/data/chinese', '/hpc2hdd/home/xli839/lxy/work/data/data/math']
TARGET_DIR = 'dataset'
LABEL_CLASS = 0  # 统一使用一个类别编号

# 创建目标目录
for split in ['train', 'val']:
    os.makedirs(os.path.join(TARGET_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DIR, 'labels', split), exist_ok=True)

# 获取所有图像-xml对
samples = []
for folder in SOURCE_DIRS:
    imgs = glob.glob(os.path.join(folder, '*.jpg'))  # 你可以改为*.png等
    for img_path in imgs:
        xml_path = os.path.splitext(img_path)[0] + '.xml'
        if os.path.exists(xml_path):
            samples.append((img_path, xml_path))

# 打乱并划分
random.shuffle(samples)
split_idx = int(0.8 * len(samples))
train_samples = samples[:split_idx]
val_samples = samples[split_idx:]

def convert_xml_to_yolo(xml_path, img_size):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    w, h = img_size
    yolo_labels = []

    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        box_width = (xmax - xmin) / w
        box_height = (ymax - ymin) / h

        yolo_labels.append(f"{LABEL_CLASS} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    return yolo_labels

def process_dataset(samples, split):
    for img_path, xml_path in samples:
        img = Image.open(img_path)
        w, h = img.size
        filename = os.path.basename(img_path)
        label_filename = os.path.splitext(filename)[0] + '.txt'

        # 拷贝图像
        shutil.copy(img_path, os.path.join(TARGET_DIR, 'images', split, filename))

        # 写入标签
        yolo_labels = convert_xml_to_yolo(xml_path, (w, h))
        with open(os.path.join(TARGET_DIR, 'labels', split, label_filename), 'w') as f:
            f.write('\n'.join(yolo_labels))

process_dataset(train_samples, 'train')
process_dataset(val_samples, 'val')

# 写data.yaml文件
yaml_content = f"""\
path: {TARGET_DIR}
train: images/train
val: images/val

names:
  0: question_number
"""

with open(os.path.join(TARGET_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print("✅ 数据处理完毕，可用于YOLOv8训练！")