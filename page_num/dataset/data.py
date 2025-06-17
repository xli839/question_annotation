import os
import json
import random
from PIL import Image
from pathlib import Path
import shutil

# === 配置 ===
json_path = '/hpc2hdd/home/xli839/lxy/work/page_num/dataset/dataset_info.json'  # JSON 路径
base_dir = '/hpc2hdd/home/xli839/lxy/work/page_num/dataset/tesstrain_data'
train_dir = os.path.join(base_dir, 'train', 'ground-truth')
test_img_dir = os.path.join(base_dir, 'test', 'images')
test_label_path = os.path.join(base_dir, 'test', 'labels.csv')

# === 创建目录 ===
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_img_dir, exist_ok=True)

# === 读取并划分数据 ===
with open(json_path, 'r') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)
split_idx = int(len(data) * 0.8)
train_data = data[:split_idx]
test_data = data[split_idx:]

def process_and_save(item, is_train=True):
    image_path = item["image_path"]
    page_number = str(item["page_number"]).strip()
    img_name = Path(image_path).stem

    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        crop_box = (0, int(h * 4 / 5), w, h)
        cropped = img.crop(crop_box)

        if is_train:
            save_path = os.path.join(train_dir, f"{img_name}.tif")
            cropped.save(save_path)
            with open(os.path.join(train_dir, f"{img_name}.gt.txt"), 'w') as f:
                f.write(page_number)
        else:
            save_path = os.path.join(test_img_dir, f"{img_name}.tif")
            cropped.save(save_path)
            return img_name, page_number
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# === 处理训练集 ===
for item in train_data:
    process_and_save(item, is_train=True)

# === 处理测试集并生成标签CSV ===
with open(test_label_path, 'w') as f:
    f.write("filename,page_number\n")
    for item in test_data:
        result = process_and_save(item, is_train=False)
        if result:
            f.write(f"{result[0]},{result[1]}\n")

print("✅ 数据划分与处理完成")