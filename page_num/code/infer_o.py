import pytesseract
from PIL import Image
import os
import pandas as pd
import re

test_dir = '/hpc2hdd/home/xli839/lxy/work/page_num/dataset/tesstrain_data/test/images'
label_file = '/hpc2hdd/home/xli839/lxy/work/page_num/dataset/tesstrain_data/test/labels.csv'
use_model = 'eng'  # 使用系统自带模型

df = pd.read_csv(label_file)
correct = 0

for _, row in df.iterrows():
    filename, gt = row['filename'], str(row['page_number']).strip()
    img_path = os.path.join(test_dir, f"{filename}.tif")
    text = pytesseract.image_to_string(Image.open(img_path), lang=use_model, config='--psm 6').strip()
    print(text)

    # 提取其中的数字（可以用正则优化）
    # 用正则提取所有连续数字
    numbers = re.findall(r'\b\d+\b', text)
    
    # 返回找到的最后一个数字作为页码
    print(numbers)

    # 取最后一个数字作为结果
    pred = int(numbers[-1]) if numbers else 1


    if int(pred) == int(gt):
        correct += 1
    else:
        print(f"[错] {filename}: GT={gt}, Pred={pred}")

print(f"✅ 微调前测试准确率: {correct}/{len(df)} = {correct / len(df):.2%}")