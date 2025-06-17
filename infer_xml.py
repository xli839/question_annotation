from ultralytics import YOLO
import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import pytesseract  # 添加OCR支持
import re

# 配置路径
model_path = '/hpc2hdd/home/xli839/lxy/work/yolo-main/runs/detect/train/weights/best.pt'
# image_folder = '/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/images/val'
image_folder = '/hpc2hdd/home/xli839/lxy/work/data/data/英语'
output_folder = '/hpc2hdd/home/xli839/lxy/work/yolo-main/dataset/xml_output'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 加载模型
model = YOLO(model_path)

def extract_page_number(img):
    """
    从图片中提取页码，只使用图片底部1/5部分
    """
    # 获取图片高度
    height = img.shape[0]
    # 计算底部1/5部分的起始位置
    start_y = int(height * 0.9)
    
    # 只保留底部1/5部分
    bottom_part = img[start_y:height, :]
    
    # 转换为灰度图
    gray = cv2.cvtColor(bottom_part, cv2.COLOR_BGR2GRAY)
    
    # 使用OCR识别文字

    text = pytesseract.image_to_string(gray, lang='eng')
    print(text)
    
    # 使用正则表达式查找页码
    # 假设页码格式为纯数字
    numbers = re.findall(r'\b\d+\b', text)
    
    # 返回找到的最后一个数字作为页码
    print(numbers)
    return int(numbers[-1]) if numbers else 1

def create_xml(img_name, img_path, boxes, page_number):
    # 读取图片获取尺寸
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    
    # 创建XML根元素
    root = ET.Element("annotation")
    
    # 添加基本信息
    folder = ET.SubElement(root, "folder")
    # folder.text = "数学-人教版-三年级下册-53天天练(春)(2025)-题目"
    folder.text = "英语"
    
    filename = ET.SubElement(root, "filename")
    filename.text = img_name  # 使用原始图片名称
    
    path = ET.SubElement(root, "path")
    path.text = img_path
    
    source = ET.SubElement(root, "source")
    database = ET.SubElement(source, "database")
    database.text = "Unknown"
    
    size = ET.SubElement(root, "size")
    width = ET.SubElement(size, "width")
    width.text = str(img_w)
    height = ET.SubElement(size, "height")
    height.text = str(img_h)
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(root, "segmented")
    segmented.text = "0"
    
    # 添加检测框，每个检测框命名为"页码-序号"
    for idx, box in enumerate(boxes, 1):
        obj = ET.SubElement(root, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = f"{page_number}-{idx}"  # 使用"页码-序号"格式
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(box["bbox"][0]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(box["bbox"][1]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(box["bbox"][2]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(box["bbox"][3]))
    
    # 美化XML输出
    xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
    
    # 保存XML文件，使用原始图片名称
    xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + '.xml')
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(xmlstr)

# 遍历图片进行预测
for img_name in os.listdir(image_folder):
    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue
    
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    
    # 从文件名提取页码
    page_number = extract_page_number(img)
    
    result = model(img_path)[0]  # 获取第一个结果（单张图）
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

        boxes.append({
            "class_id": cls_id,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })
    
    # 根据y坐标对boxes进行排序，确保题目按顺序编号
    boxes.sort(key=lambda x: x["bbox"][1])

    # 为每张图片创建一个XML文件，包含所有检测到的题目
    create_xml(img_name, img_path, boxes, page_number)