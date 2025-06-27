import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

# ==== 设置路径 ====
pred_dir = "../output_folder/math"
gt_dir = '../dataset/数学/数学-北师大版-二年级下册-同步伴读综合练习(春)(2025)题目'
image_dir = gt_dir  # 图片和gt同目录
save_dir = "../output_folder/vis_output/math"
os.makedirs(save_dir, exist_ok=True)

# ==== IoU 计算 ====
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = inter / float(areaA + areaB - inter + 1e-6)
    return iou

# ==== 解析正确文档的name ====
def load_boxes(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    page_id = None
    for idx, obj in enumerate(root.findall("object")):
        name = obj.find("name").text
        if page_id is None:
            page_id = name.split("-")[0]
        bndbox = obj.find("bndbox")
        box = [int(bndbox.find("xmin").text),
               int(bndbox.find("ymin").text),
               int(bndbox.find("xmax").text),
               int(bndbox.find("ymax").text)]
        
        boxes.append((name, box))
    return boxes



# ==== 主循环 ====
all_TP, all_FP, all_FN = 0, 0, 0
# 新增统计变量
total_q_items = 0
correct_page_items = 0

xml_list = sorted(glob(os.path.join(gt_dir, "*.xml")))
num = 0
p_num = 0
for gt_xml in tqdm(xml_list):
    num += 1
    basename = os.path.basename(gt_xml)
    pred_xml = os.path.join(pred_dir, basename)
    img_path = os.path.join(image_dir, basename.replace(".xml", ".jpg"))
    if not os.path.exists(pred_xml) or not os.path.exists(img_path):
        continue

    gt_boxes = load_boxes(gt_xml)
    pred_boxes = load_boxes(pred_xml)

    print(gt_boxes)
    print(pred_boxes)
    total_q_items += len(gt_boxes)
    # print(o)

    gt_used = set()
    TP = 0
    

    # 匹配预测框与GT框
    for pname, pbox in pred_boxes:
        matched = False
        
        for i, (gname, gbox) in enumerate(gt_boxes):
            try:
                pname = f'{pname.split("--")[0]}-{pname.split("--")[1]}'
                print(pname)
            except:
                pass

            if f'{pname.split("-")[1]}-{pname.split("-")[2]}' == f'{gname.split("-")[1]}-{gname.split("-")[2]}':
                p_num += 1

            iou = compute_iou(pbox, gbox)
            if iou >= 0.5 and i not in gt_used:
                TP += 1
                gt_used.add(i)
                matched = True
                break
        if not matched:
            all_FP += 1
    all_TP += TP
    all_FN += len(gt_boxes) - len(gt_used)

    

    # === 可视化 ===
    # print(num)
    if num == 0 or num % 1 == 0:
        img = cv2.imread(img_path)
        for name, box in gt_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for name, box in pred_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"P:{name}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_dir, basename.replace(".xml", ".jpg")), img)
        

# ==== 评估结果 ====
precision = all_TP / (all_TP + all_FP + 1e-6)
recall = all_TP / (all_TP + all_FN + 1e-6)
# page_acc = correct_page_items / (total_page_items + 1e-6)
q_acc = p_num / total_q_items
print(total_q_items)
print(p_num)

print(f"\n========= Evaluation Result =========")
print(f"Total GT boxes: {all_TP + all_FN}")
print(f"Total Predicted boxes: {all_TP + all_FP}")
print(f"True Positives (TP): {all_TP}")
print(f"False Positives (FP): {all_FP}")
print(f"False Negatives (FN): {all_FN}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Question Number Accuracy: {q_acc:.4f}")
