import os
import glob

def count_classes(label_dir):
    class_counts = {0: 0, 1: 0, 2: 0}  # 初始化三个类别的计数器
    
    # 遍历所有标签文件
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                class_id = int(line.split()[0])  # 获取类别ID
                class_counts[class_id] += 1
    
    return class_counts

# 统计训练集
train_label_dir = 'work/yolo-main/dataset/labels/val'
train_counts = count_classes(train_label_dir)

print("训练集类别分布：")
print(f"一级题目 (level1): {train_counts[0]} 个")
print(f"二级题目 (level2): {train_counts[1]} 个")
print(f"三级题目 (level3): {train_counts[2]} 个")
print(f"总计: {sum(train_counts.values())} 个")
