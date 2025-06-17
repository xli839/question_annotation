import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def process_dataset(base_dir):
    result = []
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(base_dir):
        # 获取所有xml文件并按名称排序
        xml_files = sorted([f for f in files if f.endswith('.xml')])
        
        for xml_file in xml_files:
            xml_path = os.path.join(root, xml_file)
            try:
                # 解析XML文件
                tree = ET.parse(xml_path)
                root_elem = tree.getroot()
                
                # 获取所有name标签
                name_elems = root_elem.findall(".//name")
                if name_elems:
                    # 获取最后一个name标签的内容
                    last_name = name_elems[-1].text
                    page_num = last_name.split("-")[0]
                    
                    # 获取对应的图片文件
                    img_file = xml_path.replace(".xml", ".jpg")
                    if not os.path.exists(img_file):
                        img_file = xml_path.replace(".xml", ".png")
                    
                    if os.path.exists(img_file):
                        result.append({
                            "image_path": img_file,
                            "page_number": page_num
                        })
            except Exception as e:
                print(f"处理文件 {xml_path} 时出错: {str(e)}")
    
    return result

def main():
    # 设置数据集根目录
    base_dir = "/hpc2hdd/home/xli839/lxy/work/data/data"  # 请替换为实际的数据集路径
    
    # 处理数据集
    result = process_dataset(base_dir)
    
    # 保存为JSON文件
    output_file = "dataset_info.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    
    print(f"处理完成，结果已保存到 {output_file}")

if __name__ == "__main__":
    main()
