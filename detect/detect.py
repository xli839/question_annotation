from ultralytics import YOLO
import cv2
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from tqdm import tqdm
import cv2
import re
import logging
import argparse
import sys
from datetime import datetime
import traceback
import time

# ==================== 日志配置 ====================
def setup_logging(debug_mode=False, log_file=None):
    """
    设置日志系统
    :param debug_mode: 是否启用debug模式
    :param log_file: 日志文件路径，如果为None则只输出到控制台
    """
    # 设置日志级别
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # 创建日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 清除已有的处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 配置根日志记录器
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[]
    )
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 创建文件处理器（如果指定了日志文件）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(log_format, date_format)
        file_handler.setFormatter(file_formatter)
        logging.getLogger().addHandler(file_handler)
    
    logging.getLogger().addHandler(console_handler)
    
    # 设置第三方库的日志级别
    os.environ['GLOG_minloglevel'] = '2'  # 设置 GLOG 日志级别
    logging.getLogger('ppocr').setLevel(logging.ERROR)  # 设置 ppocr 日志级别
    logging.getLogger('ultralytics').setLevel(logging.WARNING)  # 设置 ultralytics 日志级别
    
    return logging.getLogger(__name__)

# ==================== 初始化日志 ====================
# 创建默认日志记录器
main_logger = logging.getLogger(__name__)

# 设置日志级别，减少输出
os.environ['GLOG_minloglevel'] = '2'  # 设置 GLOG 日志级别
logging.getLogger('ppocr').setLevel(logging.ERROR)  # 设置 ppocr 日志级别

# 配置路径
model_path = './weights/best_2class.pt'
image_folder = '../dataset/数学/数学-北师大版-二年级下册-同步伴读综合练习(春)(2025)题目'
output_folder = '../output_folder/math'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 中文数字转阿拉伯数字字典（支持一~二十）
cn2num = {
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20
}

# 加载模型
main_logger.info("开始加载YOLO模型...")
try:
    model = YOLO(model_path)
    main_logger.info(f"YOLO模型加载成功: {model_path}")
except Exception as e:
    main_logger.error(f"YOLO模型加载失败: {e}")
    raise

def create_xml(img_name, img_path, boxes, page_number, logger=None):
    """
    创建XML标注文件
    :param img_name: 图片名称
    :param img_path: 图片路径
    :param boxes: 检测框列表
    :param page_number: 页码
    :param logger: 日志记录器
    """
    if logger is None:
        logger = main_logger
        
    try:
        logger.debug(f"开始为图片 {img_name} 创建XML文件")
        
        # 读取图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"无法读取图片: {img_path}")
            return False
            
        img_h, img_w = img.shape[:2]
        logger.debug(f"图片尺寸: {img_w}x{img_h}")
        
        # 创建XML根元素
        root = ET.Element("annotation")
        
        # 添加基本信息
        folder = ET.SubElement(root, "folder")
        folder.text = f"{image_folder.split('/')[-1]}"
        
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
        
        # 添加检测框，每个检测框命名为"页码-大题号-序号"
        valid_boxes = 0
        for idx, box in enumerate(boxes, 1):
            try:
                obj = ET.SubElement(root, "object")
                
                name = ET.SubElement(obj, "name")
                num = int(box['num'])
                question_num = int(box['question_id'])

                # 处理页码
                try:
                    page_number = int(page_number)
                except (ValueError, TypeError):
                    try:
                        page_number = int(str(page_number).split("-")[0])
                    except (ValueError, TypeError):
                        page_number = 0
                        logger.warning(f"无法解析页码: {page_number}")

                name.text = f"{page_number}-{question_num}-{num}"  # 使用"页码-题号-序号"格式
                
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
                
                valid_boxes += 1
                logger.debug(f"添加检测框 {idx}: {name.text} at {box['bbox']}")
                
            except Exception as e:
                logger.error(f"处理检测框 {idx} 时出错: {e}")
                continue
        
        # 美化XML输出
        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t")
        
        # 保存XML文件，使用原始图片名称
        xml_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + '.xml')
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xmlstr)
        
        logger.info(f"成功创建XML文件: {xml_path}, 包含 {valid_boxes} 个有效检测框")
        return True
        
    except Exception as e:
        logger.error(f"创建XML文件失败: {e}")
        logger.debug(f"详细错误信息: {traceback.format_exc()}")
        return False

#--------------- page num --------------------------
import os
import sys
import subprocess

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image
import PaddleOCR.tools.infer.utility as utility
import PaddleOCR.tools.infer.predict_rec as predict_rec
import PaddleOCR.tools.infer.predict_det as predict_det
import PaddleOCR.tools.infer.predict_cls as predict_cls
from PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read
from PaddleOCR.ppocr.utils.logging import get_logger
from PaddleOCR.tools.infer.utility import (
    draw_ocr_box_txt,
    get_rotate_crop_image,
    get_minarea_rect_crop,
    slice_generator,
    merge_fragmented,
)

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(
                    output_dir, f"mg_crop_{bno+self.crop_image_res_index}.jpg"
                ),
                img_crop_list[bno],
            )
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True, slice={}):
        time_dict = {"det": 0, "rec": 0, "cls": 0, "all": 0}

        if img is None:
            logger.debug("no valid image provided")
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        if slice:
            slice_gen = slice_generator(
                img,
                horizontal_stride=slice["horizontal_stride"],
                vertical_stride=slice["vertical_stride"],
            )
            elapsed = []
            dt_slice_boxes = []
            for slice_crop, v_start, h_start in slice_gen:
                dt_boxes, elapse = self.text_detector(slice_crop, use_slice=True)
                if dt_boxes.size:
                    dt_boxes[:, :, 0] += h_start
                    dt_boxes[:, :, 1] += v_start
                    dt_slice_boxes.append(dt_boxes)
                    elapsed.append(elapse)
            dt_boxes = np.concatenate(dt_slice_boxes)

            dt_boxes = merge_fragmented(
                boxes=dt_boxes,
                x_threshold=slice["merge_x_thres"],
                y_threshold=slice["merge_y_thres"],
            )
            elapse = sum(elapsed)
        else:
            dt_boxes, elapse = self.text_detector(img)

        time_dict["det"] = elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            end = time.time()
            time_dict["all"] = end - start
            return None, None, time_dict
        else:
            logger.debug(
                "dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse)
            )
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(img_crop_list)
            time_dict["cls"] = elapse
            logger.debug(
                "cls num  : {}, elapsed : {}".format(len(img_crop_list), elapse)
            )
        if len(img_crop_list) > 1000:
            logger.debug(
                f"rec crops num: {len(img_crop_list)}, time and memory cost may be large."
            )

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict["rec"] = elapse
        logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict["all"] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and (
                _boxes[j + 1][0][0] < _boxes[j][0][0]
            ):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


def load_page_number_dict(json_path):
    """加载页码字典"""
    main_logger.debug(f"加载页码字典: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容你的json格式
        if isinstance(data, dict) and "image_path" in data:
            data = [data]
        result = {item["image_path"]: str(item["page_number"]) for item in data}
        main_logger.info(f"成功加载 {len(result)} 个页码映射")
        return result
    except Exception as e:
        main_logger.error(f"加载页码字典失败: {e}")
        return {}


def extract_page_number(image_path, last_question_y2, main_logger=None):
    """提取页码"""
    if main_logger is None:
        main_logger = logging.getLogger(__name__)
        
    main_logger.info(f"开始提取页码: {image_path}, last_y2: {last_question_y2}")
    
    try:
        # 获取待处理图像列表（支持文件夹输入），并按进程划分任务
        image_file_list = get_image_file_list(image_path)
        image_file_list = image_file_list[args.process_id :: args.total_process_num]
        main_logger.debug(f"待处理图像列表: {len(image_file_list)} 张")

        # 初始化OCR系统
        text_sys = TextSystem(args)
        is_visualize = True
        font_path = args.vis_font_path
        drop_score = args.drop_score
        draw_img_save_dir = args.draw_img_save_dir
        os.makedirs(draw_img_save_dir, exist_ok=True)
        save_results = []

        # 预热OCR模型，加速推理
        if args.warmup:
            main_logger.debug("预热OCR模型中...")
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = text_sys(img)
            main_logger.debug("OCR模型预热完成")

        total_time = 0
        _st = time.time()

        for idx, image_file in enumerate(image_file_list):
            # 加载图像，自动处理 gif / pdf / 图片
            img, flag_gif, flag_pdf = check_and_read(image_file)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(image_file)

            if not flag_pdf:
                if img is None:
                    main_logger.warning(f"无法加载图像: {image_file}")
                    continue
                imgs = [img]  # 普通图像直接放入列表
            else:
                # PDF多页文档处理逻辑
                page_num = args.page_num
                if page_num > len(img) or page_num == 0:
                    page_num = len(img)
                imgs = img[:page_num]
                main_logger.debug(f"PDF文档包含 {len(imgs)} 页")

            for index, img in enumerate(imgs):
                # 裁剪图像，仅保留最后一道题下方区域（用于提取页码）
                h = img.shape[0]
                crop_img = img[last_question_y2:, :, :]
                main_logger.debug(f"裁剪区域: y={last_question_y2}到底部, 裁剪后尺寸: {crop_img.shape}")

                # OCR识别该裁剪区域
                starttime = time.time()
                dt_boxes, rec_res, time_dict = text_sys(crop_img)
                elapse = time.time() - starttime
                total_time += elapse

                main_logger.debug(f"{idx}_{index} OCR预测时间: {elapse:.3f}s")

                # 记录识别到的文本
                main_logger.debug(f"识别到 {len(rec_res)} 个文本框")
                for i, (text, score) in enumerate(rec_res):
                    main_logger.debug(f"文本 {i+1}: '{text}', 置信度: {score:.3f}")

                # 保存OCR结果（包括识别框位置和文本）
                res = [
                    {
                        "transcription": rec_res[i][0],
                        "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
                    }
                    for i in range(len(dt_boxes))
                ]

                save_pred = (
                    os.path.basename(image_file)
                    + (f"_{index}" if len(imgs) > 1 else "")
                    + "\t"
                    + json.dumps(res, ensure_ascii=False)
                    + "\n"
                )
                save_results.append(save_pred)

                # 可视化识别结果并保存
                if is_visualize:
                    try:
                        image = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
                        txts = [rec_res[i][0] for i in range(len(rec_res))]
                        scores = [rec_res[i][1] for i in range(len(rec_res))]

                        draw_img = draw_ocr_box_txt(
                            image,
                            dt_boxes,
                            txts,
                            scores,
                            drop_score=drop_score,
                            font_path=font_path,
                        )

                        # 构造保存路径并写入图像
                        if flag_gif:
                            save_file = image_file[:-3] + "png"
                        elif flag_pdf:
                            save_file = image_file.replace(".pdf", "_" + str(index) + ".png")
                        else:
                            save_file = image_file

                        vis_save_path = os.path.join(draw_img_save_dir, os.path.basename(save_file))
                        cv2.imwrite(vis_save_path, draw_img[:, :, ::-1])  # RGB转BGR保存
                        main_logger.debug(f"可视化结果已保存: {vis_save_path}")
                    except Exception as e:
                        main_logger.warning(f"保存可视化结果失败: {e}")

                # ========= 提取置信度最高的数字文本作为页码 =========
                import re
                best_num = ""
                best_score = -1
                for text, score in rec_res:
                    nums = re.findall(r"\d+", text)  # 提取所有数字序列
                    if nums and score > best_score:
                        best_num = nums[0]  # 仅取第一个数字作为页码（避免"第1页，共5页"这种情况干扰）
                        best_score = score
                        main_logger.debug(f"找到更好的页码候选: '{text}' -> {best_num}, 置信度: {score:.3f}")
                pred_page_num = best_num

        # 输出自动评估报告（若开启 benchmark）
        if args.benchmark:
            text_sys.text_detector.autolog.report()
            text_sys.text_recognizer.autolog.report()

        main_logger.info(f"页码提取完成, 结果: {pred_page_num}, 总耗时: {total_time:.3f}s")
        return pred_page_num
        
    except Exception as e:
        main_logger.error(f"页码提取失败: {e}")
        main_logger.debug(f"详细错误信息: {traceback.format_exc()}")
        return ""


def chinese_to_int(text):
    """匹配中文数字并转为整数"""
    for cn, num in cn2num.items():
        if cn in text:
            return num
    return None

def extract_question_number(image_path, crop_box, main_logger=None):
    """
    从图像指定区域提取题号，并转换为整数
    :param image_path: 图像路径或目录
    :param crop_box: 裁剪坐标 [x1, y1, x2, y2]
    :return: 整数题号，如 1、2、3
    """
    if main_logger is None:
        main_logger = logging.getLogger(__name__)
        
    main_logger.debug(f"开始提取题号: 区域 {crop_box}")
    
    try:
        image_file_list = get_image_file_list(image_path)
        image_file_list = image_file_list[args.process_id :: args.total_process_num]
        text_sys = TextSystem(args)

        # 预热OCR模型，加速推理
        if args.warmup:
            main_logger.debug("预热OCR模型中...")
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = text_sys(img)

        for image_file in image_file_list:
            # 加载图像，自动处理 gif / pdf / 图片
            img, flag_gif, flag_pdf = check_and_read(image_file)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(image_file)

            if img is None:
                main_logger.warning(f"无法加载图像: {image_file}")
                continue

            # 确保裁剪坐标为整数
            x1, y1, x2, y2 = map(int, crop_box)
            main_logger.debug(f"裁剪坐标: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
            
            # 扩展裁剪区域，向左扩展5个像素
            crop_img = img[y1:y2, max(0, int(x1-5)):x2, :]
            main_logger.debug(f"裁剪后图像尺寸: {crop_img.shape}")

            dt_boxes, rec_res, _ = text_sys(crop_img)
            main_logger.debug(f"OCR识别到 {len(rec_res)} 个文本")

            for i, (text, score) in enumerate(rec_res):
                text = text.strip()
                main_logger.debug(f"识别文本 {i+1}: '{text}', 置信度: {score:.3f}")

                # 检查是否为阿拉伯数字题号（如 "1." 或 "2、"）
                match = re.match(r"^(\d+)[\.、\s)]", text)
                if match:
                    result = int(match.group(1))
                    main_logger.info(f"成功提取阿拉伯数字题号: {result}")
                    return result

                # 检查是否为中文数字题号（如 "一、", "(一)", "一."）
                match_cn = re.match(r"^[（(]?(十[一二]?|[一二三四五六七八九十])[)）、.\s]?", text)
                if match_cn:
                    num = chinese_to_int(match_cn.group(1))
                    if num:
                        main_logger.info(f"成功提取中文数字题号: '{match_cn.group(1)}' -> {num}")
                        return num

        main_logger.warning("未识别到有效题号")
        return None  # 未识别到有效题号
        
    except Exception as e:
        main_logger.error(f"题号提取失败: {e}")
        main_logger.debug(f"详细错误信息: {traceback.format_exc()}")
        return None

def gen_new_boxes(image_path, boxes, main_logger=None):
    """生成带题号信息的检测框"""
    if main_logger is None:
        main_logger = logging.getLogger(__name__)
        
    main_logger.debug(f"开始生成新的检测框, 输入框数量: {len(boxes)}")
    
    annotated = []
    current_qid = None
    l2_count = 0

    for i, box in enumerate(boxes):
        class_id = box['class_id']
        main_logger.debug(f"处理检测框 {i+1}: 类别={class_id}, 位置={box['bbox']}")

        if class_id == 0:  # 题目标题
            qid = extract_question_number(image_path, box['bbox'], main_logger)

            if qid is not None:
                box['question_id'] = qid
                main_logger.debug(f"题目标题检测框 {i+1} 题号: {qid}")
            else:
                box['question_id'] = int(-1)
                main_logger.warning(f"题目标题检测框 {i+1} 未能提取题号")
            box['num'] = 0
            current_qid = box['question_id']
            l2_count = 0

        elif class_id == 1:  # 题目内容
            if current_qid is not None and isinstance(current_qid, int):
                l2_count += 1
                box['question_id'] = current_qid
                box['num'] = l2_count
                main_logger.debug(f"题目内容检测框 {i+1} 归属题号: {current_qid}, 序号: {l2_count}")
            else:
                l2_count += 1
                box['question_id'] = int(-1)
                box['num'] = l2_count
                main_logger.warning(f"题目内容检测框 {i+1} 无法归属到有效题号")

        else:
            box['question_id'] = "跳过"
            box['num'] = -1
            main_logger.debug(f"检测框 {i+1} 跳过处理 (类别={class_id})")

        annotated.append(box)

    main_logger.info(f"生成检测框完成, 输出框数量: {len(annotated)}")
    return annotated

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='数学题目检测和标注工具')
    parser.add_argument('--debug', action='store_true', help='启用debug模式')
    parser.add_argument('--log-file', type=str, help='日志文件路径')
    
    # 解析已知参数，其余传递给PaddleOCR
    custom_args, paddle_args = parser.parse_known_args()
    
    # 设置日志
    if custom_args.log_file is None and custom_args.debug:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_args.log_file = f'logs/detect_{timestamp}.log'
    
    main_logger = setup_logging(custom_args.debug, custom_args.log_file)
    
    main_logger.info("=" * 60)
    main_logger.info("数学题目检测和标注工具启动")
    main_logger.info(f"Debug模式: {'启用' if custom_args.debug else '禁用'}")
    main_logger.info(f"模型路径: {model_path}")
    main_logger.info(f"图片文件夹: {image_folder}")
    main_logger.info(f"输出文件夹: {output_folder}")
    if custom_args.log_file:
        main_logger.info(f"日志文件: {custom_args.log_file}")
    main_logger.info("=" * 60)
    
    # 恢复原始的sys.argv用于PaddleOCR参数解析
    sys.argv = [sys.argv[0]] + paddle_args
    
    try:
        args = utility.parse_args()
        if args.use_mp:
            main_logger.info("启用多进程模式")
            p_list = []
            total_process_num = args.total_process_num
            for process_id in range(total_process_num):
                cmd = (
                    [sys.executable, "-u"]
                    + sys.argv
                    + ["--process_id={}".format(process_id), "--use_mp={}".format(False)]
                )
                p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
                p_list.append(p)
            for p in p_list:
                p.wait()
        else:
            main_logger.info("启用单进程模式")
            
            # 获取图片列表
            img_list = [f for f in os.listdir(image_folder) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            main_logger.info(f"找到 {len(img_list)} 张图片")
            
            if not img_list:
                main_logger.warning("未找到任何图片文件")
                sys.exit(0)
            
            # 处理统计
            success_count = 0
            error_count = 0
            start_time = time.time()
            
            # 遍历图片进行预测
            for img_name in tqdm(img_list, desc="处理图片"):
                try:
                    main_logger.info(f"开始处理图片: {img_name}")
                    
                    if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        main_logger.debug(f"跳过非图片文件: {img_name}")
                        continue
                    
                    img_path = os.path.join(image_folder, img_name)
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        main_logger.error(f"无法读取图片: {img_path}")
                        error_count += 1
                        continue
                    
                    img_h, img_w = img.shape[:2]
                    main_logger.debug(f"图片尺寸: {img_w}x{img_h}")
                    
                    # YOLO检测
                    main_logger.debug("开始YOLO检测")
                    result = model(img_path)[0]  # 获取第一个结果（单张图）

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
                    
                    main_logger.info(f"YOLO检测到 {len(boxes)} 个目标")
                    if custom_args.debug:
                        for i, box in enumerate(boxes):
                            main_logger.debug(f"目标 {i+1}: 类别={box['class_id']}, 置信度={box['confidence']:.3f}")
                    
                    # 根据y坐标对boxes进行排序，确保题目按顺序编号
                    boxes.sort(key=lambda x: x["bbox"][1])
                    main_logger.debug("检测框已按y坐标排序")

                    new_boxes = gen_new_boxes(img_path, boxes, main_logger)
                    
                    # 获取最后一道题的y2坐标
                    if boxes:
                        last_question_y2 = int(boxes[-1]["bbox"][3])
                        main_logger.debug(f"最后一题的y2坐标为: {last_question_y2}")
                    else:
                        last_question_y2 = 0
                        main_logger.warning("未检测到任何目标，设置last_question_y2=0")

                    # 传递 y2 坐标给页码提取函数
                    try:
                        page_number = extract_page_number(img_path, last_question_y2, main_logger)
                        main_logger.info(f"提取到页码: {page_number}")
                    except Exception as e:
                        main_logger.error(f"提取页码失败: {e}")
                        if custom_args.debug:
                            main_logger.debug(f"详细错误信息: {traceback.format_exc()}")
                        page_number = 0

                    # 为每张图片创建一个XML文件，包含所有检测到的题目
                    if create_xml(img_name, img_path, new_boxes, page_number, main_logger):
                        success_count += 1
                        main_logger.info(f"成功处理: {img_name}")
                    else:
                        error_count += 1
                        main_logger.error(f"处理失败: {img_name}")
                        
                except Exception as e:
                    error_count += 1
                    main_logger.error(f"处理图片 {img_name} 时出错: {e}")
                    if custom_args.debug:
                        main_logger.debug(f"详细错误信息: {traceback.format_exc()}")
            
            # 输出统计信息
            total_time = time.time() - start_time
            main_logger.info("=" * 60)
            main_logger.info("处理完成!")
            main_logger.info(f"总图片数: {len(img_list)}")
            main_logger.info(f"成功处理: {success_count}")
            main_logger.info(f"处理失败: {error_count}")
            main_logger.info(f"成功率: {success_count/len(img_list)*100:.1f}%")
            main_logger.info(f"总耗时: {total_time:.2f}秒")
            main_logger.info(f"平均每张: {total_time/len(img_list):.2f}秒")
            main_logger.info("=" * 60)
            
    except KeyboardInterrupt:
        main_logger.info("用户中断程序")
    except Exception as e:
        main_logger.error(f"程序执行出错: {e}")
        if custom_args.debug:
            main_logger.debug(f"详细错误信息: {traceback.format_exc()}")
        sys.exit(1)
