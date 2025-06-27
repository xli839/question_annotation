# 图像题目检测与页码提取工具

本项目旨在对课本练习题图片进行自动化识别与定位，能够高效、准确地检测出每一道题目的具体位置，并按照规范自动命名。系统会将所有检测结果以标准 Pascal VOC 格式输出为 XML 文件，便于后续的数据标注、分析与应用。经实际测试，本工具对题目区域的检测与命名准确率可达 95% 以上，适用于大规模教育类图像的自动标注与整理。

本项目基于 YOLOv8 和 PaddleOCR，完成以下功能：
- 检测图像中的题目区域并输出为 VOC 格式 XML 标注文件；
- 自动提取图像底部的页码信息；
- 支持批量处理图像文件。

---

## 目录结构

```
.
├── detect.py                # 主程序脚本
├── config.yaml            # 配置文件（推荐使用）
├── results/               # 生成的 XML 文件保存路径
├── PaddleOCR/             # 克隆的 PaddleOCR 工程
│   ├── tools/infer/       # OCR 推理工具
│   └── ppocr/             # OCR 核心库
│   └── inference_model/   
│       └── ch_PP-OCRv3_det_infer/  # 训练好的detection模型权重
│       └── ch_PP-OCRv3_rec_infer/  # 训练好的recognization模型权重
├── ultralytics           # 克隆的 yolo 工程
├── dataset/
│   └── images/            # 原始图像存放路径
├── weights/
│   └── best_yolo.pt            # 训练好的 YOLOv8 模型权重
├── requirements.txt       # 依赖包列表
└── README.md              # 使用说明文档
```

---

## 环境依赖

建议使用 Python 3.8+，推荐在虚拟环境中安装依赖。

### 安装依赖

```bash
pip install -r requirements.txt
```

或单独安装核心库：

```bash
pip install ultralytics opencv-python tqdm numpy Pillow paddleocr paddlepaddle
```

> 如需 GPU 加速，请安装 `paddlepaddle-gpu`，详见 [PaddlePaddle 官网](https://www.paddlepaddle.org.cn/install/quick)。

---

## 使用说明

1. **配置参数**

   推荐使用 `config.yaml` 进行参数配置，示例：

   ```yaml
   model_path: "weights/best.pt"
   image_folder: "dataset/images"
   output_folder: "results"
   log_path: "detect.log"
   log_level: "INFO"
   ```

   或在脚本中直接修改以下变量：

   ```python
   model_path = '/path/to/best.pt'
   image_folder = '/path/to/input/images'
   output_folder = '/path/to/save/xml'
   ```

2. **运行主程序**

   ```bash
   cd detect
   python detect.py

   # Debug模式（详细日志）
   python detect.py --debug

   # 指定日志文件
   python detect.py --debug --log-file my_log.log

   # 继续使用PaddleOCR参数
   python detect.py --debug --det_model_dir="../PaddleOCR/inference_model/ch_PP-OCRv3_det_infer/" --rec_model_dir="../PaddleOCR/inference_model/ch_PP-OCRv3_rec_infer"

   ```

3. **输出结果**

   - 每张图片对应一个 Pascal VOC 格式的 XML 文件，保存在 `output_folder` 目录下；
   - 控制台和日志文件中会打印提取的页码信息；
   - 可选：保存页码检测可视化图像。

---

## 结果说明

生成的 XML 文件采用标准 VOC 格式，每个 object 的标签为 `页码-编号`，例如：

```xml
<object>
  <name>15-2</name>  <!-- 第15页，第2道题 -->
  <bndbox>
    <xmin>...</xmin>
    <ymin>...</ymin>
    <xmax>...</xmax>
    <ymax>...</ymax>
  </bndbox>
</object>
```

---

## 模型准备

- **YOLOv8**：请使用 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) 训练并导出权重文件（`best.pt`）。
- **PaddleOCR**：建议直接克隆 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 仓库，并使用其 `tools/infer/` 方式进行推理。

---

## 示例图像要求

- 图像需包含底部页码信息；
- 题目名可反映“页码-题号-序号”
- 支持 `.jpg`、`.png`、`.jpeg` 格式。

---

## 常见问题

- **无法提取页码**：请确认图像底部有清晰的页码区域；
- **找不到 OCR 模块**：请确保 PaddleOCR 目录结构正确，并已加入 `sys.path`；
- **检测结果顺序异常**：YOLO 检测框已按 y 坐标排序，建议保证训练数据标注合理。

---

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

---

## License

MIT License
