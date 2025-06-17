#!/bin/bash

# ============ 配置默认参数 ============
MODEL_NAME="page_num"                # 语言模型名称
START_MODEL="eng"                    # 基础模型
TESSDATA_DIR="/usr/share/tesseract-ocr/4.00/tessdata"
GROUND_TRUTH_DIR="/hpc2hdd/home/xli839/lxy/work/page_num/dataset/tesstrain_data/train/ground-truth"

DATA_DIR="data"
LANGDATA_DIR="${DATA_DIR}/langdata"
LANG_DIR="${DATA_DIR}/${MODEL_NAME}"
OUTPUT_DIR="${LANG_DIR}"
PROTO_MODEL="${OUTPUT_DIR}/${MODEL_NAME}.traineddata"
MAX_ITERATIONS=500

# ============ 解析命令行参数 ============
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model_name) MODEL_NAME="$2"; shift ;;
        --start_model) START_MODEL="$2"; shift ;;
        --tessdata_dir) TESSDATA_DIR="$2"; shift ;;
        --ground_truth_dir) GROUND_TRUTH_DIR="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --max_iterations) MAX_ITERATIONS="$2"; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$LANG_DIR"
mkdir -p "$LANGDATA_DIR/$MODEL_NAME"

# ============ 打印参数 ============
echo "====== 开始 Tesseract 微调训练 ======"
echo "模型名称:        $MODEL_NAME"
echo "基础模型:        $START_MODEL"
echo "训练轮数:        $MAX_ITERATIONS"
echo "TESSDATA 路径:   $TESSDATA_DIR"
echo "训练数据目录:    $GROUND_TRUTH_DIR"
echo "输出目录:        $OUTPUT_DIR"
echo "语言目录:        $LANG_DIR"
echo "==================================="

# ============ 自动创建页码语言文件 ============
echo "📦 正在生成最小语言文件（数字页码专用）..."

# 数字字符集
echo -e "0\n1\n2\n3\n4\n5\n6\n7\n8\n9" > "$LANG_DIR/${MODEL_NAME}.numbers"

# 空标点和词表
touch "$LANG_DIR/${MODEL_NAME}.punc"
touch "$LANG_DIR/${MODEL_NAME}.wordlist"

# 简单 config（禁用空格分隔）
echo "has_space_delimiters 0" > "$LANGDATA_DIR/$MODEL_NAME/${MODEL_NAME}.config"

# ============ 启动训练流程 ============
make training \
    MODEL_NAME="$MODEL_NAME" \
    START_MODEL="$START_MODEL" \
    TESSDATA="$TESSDATA_DIR" \
    GROUND_TRUTH_DIR="$GROUND_TRUTH_DIR" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    LANGDATA_DIR="$LANGDATA_DIR" \
    MAX_ITERATIONS="$MAX_ITERATIONS"

echo "✅ 微调训练完成，模型保存于: $OUTPUT_DIR/$MODEL_NAME.traineddata"