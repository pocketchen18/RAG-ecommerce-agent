#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "==> 当前目录: $PWD"

# 1. 同步依赖
echo "==> 同步依赖"
pip install -r requirements.txt -q

# 2. 检查 .env
if [ ! -s .env ]; then
    echo "⚠️  .env 文件为空，请配置 OPENAI_API_KEY 等环境变量"
else
    echo "✅ .env 文件已存在"
fi

# 3. 检查数据目录
if [ ! -d data ]; then
    echo "⚠️  data/ 目录不存在，请先运行数据生成器: python -m data_generator"
else
    echo "✅ data/ 目录存在"
    if [ -f data/products.json ]; then
        COUNT=$(python -c "import json; print(len(json.load(open('data/products.json', encoding='utf-8'))))")
        echo "   products.json: ${COUNT} 条商品数据"
    else
        echo "⚠️  data/products.json 不存在，请先运行数据生成器: python -m data_generator"
    fi
fi

# 4. 检查 ChromaDB 是否已初始化
if [ -d data/chroma_db ]; then
    echo "✅ ChromaDB 数据库已存在"
else
    echo "⚠️  ChromaDB 数据库不存在，请运行: python -m rag.indexer"
fi

# 5. 运行基础 smoke test（如果存在）
if [ -d tests ] && ls tests/test_*.py 1>/dev/null 2>&1; then
    echo "==> 运行基础测试"
    python -m pytest tests/ -x -q --tb=short 2>/dev/null || echo "⚠️  部分测试未通过，请检查"
else
    echo "⚠️  暂无测试文件"
fi

echo ""
echo "==> 启动命令：streamlit run app.py"
echo "如果希望 init.sh 直接启动应用，请设置 RUN_START_COMMAND=1。"

if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
    echo "==> 启动 Streamlit 应用"
    exec streamlit run app.py
fi
