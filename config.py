"""
全局配置文件

所有 API Key 和模型配置集中在此处管理。
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（项目根目录）
load_dotenv(Path(__file__).resolve().parent / ".env")

# ── 项目路径 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── 嵌入模型 API（用于 RAG 向量检索） ────────────────────
# DashScope（阿里云）
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")           # ← 填 DashScope API Key
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v3")

# ── 语言模型 API（用于 Agent 推理、生成、审查） ──────────
# DashScope（阿里云）
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")                       # ← 填 DashScope API Key
LLM_MODEL = os.getenv("LLM_MODEL", "qwen-plus")

# ── 检索参数 ──────────────────────────────────────────────
RETRIEVER_TOP_K = 5          # 检索返回的候选数量
BM25_WEIGHT = 0.3            # BM25 在 RRF 融合中的权重
VECTOR_WEIGHT = 0.7          # 向量检索在 RRF 融合中的权重
