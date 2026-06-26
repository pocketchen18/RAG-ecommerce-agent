"""
混合检索器模块

实现：
1. 向量检索（ChromaDB 语义相似度）
2. 元数据过滤（价格、屏幕类型等硬性约束）
3. BM25 关键词检索（精确型号名匹配）
4. RRF（Reciprocal Rank Fusion）融合排序

返回符合硬性约束且语义相关的商品列表。
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field

import chromadb
from chromadb.api.types import EmbeddingFunction

# ── 从 config 加载配置 ─────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL,
    RETRIEVER_TOP_K, BM25_WEIGHT, VECTOR_WEIGHT, CHROMA_DIR,
)

# 复用 indexer 中的 embedding 函数
from rag.indexer import DashScopeEmbedding


# ── 数据结构 ──────────────────────────────────────────────

@dataclass
class RetrieverResult:
    """单条检索结果"""
    sku_id: str
    name: str
    brand: str
    price: float
    screen_type: str
    processor: str
    section_type: str
    parent_text: str  # 商品详情全文
    score: float = 0.0  # RRF 融合分数
    source: str = ""  # 来源标记：vector / bm25 / both


@dataclass
class SearchConstraints:
    """搜索约束条件（由 Planner Agent 解析生成）"""
    budget_max: Optional[float] = None  # 最高预算
    budget_min: Optional[float] = None  # 最低预算
    brands: List[str] = field(default_factory=list)  # 指定品牌
    exclude_brands: List[str] = field(default_factory=list)  # 排除品牌
    screen_type_keywords: List[str] = field(default_factory=list)  # 屏幕类型要求
    exclude_screen_keywords: List[str] = field(default_factory=list)  # 排除屏幕类型
    processor_keywords: List[str] = field(default_factory=list)  # 处理器要求
    tags: List[str] = field(default_factory=list)  # 标签要求（如拍照、游戏）
    exclude_tags: List[str] = field(default_factory=list)  # 排除标签


# ── BM25 实现 ──────────────────────────────────────────────

class BM25Index:
    """
    简易 BM25 索引（基于倒排索引）。

    适用于精确型号名匹配（如 '小米14'、'iPhone 16'）。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lengths: Dict[str, int] = {}
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # token -> {doc_id: tf}
        self.doc_meta: Dict[str, Dict[str, Any]] = {}  # doc_id -> metadata

    def tokenize(self, text: str) -> List[str]:
        """中文分词（简易实现：按字符 unigram + 关键词提取）"""
        tokens = []

        # 提取英文/数字词
        en_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        tokens.extend(en_tokens)

        # 提取中文字符（unigram）
        zh_chars = re.findall(r'[一-鿿]', text)
        tokens.extend(zh_chars)

        # 提取中文词组（2-gram）
        for i in range(len(zh_chars) - 1):
            tokens.append(zh_chars[i] + zh_chars[i + 1])

        return tokens

    def add_document(self, doc_id: str, text: str, metadata: Dict[str, Any]):
        """添加文档到索引"""
        tokens = self.tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_meta[doc_id] = metadata

        # 统计词频
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        # 更新倒排索引
        for token, count in tf.items():
            if token not in self.inverted_index:
                self.inverted_index[token] = {}
            self.inverted_index[token][doc_id] = count

        self.doc_count += 1

    def build_from_collection(self, collection):
        """从 ChromaDB collection 构建索引"""
        # 获取所有文档
        all_docs = collection.get(limit=10000)

        total_len = 0
        for doc_id, doc_text, meta in zip(
            all_docs["ids"],
            all_docs["documents"],
            all_docs["metadatas"]
        ):
            # 索引内容：文档文本 + 商品名
            index_text = f"{meta.get('name', '')} {doc_text}"
            self.add_document(doc_id, index_text, meta)
            total_len += len(self.tokenize(index_text))

        if self.doc_count > 0:
            self.avg_doc_len = total_len / self.doc_count

    def search(self, query: str, top_k: int = 20) -> List[tuple]:
        """
        BM25 搜索。

        Returns:
            [(doc_id, score, metadata), ...] 按分数降序
        """
        if self.doc_count == 0:
            return []

        query_tokens = self.tokenize(query)
        scores: Dict[str, float] = {}

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            # 包含该 token 的文档数
            df = len(self.inverted_index[token])
            idf = max(0, (self.doc_count - df + 0.5) / (df + 0.5))
            idf = 1.0 + idf  # 简化 IDF

            for doc_id, tf in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_id]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_len
                )
                score = idf * numerator / denominator
                scores[doc_id] = scores.get(doc_id, 0) + score

        # 排序并返回 top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, self.doc_meta.get(doc_id, {}))
                for doc_id, score in sorted_results]


# ── 混合检索器 ──────────────────────────────────────────────

class HybridRetriever:
    """
    混合检索器：向量检索 + 元数据过滤 + BM25 + RRF 融合。

    使用流程：
    1. 从用户 query 解析出 constraints（硬性约束）和 semantic_query（语义查询）
    2. 向量检索：使用 semantic_query 在 ChromaDB 中进行语义搜索
    3. 元数据过滤：根据 constraints 过滤掉不符合条件的结果
    4. BM25 检索：使用原始 query 进行关键词匹配
    5. RRF 融合：将两路结果按 RRF 公式融合排序
    """

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        embedding_fn: Optional[EmbeddingFunction] = None,
        top_k: int = RETRIEVER_TOP_K,
    ):
        self.chroma_dir = chroma_dir
        self.top_k = top_k

        # 初始化 embedding 函数
        if embedding_fn is None:
            self.embedding_fn = DashScopeEmbedding(
                api_base=EMBEDDING_API_BASE,
                api_key=EMBEDDING_API_KEY,
                model=EMBEDDING_MODEL,
            )
        else:
            self.embedding_fn = embedding_fn

        # 连接 ChromaDB
        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.products_col = self.client.get_collection(
            "phone_products",
            embedding_function=self.embedding_fn,
        )
        self.reviews_col = self.client.get_collection(
            "phone_reviews",
            embedding_function=self.embedding_fn,
        )

        # 构建 BM25 索引
        print("🔍 构建 BM25 索引...")
        self.bm25_index = BM25Index()
        self.bm25_index.build_from_collection(self.products_col)
        print(f"   索引完成：{self.bm25_index.doc_count} 个文档")

    def parse_constraints(self, query: str) -> tuple:
        """
        从用户 query 解析约束条件。

        Returns:
            (constraints: SearchConstraints, semantic_query: str)
        """
        constraints = SearchConstraints()
        semantic_parts = []

        # 提取价格约束
        price_patterns = [
            (r'(\d+)\s*元以内', lambda m: setattr(constraints, 'budget_max', float(m.group(1)))),
            (r'(\d+)\s*以下', lambda m: setattr(constraints, 'budget_max', float(m.group(1)))),
            (r'不超过\s*(\d+)', lambda m: setattr(constraints, 'budget_max', float(m.group(1)))),
            (r'(\d+)\s*[-~到至]\s*(\d+)\s*元', lambda m: (
                setattr(constraints, 'budget_min', float(m.group(1))),
                setattr(constraints, 'budget_max', float(m.group(2))),
            )),
        ]
        for pattern, setter in price_patterns:
            match = re.search(pattern, query)
            if match:
                setter(match)

        # 提取品牌
        brand_keywords = {
            '华为': '华为', '小米': '小米', 'OPPO': 'OPPO', 'oppo': 'OPPO',
            'vivo': 'vivo', '苹果': '苹果', 'iPhone': '苹果', 'iphone': '苹果',
            '三星': '三星', '荣耀': '荣耀', '一加': '一加', 'realme': 'realme',
        }
        for keyword, brand in brand_keywords.items():
            if keyword in query:
                if brand not in constraints.brands:
                    constraints.brands.append(brand)

        # 提取排除屏幕类型
        exclude_screen_patterns = [
            r'不要\s*曲面屏', r'不要\s*曲面', r'排除\s*曲面',
            r'直屏', r'不要\s*刘海屏', r'不要\s*水滴屏',
        ]
        for pattern in exclude_screen_patterns:
            if re.search(pattern, query):
                if '曲面' in pattern:
                    constraints.exclude_screen_keywords.append('曲面')
                elif '刘海' in pattern:
                    constraints.exclude_screen_keywords.append('刘海')
                elif '水滴' in pattern:
                    constraints.exclude_screen_keywords.append('水滴')

        # 提取屏幕类型要求
        if '直屏' in query and '曲面' not in query:
            constraints.screen_type_keywords.append('直屏')

        # 提取标签需求
        tag_mapping = {
            '拍照': '拍照', '摄影': '拍照', '摄像': '拍照',
            '游戏': '游戏', '电竞': '游戏',
            '续航': '续航', '电池': '续航',
            '轻薄': '轻薄', '轻': '轻薄',
            '商务': '商务',
            '学生': '学生',
        }
        for keyword, tag in tag_mapping.items():
            if keyword in query:
                if tag not in constraints.tags:
                    constraints.tags.append(tag)

        # 构建语义查询（移除已解析的约束部分）
        semantic_query = query
        # 移除价格描述
        semantic_query = re.sub(r'\d+\s*元以内', '', semantic_query)
        semantic_query = re.sub(r'\d+\s*以下', '', semantic_query)
        semantic_query = re.sub(r'不超过\s*\d+', '', semantic_query)
        semantic_query = re.sub(r'\d+\s*[-~到至]\s*\d+\s*元', '', semantic_query)
        # 移除"不要xxx"
        semantic_query = re.sub(r'不要\s*\S+', '', semantic_query)
        # 清理多余空格
        semantic_query = re.sub(r'\s+', ' ', semantic_query).strip()

        # 如果语义查询为空，使用原始查询
        if not semantic_query:
            semantic_query = query

        return constraints, semantic_query

    def _apply_metadata_filter(
        self,
        results: Dict[str, Any],
        constraints: SearchConstraints,
    ) -> List[Dict[str, Any]]:
        """应用元数据过滤"""
        filtered = []

        for i in range(len(results["ids"])):
            doc_id = results["ids"][i]
            meta = results["metadatas"][i]
            doc = results["documents"][i] if results.get("documents") else ""

            # 价格过滤
            price = meta.get("price", 0)
            if constraints.budget_max and price > constraints.budget_max:
                continue
            if constraints.budget_min and price < constraints.budget_min:
                continue

            # 品牌过滤
            brand = meta.get("brand", "")
            if constraints.brands and brand not in constraints.brands:
                continue
            if constraints.exclude_brands and brand in constraints.exclude_brands:
                continue

            # 屏幕类型过滤
            screen_type = meta.get("screen_type", "")
            if constraints.exclude_screen_keywords:
                if any(kw in screen_type for kw in constraints.exclude_screen_keywords):
                    continue
            if constraints.screen_type_keywords:
                if not any(kw in screen_type for kw in constraints.screen_type_keywords):
                    continue

            # 标签过滤（检查 parent_text 中是否包含标签关键词）
            parent_text = meta.get("parent_text", "")
            if constraints.exclude_tags:
                if any(kw in parent_text for kw in constraints.exclude_tags):
                    continue

            filtered.append({
                "id": doc_id,
                "metadata": meta,
                "document": doc,
            })

        return filtered

    def vector_search(
        self,
        query: str,
        constraints: SearchConstraints,
        n_results: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        向量检索 + 元数据过滤。

        Args:
            query: 语义查询文本
            constraints: 硬性约束条件
            n_results: 初始检索数量（过滤前）

        Returns:
            过滤后的结果列表
        """
        # ChromaDB 向量检索
        results = self.products_col.query(
            query_texts=[query],
            n_results=n_results,
        )

        # 展平嵌套列表
        flat_results = {
            "ids": results["ids"][0],
            "metadatas": results["metadatas"][0],
            "documents": results["documents"][0] if results.get("documents") else [],
            "distances": results["distances"][0] if results.get("distances") else [],
        }

        # 应用元数据过滤
        filtered = self._apply_metadata_filter(flat_results, constraints)

        return filtered

    def bm25_search(
        self,
        query: str,
        constraints: SearchConstraints,
        top_k: int = 30,
    ) -> List[Dict[str, Any]]:
        """
        BM25 关键词检索 + 元数据过滤。

        Args:
            query: 原始查询文本
            constraints: 硬性约束条件
            top_k: 初始检索数量（过滤前）

        Returns:
            过滤后的结果列表
        """
        # BM25 搜索
        bm25_results = self.bm25_index.search(query, top_k=top_k)

        # 转换为统一格式
        results_list = []
        for doc_id, score, meta in bm25_results:
            results_list.append({
                "id": doc_id,
                "metadata": meta,
                "score": score,
            })

        # 应用元数据过滤
        filtered = []
        for item in results_list:
            meta = item["metadata"]
            price = meta.get("price", 0)

            # 价格过滤
            if constraints.budget_max and price > constraints.budget_max:
                continue
            if constraints.budget_min and price < constraints.budget_min:
                continue

            # 品牌过滤
            brand = meta.get("brand", "")
            if constraints.brands and brand not in constraints.brands:
                continue
            if constraints.exclude_brands and brand in constraints.exclude_brands:
                continue

            # 屏幕类型过滤
            screen_type = meta.get("screen_type", "")
            if constraints.exclude_screen_keywords:
                if any(kw in screen_type for kw in constraints.exclude_screen_keywords):
                    continue

            filtered.append(item)

        return filtered

    def rrf_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[RetrieverResult]:
        """
        RRF（Reciprocal Rank Fusion）融合排序。

        RRF 公式：score = sum(1 / (k + rank_i))

        Args:
            vector_results: 向量检索结果
            bm25_results: BM25 检索结果
            k: RRF 参数（默认 60）

        Returns:
            融合排序后的结果列表
        """
        # 构建 sku_id -> 结果的映射（去重，同一商品只保留最高分的 chunk）
        sku_results: Dict[str, RetrieverResult] = {}
        sku_vector_rank: Dict[str, int] = {}
        sku_bm25_rank: Dict[str, int] = {}

        # 处理向量检索结果
        for rank, item in enumerate(vector_results):
            meta = item["metadata"]
            sku_id = meta.get("sku_id", "")

            # 同一商品只保留 section_type 为"概述"的结果，或第一个结果
            if sku_id not in sku_results or meta.get("section_type") == "概述":
                sku_results[sku_id] = RetrieverResult(
                    sku_id=sku_id,
                    name=meta.get("name", ""),
                    brand=meta.get("brand", ""),
                    price=meta.get("price", 0),
                    screen_type=meta.get("screen_type", ""),
                    processor=meta.get("processor", ""),
                    section_type=meta.get("section_type", ""),
                    parent_text=meta.get("parent_text", ""),
                    source="vector",
                )
            if sku_id not in sku_vector_rank:
                sku_vector_rank[sku_id] = rank

        # 处理 BM25 结果
        for rank, item in enumerate(bm25_results):
            meta = item["metadata"]
            sku_id = meta.get("sku_id", "")

            if sku_id not in sku_results:
                sku_results[sku_id] = RetrieverResult(
                    sku_id=sku_id,
                    name=meta.get("name", ""),
                    brand=meta.get("brand", ""),
                    price=meta.get("price", 0),
                    screen_type=meta.get("screen_type", ""),
                    processor=meta.get("processor", ""),
                    section_type=meta.get("section_type", ""),
                    parent_text=meta.get("parent_text", ""),
                    source="bm25",
                )
            elif sku_results[sku_id].source == "vector":
                sku_results[sku_id].source = "both"

            if sku_id not in sku_bm25_rank:
                sku_bm25_rank[sku_id] = rank

        # 计算 RRF 分数
        for sku_id, result in sku_results.items():
            rrf_score = 0.0

            # 向量检索分数
            if sku_id in sku_vector_rank:
                rrf_score += VECTOR_WEIGHT / (k + sku_vector_rank[sku_id] + 1)

            # BM25 分数
            if sku_id in sku_bm25_rank:
                rrf_score += BM25_WEIGHT / (k + sku_bm25_rank[sku_id] + 1)

            result.score = rrf_score

        # 按 RRF 分数排序
        sorted_results = sorted(sku_results.values(), key=lambda x: x.score, reverse=True)

        return sorted_results

    def search(
        self,
        query: str,
        constraints: Optional[SearchConstraints] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrieverResult]:
        """
        混合检索主入口。

        Args:
            query: 用户查询
            constraints: 约束条件（如果为 None，自动解析）
            top_k: 返回结果数量（默认使用配置值）

        Returns:
            融合排序后的商品列表
        """
        if top_k is None:
            top_k = self.top_k

        # 解析约束条件
        if constraints is None:
            constraints, semantic_query = self.parse_constraints(query)
        else:
            semantic_query = query

        print(f"🔍 检索参数:")
        print(f"   原始查询: {query}")
        print(f"   语义查询: {semantic_query}")
        print(f"   约束条件: {constraints}")

        # 1. 向量检索
        print(f"\n📊 向量检索...")
        vector_results = self.vector_search(semantic_query, constraints)
        print(f"   返回 {len(vector_results)} 条结果")

        # 2. BM25 检索
        print(f"\n📝 BM25 检索...")
        bm25_results = self.bm25_search(query, constraints)
        print(f"   返回 {len(bm25_results)} 条结果")

        # 3. RRF 融合
        print(f"\n🔄 RRF 融合排序...")
        fused_results = self.rrf_fusion(vector_results, bm25_results)

        # 返回 top_k
        final_results = fused_results[:top_k]
        print(f"   最终返回 {len(final_results)} 条结果")

        return final_results

    def search_simple(self, query: str, top_k: int = 5) -> List[RetrieverResult]:
        """
        简化版搜索（自动解析约束，返回 top_k 结果）。

        Args:
            query: 用户查询
            top_k: 返回结果数量

        Returns:
            检索结果列表
        """
        return self.search(query, top_k=top_k)


# ── 便捷函数 ──────────────────────────────────────────────

def create_retriever() -> HybridRetriever:
    """创建检索器实例"""
    return HybridRetriever()


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("混合检索器测试")
    print("=" * 60)

    retriever = create_retriever()

    # 测试用例
    test_queries = [
        "3000元以内拍照好的手机",
        "不要曲面屏",
        "小米14",
        "游戏手机推荐",
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询: {query}")
        print('=' * 60)

        results = retriever.search_simple(query, top_k=5)

        print(f"\n📋 Top-{len(results)} 结果:")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.name} ({r.brand}, {r.price:.0f}元)")
            print(f"   屏幕: {r.screen_type} | 处理器: {r.processor}")
            print(f"   RRF分数: {r.score:.4f} | 来源: {r.source}")
            print()
