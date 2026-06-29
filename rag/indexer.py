"""
数据处理与 ChromaDB 入库模块

功能：
1. 加载 data/products.json 商品数据
2. 构建商品详情文本，按 Parent-Child 策略切分
3. 写入 ChromaDB 两个 collection：
   - phone_products: 商品详情的 child chunks
   - phone_reviews: 用户评价
"""
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 强制 sys.stdout 和 sys.stderr 使用 UTF-8 编码，防止在 Windows GBK 终端下打印 emoji 崩溃
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from openai import OpenAI

# ── 从 config 加载配置 ─────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL


# ── DashScope Embedding 函数 ──────────────────────────────
# 使用 DashScope text-embedding-v3/v4 API（OpenAI 兼容接口）

class DashScopeEmbedding(EmbeddingFunction):
    """
    基于 DashScope API 的 embedding 函数。
    调用 text-embedding-v3/v4 模型生成真实语义向量。
    """

    def __init__(self, api_base: str, api_key: str, model: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base,
        )
        self.model = model

    def __call__(self, input: Documents) -> Embeddings:
        # DashScope text-embedding 批量限制为 10 条/次
        batch_size = 10
        all_embeddings = []
        total = len(input)

        for i in range(0, total, batch_size):
            batch = input[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            # 进度输出
            done = min(i + batch_size, total)
            if done % 100 == 0 or done == total:
                print(f"   embedding 进度: {done}/{total}")

        return all_embeddings


def get_embedding_fn():
    """获取全局 embedding 函数实例"""
    import os
    from config import EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL
    api_key = os.getenv("EMBEDDING_API_KEY", EMBEDDING_API_KEY)
    if not api_key:
        # 如果没有配置 API Key，返回一个虚拟的 embedding 函数以防止 import 报错，
        # 或者在实际调用时再抛出异常。
        # 为了兼容性，如果没有配置，这里先暂时用空串（会在实际调用时报错）。
        api_key = "dummy"
        
    return DashScopeEmbedding(
        api_base=os.getenv("EMBEDDING_API_BASE", EMBEDDING_API_BASE),
        api_key=api_key,
        model=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
    )

# ── 路径配置 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PRODUCTS_JSON = DATA_DIR / "products.json"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── 文本构建模板 ─────────────────────────────────────────

def build_product_text(product: Dict[str, Any]) -> str:
    """从结构化数据构建商品详情全文（Parent 文本）"""
    parts = []

    # 基本信息
    name = product["name"]
    brand = product["brand"]
    series = product.get("series", "")
    price = product["price"]
    original_price = product.get("original_price")
    desc = product.get("description", "")
    tags = product.get("tags", [])

    parts.append(f"{name}是{brand}品牌")
    if series:
        parts[0] += f"{series}系列"
    parts[0] += f"的智能手机，售价{price:.0f}元"
    if original_price and original_price > price:
        parts[0] += f"（原价{original_price:.0f}元）"
    parts[0] += "。"
    if desc:
        parts.append(desc + "。")
    if tags:
        parts.append(f"产品标签：{'、'.join(tags)}。")

    # 屏幕参数
    screen_parts = []
    screen_type = product.get("screen_type", "")
    screen_size = product.get("screen_size", "")
    resolution = product.get("screen_resolution", "")
    refresh = product.get("refresh_rate", "")
    if screen_type:
        screen_parts.append(screen_type)
    if screen_size:
        screen_parts.append(f"{screen_size}英寸")
    if resolution:
        screen_parts.append(f"{resolution}分辨率")
    if refresh:
        screen_parts.append(f"{refresh}Hz刷新率")
    if screen_parts:
        parts.append(f"屏幕方面，采用{'、'.join(screen_parts)}的屏幕配置。")

    # 性能参数
    processor = product.get("processor", "")
    ram = product.get("ram", "")
    storage = product.get("storage", "")
    perf_parts = []
    if processor:
        perf_parts.append(f"搭载{processor}处理器")
    if ram:
        perf_parts.append(f"{ram}GB运行内存")
    if storage:
        perf_parts.append(f"{storage}GB存储")
    if perf_parts:
        parts.append(f"性能方面，{name}{'，'.join(perf_parts)}。")

    # 拍照参数
    camera_main = product.get("camera_main", "")
    camera_ultra = product.get("camera_ultra_wide", "")
    camera_tele = product.get("camera_telephoto", "")
    camera_front = product.get("camera_front", "")
    cam_parts = []
    if camera_main:
        cam_parts.append(f"{camera_main}主摄")
    if camera_ultra:
        cam_parts.append(f"{camera_ultra}超广角")
    if camera_tele:
        cam_parts.append(f"{camera_tele}长焦")
    if camera_front:
        cam_parts.append(f"{camera_front}前置")
    if cam_parts:
        parts.append(f"拍照方面，配备{'、'.join(cam_parts)}的摄像头组合，拍照表现出色。")

    # 续航参数
    battery = product.get("battery", "")
    charging = product.get("charging", "")
    if battery:
        batt_text = f"续航方面，内置{battery}mAh大电池"
        if charging:
            batt_text += f"，支持{charging}"
        parts.append(batt_text + "。")

    # 其他参数
    weight = product.get("weight", "")
    os_name = product.get("os", "")
    other_parts = []
    if weight:
        other_parts.append(f"重量{weight}g")
    if os_name:
        other_parts.append(f"运行{os_name}系统")
    if other_parts:
        parts.append(f"其他方面，{'，'.join(other_parts)}。")

    # 评价摘要
    reviews = product.get("reviews", [])
    if reviews:
        avg_rating = sum(r["rating"] for r in reviews) / len(reviews)
        parts.append(f"用户评价方面，共{len(reviews)}条评价，平均评分{avg_rating:.1f}分。")
        # 取 top 3 高赞评价作为摘要
        top_reviews = sorted(reviews, key=lambda r: r.get("likes", 0), reverse=True)[:3]
        for r in top_reviews:
            parts.append(f"用户{r['username']}（{r['rating']}星）评价：{r['content']}")

    return "\n".join(parts)


def split_into_sections(text: str, product_name: str) -> List[Tuple[str, str]]:
    """
    将商品详情文本按语义段落切分为 child chunks。

    返回: [(section_type, chunk_text), ...]
    """
    sections = []
    current_type = "概述"
    current_lines = []

    # 按段落切分
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # 判断段落类型
        if "屏幕方面" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "屏幕"
        elif "性能方面" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "性能"
        elif "拍照方面" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "拍照"
        elif "续航方面" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "续航"
        elif "其他方面" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "其他"
        elif "用户评价" in line:
            if current_lines:
                sections.append((current_type, "\n".join(current_lines)))
                current_lines = []
            current_type = "评价"

        current_lines.append(line)

    if current_lines:
        sections.append((current_type, "\n".join(current_lines)))

    # 确保每个 chunk 包含产品名上下文
    result = []
    for section_type, chunk_text in sections:
        # 如果 chunk 太短，添加产品名前缀
        if len(chunk_text) < 20:
            chunk_text = f"{product_name} - {chunk_text}"
        result.append((section_type, chunk_text))

    return result


def build_review_text(review: Dict[str, Any], product_name: str) -> str:
    """构建评价文本（包含产品上下文）"""
    return f"关于{product_name}的用户评价：{review['content']}"


# ── 入库主逻辑 ──────────────────────────────────────────

def load_products(json_path: Path) -> List[Dict[str, Any]]:
    """加载商品 JSON 数据"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_to_chromadb(
    products: List[Dict[str, Any]],
    chroma_dir: Path = CHROMA_DIR,
) -> Tuple[int, int]:
    """
    将商品和评价数据写入 ChromaDB。

    Returns:
        (product_doc_count, review_doc_count)
    """
    # 确保目录存在
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 ChromaDB 客户端（持久化存储）
    client = chromadb.PersistentClient(path=str(chroma_dir))

    # 删除旧 collection（如果存在），确保干净重建
    try:
        client.delete_collection("phone_products")
    except Exception:
        pass
    try:
        client.delete_collection("phone_reviews")
    except Exception:
        pass

    # 创建 collection（使用自定义 embedding 函数）
    products_col = client.create_collection(
        name="phone_products",
        metadata={"description": "手机商品详情的 child chunks，用于语义检索"},
        embedding_function=get_embedding_fn(),
    )
    reviews_col = client.create_collection(
        name="phone_reviews",
        metadata={"description": "手机用户评价，用于语义检索"},
        embedding_function=get_embedding_fn(),
    )

    # ── 写入商品数据 ──────────────────────────────────
    product_ids = []
    product_docs = []
    product_metas = []

    for product in products:
        sku_id = product["sku_id"]
        name = product["name"]
        brand = product["brand"]
        price = product["price"]
        screen_type = product.get("screen_type", "")
        processor = product.get("processor", "")
        tags = product.get("tags", [])

        # 构建详情全文并切分
        full_text = build_product_text(product)
        sections = split_into_sections(full_text, name)

        for idx, (section_type, chunk_text) in enumerate(sections):
            doc_id = f"{sku_id}_detail_{idx}"
            product_ids.append(doc_id)
            product_docs.append(chunk_text)
            product_metas.append({
                "sku_id": sku_id,
                "name": name,
                "brand": brand,
                "price": float(price),
                "screen_type": screen_type,
                "processor": processor,
                "section_type": section_type,
                "tags": ",".join(tags) if tags else "",
                "parent_text": full_text,  # Parent 全文，用于召回后展示
            })

    # 批量写入商品 collection
    if product_ids:
        # ChromaDB 单次 add 上限 5461 条，分批写入
        batch_size = 5000
        for i in range(0, len(product_ids), batch_size):
            products_col.add(
                ids=product_ids[i:i+batch_size],
                documents=product_docs[i:i+batch_size],
                metadatas=product_metas[i:i+batch_size],
            )

    # ── 写入评价数据 ──────────────────────────────────
    review_ids = []
    review_docs = []
    review_metas = []

    for product in products:
        sku_id = product["sku_id"]
        name = product["name"]
        brand = product["brand"]
        price = product["price"]

        for idx, review in enumerate(product.get("reviews", [])):
            doc_id = f"{sku_id}_review_{idx}"
            review_ids.append(doc_id)
            review_docs.append(build_review_text(review, name))
            review_metas.append({
                "sku_id": sku_id,
                "name": name,
                "brand": brand,
                "price": float(price),
                "rating": review["rating"],
                "username": review["username"],
                "date": review.get("date", ""),
                "likes": review.get("likes", 0),
                "review_id": review.get("review_id", ""),
            })

    # 批量写入评价 collection
    if review_ids:
        batch_size = 5000
        for i in range(0, len(review_ids), batch_size):
            reviews_col.add(
                ids=review_ids[i:i+batch_size],
                documents=review_docs[i:i+batch_size],
                metadatas=review_metas[i:i+batch_size],
            )

    return len(product_ids), len(review_ids)


def main():
    """主入口：加载数据 → 入库 → 验证"""
    print("=" * 60)
    print("数据处理与 ChromaDB 入库")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n📂 加载商品数据: {PRODUCTS_JSON}")
    if not PRODUCTS_JSON.exists():
        print(f"❌ 文件不存在: {PRODUCTS_JSON}")
        print("   请先运行: python -m data_generator")
        return False

    products = load_products(PRODUCTS_JSON)
    print(f"   加载 {len(products)} 条商品数据")

    # 2. 入库
    print(f"\n📦 写入 ChromaDB: {CHROMA_DIR}")
    product_count, review_count = index_to_chromadb(products)
    print(f"   phone_products collection: {product_count} 条文档")
    print(f"   phone_reviews collection: {review_count} 条文档")

    # 3. 验证
    print("\n🔍 验证入库结果...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    products_col = client.get_collection("phone_products", embedding_function=get_embedding_fn())
    reviews_col = client.get_collection("phone_reviews", embedding_function=get_embedding_fn())

    p_count = products_col.count()
    r_count = reviews_col.count()
    print(f"   phone_products 文档数: {p_count}")
    print(f"   phone_reviews 文档数: {r_count}")

    assert p_count > 0, "phone_products collection 为空"
    assert r_count > 0, "phone_reviews collection 为空"

    # 4. 检查 metadata 字段
    sample = products_col.peek(limit=1)
    sample_meta = sample["metadatas"][0]
    required_fields = ["sku_id", "brand", "price"]
    for field in required_fields:
        assert field in sample_meta, f"缺少 metadata 字段: {field}"
    print(f"   ✅ metadata 包含必填字段: {required_fields}")

    # 5. 相似度查询测试
    print("\n🔎 相似度查询测试: '拍照好'")
    results = products_col.query(
        query_texts=["拍照好"],
        n_results=5,
    )
    print("   Top-5 结果:")
    for i, (doc_id, meta) in enumerate(zip(results["ids"][0], results["metadatas"][0])):
        print(f"   {i+1}. {meta['name']} ({meta['brand']}, {meta['price']:.0f}元) "
              f"[{meta['section_type']}] - {doc_id}")

    # 检查是否包含拍照相关机型
    camera_keywords = ["拍照", "摄像", "影像", "主摄", "长焦", "摄影"]
    camera_results = []
    for meta in results["metadatas"][0]:
        doc_text = meta.get("name", "") + " " + meta.get("section_type", "")
        # 也可以检查 parent_text
        parent = meta.get("parent_text", "")
        if any(kw in parent for kw in camera_keywords):
            camera_results.append(meta["name"])

    if camera_results:
        print(f"   ✅ Top-5 中包含拍照相关机型: {camera_results}")
    else:
        print(f"   ⚠️ Top-5 中未明确找到拍照相关机型，但查询成功返回")

    print("\n" + "=" * 60)
    print("✅ 数据处理与 ChromaDB 入库完成")
    print(f"   ChromaDB 路径: {CHROMA_DIR}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
