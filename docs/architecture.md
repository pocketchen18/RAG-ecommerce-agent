# 电商导购 Agent — 架构设计文档

## Context

项目目标：构建一个基于 LangGraph 的"生成-批判"双 Agent 电商导购系统，聚焦手机品类，作为简历项目展示 RAG + Multi-Agent + Reflection 三大核心技术能力。

技术栈：LangGraph、LangChain、ChromaDB、OpenAI、Pydantic、Streamlit。

---

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit 前端                         │
│  用户输入 ──→ 对话展示 + 推荐卡片 + 反思日志面板          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│               LangGraph State Machine                    │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌─────────────┐       │
│  │ Planner  │───→│ Retriever │───→│  Generator   │       │
│  │  Agent   │    │   Agent   │    │    Agent     │       │
│  └──────────┘    └─────┬─────┘    └──────┬──────┘       │
│                        │                  │               │
│                        │    ┌─────────────┘               │
│                        │    ▼                             │
│                        │ ┌──────────┐    ┌────────────┐  │
│                        │ │  Critic  │───→│ Presenter  │  │
│                        │ │  Agent   │    │   Agent    │  │
│                        │ └────┬─────┘    └────────────┘  │
│                        │      │                           │
│                        │  未通过 + 意见                    │
│                        └──────┘                           │
│                     (最多迭代 2 次)                        │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   检索层 (RAG)                            │
│                                                          │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────┐  │
│  │ 向量检索(语义)  │  │ 元数据过滤(硬)  │  │ BM25(关键词)│ │
│  │ ChromaDB       │  │ price/brand... │  │ 倒排索引   │  │
│  └───────┬────────┘  └───────┬────────┘  └─────┬─────┘  │
│          └───────────┬───────┘                  │        │
│                      ▼                          │        │
│              ┌──────────────┐                    │        │
│              │  混合排序融合  │◀───────────────────┘        │
│              │ RRF / Weighted│                            │
│              └──────────────┘                             │
│                                                          │
│  ┌────────────────────────────────────────────────┐      │
│  │ 分层检索 (Parent-Child Retrieval)                │      │
│  │ Child chunks → 检索 → 召回 Parent chunks → LLM  │      │
│  └────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   数据层                                  │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ 商品结构化数据 │  │ 商品详情向量库 │  │ 用户评论向量库 │   │
│  │ (JSON/SQLite) │  │ (ChromaDB)   │  │ (ChromaDB)   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 二、数据层设计

### 2.1 数据爬取 — JD 手机商品

爬取目标：京东手机品类 Top 100-200 款热销机型

**商品结构化数据 schema：**
```python
class PhoneProduct(BaseModel):
    sku_id: str                  # 商品ID
    name: str                    # 商品名称（如 "小米14 Ultra"）
    brand: str                   # 品牌
    price: float                 # 当前价格
    original_price: float | None # 原价
    screen_size: float           # 屏幕尺寸
    screen_type: str             # 屏幕类型（直屏/曲面屏/微曲屏）
    processor: str               # 处理器（如 "骁龙8 Gen3"）
    ram: str                     # 内存（如 "12GB"）
    storage: str                 # 存储（如 "256GB"）
    battery: int                 # 电池容量 mAh
    camera_main: str             # 主摄像素（如 "5000万"）
    camera_features: list[str]   # 摄像头特性（潜望长焦/OIS/大底...）
    weight: float                # 重量 g
    os: str                      # 操作系统
    tags: list[str]              # 标签（如 ["拍照旗舰", "轻薄", "性价比"]）
    detail_text: str             # 商品详情页文本（用于 Parent-Child 切分）
    reviews: list[Review]        # 评论列表
    url: str                     # 商品链接

class Review(BaseModel):
    user: str
    rating: int                  # 1-5
    content: str                 # 评论内容
    sentiment: str | None        # positive/negative/neutral（可后处理）
    tags: list[str]              # 如 ["拍照好", "续航强", "发热"]
```

**爬取方案：**
- 使用 `requests` + `parsel`（或 `playwright` 处理动态加载）
- 爬取列表页获取 SKU 列表 → 逐个爬取详情页 + 评论页
- 设置合理延时（2-5s），用随机 UA
- 存为 `data/products.json`

### 2.2 数据处理与入库

**Parent-Child 分层切分策略：**
```
商品详情全文 (Parent)
    ├── [Child 1] 外观设计段落 → embedding
    ├── [Child 2] 屏幕参数段落 → embedding
    ├── [Child 3] 摄像头详细描述 → embedding
    ├── [Child 4] 性能跑分描述 → embedding
    └── [Child 5] 续航充电描述 → embedding

每条评论 (Parent)
    └── [Child] 评论全文 → embedding
```

- **Child chunk**：用于 embedding 和检索（语义精准）
- **Parent chunk**：命中 Child 后召回完整上下文给 LLM（信息完整）
- 使用 LangChain 的 `RecursiveCharacterTextSplitter`，chunk_size=200, overlap=50

**ChromaDB 集合设计：**
```
Collection: phone_products
  - documents: child chunks of product detail
  - metadatas: {sku_id, brand, price, screen_type, section_type, ...}
  - ids: "{sku_id}_detail_{chunk_idx}"

Collection: phone_reviews
  - documents: review text
  - metadatas: {sku_id, brand, rating, sentiment, ...}
  - ids: "{sku_id}_review_{idx}"
```

---

## 三、Agent 工作流设计（LangGraph）

### 3.1 State 定义

```python
class AgentState(TypedDict):
    user_query: str                          # 用户原始输入
    constraints: StructuredConstraints       # Planner 解析的结构化约束
    candidates: list[PhoneProduct]           # Retriever 检索的候选商品
    recommendation: str                      # Generator 生成的推荐文本
    comparison_table: str                    # Generator 生成的对比表格 Markdown
    critic_verdict: CriticVerdict            # Critic 的审查结果
    revision_notes: str                      # Critic 的修改意见
    iteration: int                           # 当前迭代次数
    max_iterations: int                      # 最大迭代次数（默认2）
    final_output: str                        # Presenter 的最终输出
    reflection_log: list[ReflectionEntry]    # 完整反思日志（展示用）

class StructuredConstraints(BaseModel):
    category: str                    # 品类（固定 "手机"）
    budget_max: float | None         # 预算上限
    budget_min: float | None         # 预算下限
    brand_preference: list[str]      # 偏好品牌
    brand_exclude: list[str]         # 排除品牌
    core_needs: list[str]            # 核心需求（如 ["拍照", "续航"]）
    negative_constraints: list[str]  # 负向约束（如 ["不要曲面屏"]）
    scenario: str                    # 使用场景（如 "送礼", "游戏", "商务"）
    priority_ranking: list[str]      # 需求优先级排序

class CriticVerdict(BaseModel):
    passed: bool                     # 是否通过
    checks: list[CheckResult]        # 各项检查结果
    diversity_ok: bool               # 多样性检查是否通过
    overall_comment: str             # 总体评价

class CheckResult(BaseModel):
    check_name: str                  # 检查项名称
    passed: bool
    detail: str                      # 具体说明
```

### 3.2 Graph 节点设计

```python
graph = StateGraph(AgentState)

graph.add_node("planner",    planner_node)
graph.add_node("retriever",  retriever_node)
graph.add_node("generator",  generator_node)
graph.add_node("critic",     critic_node)
graph.add_node("presenter",  presenter_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "retriever")
graph.add_edge("retriever", "generator")
graph.add_edge("generator", "critic")
graph.add_conditional_edges(
    "critic",
    should_revise,
    {
        "revise": "retriever",
        "pass": "presenter"
    }
)
graph.set_finish_point("presenter")
```

### 3.3 各节点详细逻辑

**Planner Node：**
- 使用 LLM + Pydantic structured output 解析用户 query
- 提取：预算、品牌偏好/排除、核心需求、负向约束、场景、优先级

**Retriever Node：**
- 硬性过滤：price/brand/screen_type → ChromaDB metadata filter
- 语义检索：core_needs + scenario → ChromaDB similarity search
- BM25 关键词检索：商品名+标签 → rank_bm25
- 混合排序：RRF 融合三路结果
- Parent 召回：命中 child chunk 后通过 sku_id 召回完整商品数据
- Critic 打回时根据 revision_notes 调整检索策略

**Generator Node：**
- LLM 生成推荐话术 + Markdown 对比表格 + 每款推荐理由

**Critic Node（核心亮点）：**

| 检查项 | 验证方式 | 说明 |
|--------|---------|------|
| 价格约束 | Python 代码 | 遍历 candidates，确认 price <= budget_max |
| 负向约束 | LLM + 代码 | 确认推荐中无排除项 |
| 核心需求覆盖 | LLM 审查 | 推荐理由中提到核心需求的具体参数 |
| 品牌排除 | Python 代码 | 确认无排除品牌 |
| 多样性检查 | Python 代码 | 候选来自 >=2 个品牌 |
| 推荐质量 | LLM 审查 | 推荐理由具体有说服力 |

未通过时生成结构化修改意见，驱动下一轮迭代。

**Presenter Node：**
- 格式化最终 Markdown 输出 + 反思日志摘要

---

## 四、项目目录结构

```
RAG__learn/
├── app.py                    # Streamlit 主入口
├── config.py                 # 全局配置
├── requirements.txt          # 依赖清单
├── .env                      # API Keys
├── data/
│   ├── products.json         # 爬取的商品数据
│   └── reviews.json          # 爬取的评论数据
├── crawler/
│   ├── __init__.py
│   ├── jd_spider.py          # 京东爬虫主逻辑
│   └── data_processor.py     # 数据清洗、schema 校验
├── rag/
│   ├── __init__.py
│   ├── embeddings.py         # Embedding 模型封装
│   ├── vectorstore.py        # ChromaDB 初始化 + 增删查
│   ├── parent_child.py       # Parent-Child 分层切分逻辑
│   ├── retriever.py          # 混合检索器
│   └── indexer.py            # 数据入库 pipeline
├── agents/
│   ├── __init__.py
│   ├── state.py              # Pydantic 模型定义
│   ├── planner.py            # Planner Node
│   ├── retriever_agent.py    # Retriever Node
│   ├── generator.py          # Generator Node
│   ├── critic.py             # Critic Node + 工具函数
│   ├── presenter.py          # Presenter Node
│   └── graph.py              # LangGraph 状态图组装
├── tools/
│   ├── __init__.py
│   └── verification.py       # Critic 代码验证工具
├── tests/
│   ├── test_retriever.py
│   ├── test_critic.py
│   └── test_graph.py
└── docs/
    └── architecture.md       # 本文档
```

---

## 五、关键技术亮点（简历/面试素材）

1. **RAG 混合检索**：向量语义检索 + 元数据硬过滤 + BM25 关键词检索，RRF 融合排序
2. **Parent-Child Retrieval**：解决电商长文本噪声问题，小粒度检索 + 大粒度上下文召回
3. **Reflection Workflow**：Critic Agent 带结构化检查清单 + 代码形式化验证，不只是 LLM 自审
4. **条件回退循环**：LangGraph conditional edges 实现"审查不通过 → 打回重做"，最多迭代 N 次
5. **反思可观测性**：完整的反思日志，用户可看到 Agent 的自我纠正过程
