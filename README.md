# 📱 电商导购 Agent

基于 LangGraph 的「生成-批判」双 Agent 电商导购系统，聚焦手机品类。

核心技术：**RAG 混合检索** + **Multi-Agent 协作** + **Critic 反思工作流**。

---

## 架构

```
用户输入
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│              LangGraph State Machine                      │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌─────────────┐       │
│  │ Planner  │───→│ Retriever │───→│  Generator   │       │
│  │ (意图解析)│    │ (混合检索) │    │ (推荐生成)   │       │
│  └──────────┘    └─────┬─────┘    └──────┬──────┘       │
│                        │                  │               │
│                        │    ┌─────────────┘               │
│                        │    ▼                             │
│                        │ ┌──────────┐    ┌────────────┐  │
│                        │ │  Critic  │───→│ Presenter  │  │
│                        │ │ (反思审查)│    │ (结果呈现) │  │
│                        │ └────┬─────┘    └────────────┘  │
│                        │      │                           │
│                        │  未通过 + 修改意见                │
│                        └──────┘                           │
│                     (最多迭代 N 次)                        │
└──────────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│                   RAG 检索层                              │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐           │
│  │ 向量检索    │  │ 元数据过滤  │  │ BM25     │           │
│  │ (ChromaDB) │  │ (硬性约束)  │  │ (关键词)  │           │
│  └─────┬──────┘  └─────┬──────┘  └────┬─────┘           │
│        └───────┬───────┘              │                  │
│                ▼                      │                  │
│        ┌────────────┐                 │                  │
│        │ RRF 融合排序│◀────────────────┘                  │
│        └────────────┘                                     │
└──────────────────────────────────────────────────────────┘
```

### 工作流程

1. **Planner** — 将自然语言需求解析为结构化约束（预算、品牌、场景、需求、负向约束）
2. **Retriever** — 混合检索：向量语义 + 元数据硬过滤 + BM25 关键词，RRF 融合排序
3. **Generator** — 根据候选商品和约束生成推荐话术 + Markdown 对比表格
4. **Critic** — 执行 6 项结构化检查，未通过时输出修改意见触发回退
5. **Presenter** — 格式化最终输出 + 反思日志摘要

### Critic 6 项检查

| 检查项 | 方式 | 说明 |
|--------|------|------|
| 价格约束 | 代码 | 推荐商品是否超预算 |
| 负向约束 | 代码 | 是否违反用户排除条件（如曲面屏） |
| 多样性 | 代码 | 同品牌商品是否过多 |
| 需求覆盖 | LLM | 推荐理由是否覆盖用户核心需求 |
| 信息准确性 | 代码 | 对比表格数据是否与候选一致 |
| 推荐完整性 | 代码 | 推荐文本和表格是否完整 |

---

## 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <repo-url>
cd RAG__learn

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
EMBEDDING_API_KEY=sk-your-dashscope-key
LLM_API_KEY=sk-your-dashscope-key
```

> 默认使用 DashScope（阿里云）的 `text-embedding-v4` 嵌入模型和 `qwen-plus` 语言模型。
> 支持任何 OpenAI 兼容 API（DeepSeek、OpenAI 等），修改 `API_BASE` 和 `MODEL` 即可。

### 3. 初始化数据

```bash
# 生成手机商品数据（120 条）
python -m data_generator

# 数据入库到 ChromaDB
python -m rag.indexer
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器访问 `http://localhost:8501`。

---

## Demo 查询

以下 5 个典型查询覆盖不同场景和约束组合：

### Query 1: 送礼场景 + 预算约束
```
预算3000，送女朋友，主要拍照好，不要曲面屏
```
**预期行为**：Planner 解析出 budget_max=3000, scenario=送礼, core_needs=[拍照], negative_constraints=[曲面屏]。Retriever 过滤曲面屏机型，优先返回拍照旗舰。Generator 生成针对送礼场景的推荐话术。

### Query 2: 游戏性能 + 不限预算
```
打游戏用的，预算不限
```
**预期行为**：Planner 解析出 core_needs=[游戏性能], budget_max=None。Retriever 返回高性能处理器机型（骁龙8 Gen3、天玑9300 等）。推荐理由突出游戏帧率、散热表现。

### Query 3: 孝心场景 + 续航需求
```
给父母买个手机，2000左右，屏幕大一点，续航好
```
**预期行为**：Planner 解析出 scenario=送礼, core_needs=[屏幕质量, 续航]。推荐理由突出大屏、长续航、易用性。价格控制在 2000 元附近。

### Query 4: 品牌偏好 + 负向约束
```
4000元以内小米手机，不要曲面屏
```
**预期行为**：Planner 解析出 brands=[小米], budget_max=4000, negative_constraints=[曲面屏]。Retriever 严格过滤小米品牌 + 直屏机型。结果均为小米品牌且非曲面屏。

### Query 5: 反思循环触发 ⭐
```
预算1500，拍照好，不要曲面屏
```
**预期行为**：1500 元价位段拍照机型选择有限，Retriever 返回的候选可能无法充分满足「拍照好」的需求。Critic 的「需求覆盖」或「推荐完整性」检查未通过，输出修改意见触发回退。第二轮 Retriever 根据修改意见调整策略后，Critic 通过，最终输出带有 2 轮反思日志。

> 💡 这个查询展示了系统的核心亮点：**Critic 反思审查 + 条件回退**。在侧边栏的「反思日志」中可以看到每轮检查结果和修改意见。
>
> 反思循环的逻辑已在 `test_graph.py` 中通过 3 项单元测试验证：
> - `test_should_continue_fail_retry` — Critic 未通过时正确回退到 Retriever
> - `test_should_continue_fail_max_iterations` — 达到最大迭代时强制结束
> - `test_reflection_log_accumulation` — 反思日志正确累积多轮迭代

---

## 项目结构

```
RAG__learn/
├── app.py                    # Streamlit 前端入口
├── config.py                 # 全局配置（API Key、模型、检索参数）
├── requirements.txt          # Python 依赖
├── .env                      # API Key（不提交到 Git）
├── data/
│   ├── products.json         # 手机商品数据（120 条）
│   └── chroma_db/            # ChromaDB 向量库
├── data_generator/
│   ├── schemas.py            # PhoneProduct / Review Pydantic 模型
│   ├── generator.py          # AI 数据生成器
│   └── __main__.py           # python -m data_generator
├── rag/
│   ├── indexer.py            # 数据入库 pipeline（Parent-Child 切分 + DashScope 嵌入）
│   └── retriever.py          # 混合检索器（向量 + 元数据 + BM25 + RRF）
├── agents/
│   ├── planner.py            # Planner Agent（用户意图解析）
│   ├── generator.py          # Generator Agent（推荐生成）
│   ├── critic.py             # Critic Agent（6 项结构化反思审查）
│   └── graph.py              # LangGraph 状态图组装
└── tests/
    ├── test_planner.py       # Planner 测试（5 项）
    ├── test_generator.py     # Generator 测试（6 项）
    ├── test_retriever.py     # Retriever 测试（6 项）
    ├── test_critic.py        # Critic 测试（14 项）
    ├── test_graph.py         # Graph 测试（12 项）
    └── test_ui.py            # UI 测试（5 项）
```

---

## 技术亮点

1. **RAG 混合检索** — 向量语义检索 + 元数据硬过滤 + BM25 关键词检索，RRF 融合排序，兼顾语义理解和精确匹配
2. **Parent-Child Retrieval** — 小粒度 chunk 用于检索，大粒度 parent 用于 LLM 上下文，解决电商长文本噪声问题
3. **Critic 反思审查** — 6 项结构化检查（代码 + LLM），不只是 LLM 自审，而是带形式化验证的审查
4. **条件回退循环** — LangGraph conditional edges 实现「审查不通过 → 打回重做」，最多迭代 N 次
5. **反思可观测性** — 完整的反思日志，用户可看到 Agent 的自我纠正过程

---

## 运行测试

```bash
# 全部测试（48 项）
python -m pytest tests/ -v

# 单模块测试
python -m tests.test_planner
python -m tests.test_generator
python -m tests.test_retriever
python -m tests.test_critic
python -m tests.test_graph
python -m tests.test_ui
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| Agent 框架 | LangChain |
| 向量数据库 | ChromaDB |
| 嵌入模型 | DashScope text-embedding-v4（可替换为任意 OpenAI 兼容 API） |
| 语言模型 | qwen-plus（可替换为 DeepSeek、GPT 等） |
| 前端 | Streamlit |
| 数据校验 | Pydantic |
| 关键词检索 | rank_bm25 (RRF) |
