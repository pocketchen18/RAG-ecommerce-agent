# LangGraph 完全指南：从入门到面试

> 本文档结合「电商导购 Agent」项目实战，帮助初学者系统掌握 LangGraph。
> 学完后能够独立开发 Multi-Agent 系统，并在面试中自信回答相关问题。
> 文档覆盖面：LangChain 基础 → RAG → Agent → LangGraph 核心 → 项目实战 → 面试准备。

---

## 目录

**第一部分：前置知识**
1. [LangChain 基础](#1-langchain-基础)
2. [RAG 检索增强生成](#2-rag-检索增强生成)
3. [Agent 智能体](#3-agent-智能体)

**第二部分：LangGraph 核心**
4. [什么是 LangGraph](#4-什么是-langgraph)
5. [核心概念详解](#5-核心概念详解)
6. [API 速查表](#6-api-速查表)

**第三部分：快速上手**
7. [环境搭建与安装](#7-环境搭建与安装)
8. [最小可运行示例](#8-最小可运行示例)
9. [三种典型模式](#9-三种典型模式)

**第四部分：项目实战深度解析**
10. [项目总览与架构设计](#10-项目总览与架构设计)
11. [状态设计详解](#11-状态设计详解)
12. [五大节点逐行解析](#12-五大节点逐行解析)
13. [条件边与回退机制](#13-条件边与回退机制)
14. [Streamlit 前端集成](#14-streamlit-前端集成)

**第五部分：高级特性**
15. [Human-in-the-Loop](#15-human-in-the-loop)
16. [Streaming 流式输出](#16-streaming-流式输出)
17. [并行执行与 Map-Reduce](#17-并行执行与-map-reduce)
18. [Subgraph 子图](#18-subgraph-子图)
19. [持久化与 Checkpointing](#19-持久化与-checkpointing)

**第六部分：工程实践**
20. [调试与可观测性](#20-调试与可观测性)
21. [测试策略](#21-测试策略)
22. [性能优化](#22-性能优化)
23. [常见错误与排坑](#23-常见错误与排坑)
24. [生产环境部署](#24-生产环境部署)

**第七部分：面试专区**
25. [Agent 基础概念题（12 题）](#25-agent-基础概念题)
26. [技术实现与工具题（10 题）](#26-技术实现与工具题)
27. [项目设计与架构题（10 题）](#27-项目设计与架构题)
28. [场景题与问题解决（10 题）](#28-场景题与问题解决)
29. [代码题（6 题）](#29-代码题)
30. [面试准备清单](#30-面试准备清单)

**附录**
31. [项目代码索引](#31-项目代码索引)
32. [学习资源](#32-学习资源)

---

# 第一部分：前置知识

---

## 1. LangChain 基础

> LangGraph 建立在 LangChain 生态之上，理解 LangChain 基础是学习 LangGraph 的前提。

### 1.1 LangChain 是什么

LangChain 是一个用于构建 LLM 应用的 Python/JavaScript 框架。它提供了：
- **模型抽象层**：统一接口调用 OpenAI、DashScope、DeepSeek 等各种 LLM
- **Prompt 管理**：模板化 prompt，支持变量注入
- **输出解析**：将 LLM 的文本输出转为结构化数据
- **链（Chain）**：将多个组件串联成工作流
- **工具调用**：让 LLM 调用外部工具（搜索、数据库、API）

### 1.2 核心组件

#### ChatOpenAI — 模型调用

```python
from langchain_openai import ChatOpenAI

# 初始化 LLM（兼容 OpenAI 接口的所有模型）
llm = ChatOpenAI(
    model="qwen-plus",           # 模型名称
    temperature=0,                # 0=确定性输出，1=随机性最高
    api_key="sk-xxx",             # API Key
    base_url="https://...",       # API 地址（DashScope/DeepSeek 等）
)

# 调用
response = llm.invoke("你好")
print(response.content)  # "你好！有什么可以帮助你的吗？"
```

**本项目用法**（`agents/planner.py:160-165`）：
```python
self.llm = ChatOpenAI(
    model=self.model_name,        # 默认 qwen-plus
    temperature=self.temperature,  # 默认 0
    api_key=LLM_API_KEY,
    base_url=LLM_API_BASE,
)
```

#### ChatPromptTemplate — Prompt 模板

```python
from langchain_core.prompts import ChatPromptTemplate

# 定义模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}。"),           # 系统消息
    ("human", "请回答：{question}"),           # 用户消息
])

# 填充变量
messages = prompt.invoke({
    "role": "手机推荐专家",
    "question": "3000 元拍照好的手机？"
})
```

**本项目用法**（`agents/planner.py:171-174`）：
```python
self.prompt = ChatPromptTemplate.from_messages([
    ("system", PLANNER_SYSTEM_PROMPT),    # 系统 prompt（意图解析规则）
    ("human", PLANNER_HUMAN_PROMPT),      # 用户消息（包含 query）
])
```

#### PydanticOutputParser — 结构化输出

这是本项目的**核心技术点**之一。LLM 默认输出纯文本，通过 PydanticOutputParser 可以强制 LLM 输出符合 Pydantic schema 的 JSON。

```python
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# 1. 定义输出 Schema
class PhoneConstraints(BaseModel):
    budget_max: Optional[float] = Field(description="最高预算")
    core_needs: List[str] = Field(description="核心需求")
    brands: List[str] = Field(description="品牌偏好")

# 2. 创建解析器
parser = PydanticOutputParser(pydantic_object=PhoneConstraints)

# 3. 获取格式说明（会自动注入到 prompt 中）
format_instructions = parser.get_format_instructions()
# 输出类似：
# The output should be formatted as a JSON instance...
# {"budget_max": float, "core_needs": ["str"], "brands": ["str"]}

# 4. 调用并解析
result = parser.invoke(llm_response)
print(result.budget_max)   # 3000.0
print(result.core_needs)   # ["拍照"]
```

**本项目用法**（`agents/planner.py:168-169`）：
```python
self.output_parser = PydanticOutputParser(pydantic_object=StructuredConstraints)
```

#### LCEL（LangChain Expression Language）— 管道语法

LCEL 使用 `|` 管道符将组件串联：

```python
# 构建 Chain：Prompt → LLM → Parser
chain = prompt | llm | parser

# 调用 Chain
result = chain.invoke({
    "query": "预算3000拍照好的手机",
    "format_instructions": parser.get_format_instructions(),
})
# result 是 StructuredConstraints 类型
```

**本项目用法**（`agents/planner.py:177`）：
```python
self.chain = self.prompt | self.llm | self.output_parser
```

**理解 LCEL 很重要**，因为 LangGraph 的每个节点本质上就是对这些 Chain 的调用。

### 1.3 TypedDict vs Pydantic BaseModel

LangGraph 的 State 可以用两种方式定义，面试中经常被问到：

```python
# 方式 1：TypedDict（轻量，无校验）
from typing import TypedDict

class State1(TypedDict):
    query: str
    result: str

# 方式 2：Pydantic BaseModel（类型安全，有校验，有默认值）
from pydantic import BaseModel, Field
from typing import Optional, List

class State2(BaseModel):
    query: str = Field(description="用户输入")
    result: str = Field(default="", description="结果")
    iteration: int = Field(default=0)

# 区别：
# TypedDict：
#   - 访问用 state["query"]
#   - 无运行时校验
#   - 轻量，性能好
#
# BaseModel：
#   - 访问用 state.query
#   - 有运行时类型校验
#   - 支持默认值、描述、验证器
#   - 更适合复杂项目
```

**本项目选择 BaseModel**（`agents/graph.py:37-86`），因为：
1. 需要复杂嵌套类型（`Optional[StructuredConstraints]`）
2. 需要默认值（`default_factory=list`）
3. 需要运行时类型校验

### 1.4 LangChain 生态全景

```
┌──────────────────────────────────────────────────────────┐
│                    LangChain 生态                         │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │
│  │ langchain   │  │langchain-   │  │ langchain-   │    │
│  │ (核心)      │  │community    │  │ openai       │    │
│  │             │  │(社区集成)    │  │ (OpenAI集成) │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘    │
│         └────────────────┼────────────────┘             │
│                          ▼                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐    │
│  │ langchain-  │  │ langgraph   │  │ langsmith    │    │
│  │ core        │  │ (图编排)     │  │ (可观测性)   │    │
│  │ (基础组件)  │  │             │  │              │    │
│  └─────────────┘  └─────────────┘  └──────────────┘    │
└──────────────────────────────────────────────────────────┘
```

---

## 2. RAG 检索增强生成

### 2.1 什么是 RAG

RAG（Retrieval-Augmented Generation）= **检索** + **增强** + **生成**。

```
用户提问 → 检索相关知识 → 将知识注入 LLM prompt → LLM 生成回答
```

**为什么需要 RAG？**
- LLM 的知识有截止日期，无法回答最新问题
- LLM 可能"幻觉"——编造不存在的事实
- LLM 不知道你的私有数据（公司文档、商品信息）

**RAG 如何解决？**
- 检索器从你的数据库中找到相关信息
- 将这些信息作为上下文注入 prompt
- LLM 基于这些真实信息生成回答

### 2.2 RAG 核心流程

```
┌─────────────────────────────────────────────────────────┐
│                    RAG 完整流程                          │
│                                                         │
│  【离线索引阶段】                                        │
│  原始文档 → 分块(Chunking) → 向量化(Embedding) → 存储   │
│                                                         │
│  【在线查询阶段】                                        │
│  用户Query → 向量化 → 相似度检索 → Top-K结果 → LLM生成  │
└─────────────────────────────────────────────────────────┘
```

### 2.3 关键概念

#### Embedding（向量化）

将文本转换为高维向量（一组浮点数），使得语义相似的文本在向量空间中距离更近。

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-v4",
    openai_api_key="sk-xxx",
    openai_api_base="https://...",
)

# 将文本转为向量
vector = embeddings.embed_query("拍照好的手机")
# 返回: [0.023, -0.156, 0.089, ...] (1536 维浮点数组)

# 批量向量化
vectors = embeddings.embed_documents(["手机A", "手机B", "手机C"])
```

**本项目用法**（`rag/indexer.py`）使用 DashScope 的 `text-embedding-v4`。

#### Vector Store（向量数据库）

存储和检索向量的专用数据库。常用：ChromaDB、FAISS、Pinecone、Weaviate。

```python
import chromadb

# 创建/加载向量数据库
client = chromadb.PersistentClient(path="./data/chroma_db")
collection = client.get_or_create_collection("products")

# 写入
collection.add(
    documents=["小米14 骁龙8Gen3 2K屏"],
    metadatas=[{"brand": "小米", "price": 3999}],
    ids=["product_001"]
)

# 查询（向量相似度搜索）
results = collection.query(
    query_texts=["拍照好的手机"],
    n_results=5,
)
```

**本项目用法**（`rag/indexer.py`）：Parent-Child 分层切分后写入 ChromaDB。

#### Chunking（分块策略）

将长文档切分为小块，便于检索。

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 固定长度 | 按字符/token 数切分 | 通用文本 |
| 按段落 | 按自然段落切分 | 文章、报告 |
| 语义切分 | 根据语义边界切分 | 长文档 |
| Parent-Child | 小块检索，大块提供上下文 | **电商、问答** |

**本项目用 Parent-Child 策略**：
```
Parent (全文): "小米14 概述... 屏幕... 性能... 拍照... 续航... 评价..."
                    │         │        │        │       │       │
Child chunks:    [概述]    [屏幕]   [性能]   [拍照]  [续航]  [评价]

检索时用 Child（粒度小，匹配精准）
返回时用 Parent（上下文完整，LLM 理解更好）
```

#### Retrieval 策略

| 策略 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| 向量检索 | 语义相似度 | 理解同义词/近义词 | 可能漏掉精确关键词 |
| BM25 | 关键词匹配 | 精确匹配型号/专有名词 | 不理解语义 |
| 元数据过滤 | 结构化条件过滤 | 精确过滤价格/品牌 | 无法处理模糊需求 |
| **混合检索** | **融合多种策略** | **兼顾语义和精确** | **实现复杂** |

**本项目实现混合检索**（`rag/retriever.py`）：
```python
# 三路检索 + RRF 融合
1. 向量检索（语义相似度）
2. 元数据过滤（价格、品牌、屏幕类型等硬性约束）
3. BM25 关键词检索（精确型号匹配）
→ RRF (Reciprocal Rank Fusion) 融合排序 → 最终结果
```

#### RRF（Reciprocal Rank Fusion）融合排序

RRF 是一种将多个排序结果融合为一个排序的算法：

```python
def rrf_score(rank, k=60):
    """RRF 分数计算公式"""
    return 1 / (k + rank)

# 对于每个文档，计算它在各路检索中的 RRF 分数之和
# 最终按总分排序
# k=60 是经验常数，rank 从 1 开始
```

**例子**：
```
文档 A: 向量检索排名 2, BM25 排名 5
  RRF = 1/(60+2) + 1/(60+5) = 0.0161 + 0.0154 = 0.0315

文档 B: 向量检索排名 5, BM25 排名 1
  RRF = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318

→ 文档 B 排名更高（虽然向量检索排名低，但 BM25 精确匹配得分高）
```

### 2.4 本项目的 RAG 架构

```
用户: "预算3000，拍照好的手机"
         │
         ▼
┌─────────────────────────────────────┐
│         HybridRetriever             │
│                                     │
│  1. Planner 解析出约束条件:          │
│     budget_max=3000                 │
│     core_needs=["拍照"]             │
│                                     │
│  2. 三路并行检索:                    │
│     ┌──────────┐                    │
│     │ 向量检索  │ "拍照好的手机"     │
│     └────┬─────┘                    │
│     ┌──────────┐                    │
│     │ 元数据过滤│ price<=3000       │
│     └────┬─────┘                    │
│     ┌──────────┐                    │
│     │ BM25     │ "拍照" 关键词      │
│     └────┬─────┘                    │
│          │                          │
│          ▼                          │
│     RRF 融合排序                     │
│          │                          │
│          ▼                          │
│   返回 Top-K 候选商品                │
└─────────────────────────────────────┘
```

---

## 3. Agent 智能体

### 3.1 什么是 Agent

Agent = **LLM（大脑）** + **工具（手脚）** + **记忆（状态）** + **规划（决策）**

```
┌─────────────────────────────────────┐
│            Agent = LLM + Tools      │
│                                     │
│  ┌─────────┐     ┌───────────┐     │
│  │  LLM    │────→│ 决策引擎   │     │
│  │ (大脑)  │     │           │     │
│  └─────────┘     └─────┬─────┘     │
│                        │           │
│         ┌──────────────┼──────┐    │
│         ▼              ▼      ▼    │
│     ┌──────┐     ┌──────┐ ┌─────┐ │
│     │搜索  │     │数据库│ │ API │ │
│     └──────┘     └──────┘ └─────┘ │
│          工具 (Tools)               │
└─────────────────────────────────────┘
```

### 3.2 Agent 的核心能力

| 能力 | 说明 | 本项目实现 |
|------|------|-----------|
| **规划** | 将复杂任务分解为子任务 | Planner 解析用户意图 |
| **推理** | 根据信息做判断和决策 | Critic 审查推荐质量 |
| **工具调用** | 调用外部工具获取信息 | Retriever 检索商品 |
| **反思** | 检查自身输出，发现并修正错误 | Critic → 回退循环 |
| **记忆** | 维护对话状态和历史 | GraphState |

### 3.3 Agent 模式对比

#### ReAct 模式（Reasoning + Acting）

```
Thought: 我需要搜索3000元拍照好的手机
Action: search("3000元拍照好的手机")
Observation: 找到10个结果...
Thought: 根据结果，我应该推荐小米14...
Action: respond("推荐小米14...")
```

#### 本项目的「生成-批判」模式

```
Planner: 解析意图 → {预算3000, 拍照, 送礼}
Retriever: 检索候选 → [小米14, vivo X100, ...]
Generator: 生成推荐 → "推荐小米14，理由..."
Critic: 审查质量 → ❌ 价格检查未通过（小米14 超预算）
  ↓ 回退
Retriever: 重新检索 → [OPPO Reno11, vivo S18, ...]
Generator: 重新推荐 → "推荐 OPPO Reno11..."
Critic: 审查质量 → ✅ 全部通过
Presenter: 格式化输出
```

### 3.4 什么是 Multi-Agent

Multi-Agent = 多个 Agent 协作完成任务。

| 模式 | 说明 | 例子 |
|------|------|------|
| **串行** | A → B → C，前一个的输出是后一个的输入 | 本项目 |
| **并行** | A、B 同时执行，C 合并结果 | 多路检索 |
| **层级** | Manager Agent 分配任务给 Worker Agent | 任务分解 |
| **辩论** | 多个 Agent 辩论，达成共识 | 多角度分析 |
| **反思** | 一个 Agent 生成，另一个 Agent 审查 | **本项目** |

### 3.5 为什么用 LangGraph 而不是简单的 Chain

```python
# ❌ 简单 Chain：无法实现回退
chain = planner | retriever | generator | critic
# critic 不通过怎么办？Chain 做不到回退

# ✅ LangGraph：支持条件回退
workflow.add_conditional_edges(
    "critic",
    should_continue,
    {
        "retriever": "retriever",  # 不通过 → 回退重做
        "presenter": "presenter",  # 通过 → 继续
    }
)
```

---

# 第二部分：LangGraph 核心

---

## 4. 什么是 LangGraph

### 4.1 定义

LangGraph 是 LangChain 团队开发的一个框架，用于构建**有状态的、多步骤的 AI 应用**。它基于**有向图（Directed Graph）**的概念，让你可以将多个 AI 组件（Agent、工具、函数）连接成一个工作流。

### 4.2 解决了什么问题

```
传统 LangChain Chain:
  prompt | llm | parser  → 只能线性执行，一步接一步

LangGraph:
  节点 + 边 + 条件分支 + 循环  → 支持任意复杂的控制流
```

| 能力 | LangChain Chain | LangGraph |
|------|----------------|-----------|
| 线性执行 | ✅ | ✅ |
| 条件分支 | ❌ | ✅ |
| 循环/重试 | ❌ | ✅ |
| 状态管理 | 有限 | ✅ 完整 |
| 并行执行 | ❌ | ✅ |
| 持久化 | ❌ | ✅ Checkpointing |
| Human-in-the-Loop | ❌ | ✅ |
| 可观测性 | 一般 | ✅ 每步状态可追踪 |

### 4.3 核心设计哲学

LangGraph 的设计灵感来自**状态机（State Machine）**：

```
          ┌──────────────────────────────────────┐
          │          有限状态机                    │
          │                                      │
          │  状态 (State): 系统在某个时刻的数据    │
          │  事件 (Event): 触发状态转换的条件      │
          │  转换 (Transition): 从一个状态到另一个  │
          │                                      │
          │  LangGraph 映射:                      │
          │  State = GraphState                   │
          │  Event = 条件函数返回值                │
          │  Transition = add_conditional_edges   │
          └──────────────────────────────────────┘
```

### 4.4 LangGraph vs 其他框架深度对比

| 维度 | LangGraph | AutoGen | CrewAI | Dify | Coze |
|------|-----------|---------|--------|------|------|
| 开发语言 | Python/JS | Python | Python | Python | 低代码 |
| 核心模型 | 有向图 | 对话 | 角色 | DAG | 流程 |
| 状态管理 | 内置，强大 | 有限 | 有限 | 有 | 有 |
| 循环/回退 | ✅ 原生 | ✅ 对话驱动 | ❌ | 部分 | 部分 |
| 条件分支 | ✅ 原生 | ❌ | ❌ | ✅ | ✅ |
| 并行执行 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Human-in-the-Loop | ✅ 原生 | ✅ | ❌ | ✅ | ✅ |
| 持久化 | ✅ Checkpointing | ❌ | ❌ | ✅ | ✅ |
| 可观测性 | 优秀（LangSmith） | 一般 | 一般 | 好 | 好 |
| 学习曲线 | 中等 | 低 | 低 | 低 | 低 |
| 灵活性 | 极高 | 高 | 中 | 中 | 低 |
| 适合场景 | 复杂工作流 | 多Agent对话 | 任务协作 | 低代码搭建 | 快速原型 |

**选择 LangGraph 的理由**：
1. 需要精细的控制流（条件、循环、回退）
2. 需要完整的状态管理和可观测性
3. 需要与 LangChain 生态无缝集成
4. 需要生产级的可靠性

---

## 5. 核心概念详解

### 5.1 State（状态）— 图中流转的数据

State 是 LangGraph 最核心的概念。它是一个在所有节点之间共享的数据对象。

#### 定义方式

```python
# 方式 1: TypedDict（轻量，推荐入门使用）
from typing import TypedDict, List, Optional

class State(TypedDict):
    query: str                           # 必填字段
    results: List[str]                   # 列表字段
    error: Optional[str]                 # 可选字段

# 方式 2: Pydantic BaseModel（类型安全，推荐生产使用）
from pydantic import BaseModel, Field

class State(BaseModel):
    query: str = Field(description="用户输入")
    results: List[str] = Field(default_factory=list)
    error: Optional[str] = Field(default=None)
    iteration: int = Field(default=0, ge=0, le=10)  # 支持验证

# 方式 3: 带 Annotated 的高级用法（支持自定义合并策略）
from typing import Annotated
from operator import add

class State(TypedDict):
    # messages 字段使用 add 操作符合并（列表拼接）
    messages: Annotated[List[str], add]
```

#### State 的合并机制

当一个节点返回部分更新时，LangGraph 如何合并？

```python
# 当前状态
current = {"query": "hello", "results": ["a"], "iteration": 0}

# 节点返回更新
update = {"results": ["b"], "iteration": 1}

# 合并后的状态（默认行为：覆盖同名字段）
merged = {"query": "hello", "results": ["b"], "iteration": 1}
# 注意：results 被整个覆盖，不是追加！
```

**如果想追加到列表**，需要使用 `Annotated`：

```python
from typing import Annotated
from operator import add

class State(TypedDict):
    results: Annotated[List[str], add]   # 使用 add 操作符

# 现在：
current = {"results": ["a"]}
update = {"results": ["b"]}
merged = {"results": ["a", "b"]}   # 列表拼接！
```

**本项目没有使用 Annotated**（`agents/graph.py:37-86`），因为：
- `reflection_log` 需要手动控制追加逻辑（在节点内 `state.reflection_log + [new_entry]`）
- 其他字段都是覆盖而非追加

### 5.2 Node（节点）— 执行单元

节点是图中实际执行逻辑的地方。每个节点是一个 Python 函数。

#### 节点函数签名

```python
def my_node(state: State) -> dict:
    """
    签名规则：
    - 参数：接收当前 State（完整状态）
    - 返回值：返回 State 的部分更新（dict）
    """
    # 1. 从状态读取输入
    input_data = state["query"]  # TypedDict 用 []
    # 或 input_data = state.query  # BaseModel 用 .

    # 2. 执行业务逻辑
    result = process(input_data)

    # 3. 返回状态更新
    return {"result": result}
```

#### 节点设计原则

```python
# ✅ 好的节点：单一职责
def extract_budget(state: State) -> dict:
    """只负责提取预算"""
    budget = parse_budget(state["query"])
    return {"budget": budget}

# ❌ 不好的节点：职责过多
def do_everything(state: State) -> dict:
    """解析预算 + 检索 + 生成推荐 + 审查"""
    # ... 一大堆逻辑
    # 难以测试、难以复用、难以调试
```

#### 节点可以做哪些事

```python
def versatile_node(state: State) -> dict:
    # 1. 调用 LLM
    response = llm.invoke(state["query"])

    # 2. 调用工具
    results = retriever.search(state["query"])

    # 3. 数据处理
    processed = transform(results)

    # 4. 记录日志
    log_entry = {"node": "my_node", "time": time.time()}

    # 5. 返回多个字段的更新
    return {
        "llm_response": response,
        "search_results": processed,
        "log": state["log"] + [log_entry],
    }
```

### 5.3 Edge（边）— 节点间的连接

#### 普通边（无条件）

```python
# A 执行完，必定执行 B
workflow.add_edge("node_a", "node_b")
```

#### 条件边（根据条件选择路径）

```python
def router(state: State) -> str:
    """条件函数，必须返回一个字符串"""
    if state["score"] > 0.8:
        return "good"        # 返回映射表中的 key
    elif state["iteration"] < 3:
        return "retry"
    else:
        return "fallback"

# 使用条件边
workflow.add_conditional_edges(
    "evaluator",            # 源节点名
    router,                 # 条件函数
    {                       # 映射表：返回值 → 目标节点名
        "good": "presenter",
        "retry": "generator",
        "fallback": "presenter",
    }
)
```

**条件函数的返回值必须在映射表中**，否则会报错。

#### 条件函数可以接收 config

```python
def router(state: State, config: dict) -> str:
    """可以接收运行时配置"""
    max_iter = config.get("configurable", {}).get("max_iterations", 3)
    if state["iteration"] >= max_iter:
        return "end"
    return "continue"
```

### 5.4 START 和 END — 入口和出口

```python
from langgraph.graph import StateGraph, START, END

workflow = StateGraph(State)

# 设置入口（二选一）
workflow.set_entry_point("first_node")          # 方式 1
workflow.add_edge(START, "first_node")           # 方式 2

# 设置出口
workflow.add_edge("last_node", END)              # END 是特殊标记，表示图执行结束
```

### 5.5 Graph（图）— 完整工作流

图 = 节点 + 边 + 入口 + 出口。构建过程：

```python
from langgraph.graph import StateGraph, END

# 1. 创建（传入 State 类型）
workflow = StateGraph(MyState)

# 2. 添加节点
workflow.add_node("name", function)

# 3. 添加边
workflow.add_edge("from", "to")
workflow.add_conditional_edges("from", condition_func, mapping)

# 4. 设置入口
workflow.set_entry_point("start_node")

# 5. 编译（得到可执行的图）
graph = workflow.compile()

# 6. 运行
result = graph.invoke({"query": "hello"})
```

### 5.6 CompiledGraph 的三种调用方式

```python
# 方式 1: invoke — 同步执行，返回最终状态
result = graph.invoke(initial_state)
# result 是完整的最终 State

# 方式 2: stream — 流式执行，返回每步结果
for event in graph.stream(initial_state):
    # event 是一个 dict，key 是节点名，value 是该节点的输出
    for node_name, output in event.items():
        print(f"节点 {node_name} 输出: {output}")

# 方式 3: ainvoke / astream — 异步版本
result = await graph.ainvoke(initial_state)
async for event in graph.astream(initial_state):
    ...
```

---

## 6. API 速查表

### 6.1 StateGraph

```python
from langgraph.graph import StateGraph

# 创建
workflow = StateGraph(state_schema)  # state_schema 是 TypedDict 或 BaseModel

# 添加节点
workflow.add_node(name: str, action: Callable)

# 添加普通边
workflow.add_edge(start_key: str, end_key: str)

# 添加条件边
workflow.add_conditional_edges(
    source: str,                          # 源节点
    path: Callable[[State], str],         # 条件函数
    path_map: dict[str, str],             # 返回值→节点名 映射
)

# 设置入口
workflow.set_entry_point(key: str)

# 设置出口（条件性结束）
workflow.set_finish_point(key: str)

# 编译
graph = workflow.compile(
    checkpointer=None,       # 持久化后端
    interrupt_before=None,   # 在哪些节点前暂停
    interrupt_after=None,    # 在哪些节点后暂停
    debug=False,             # 调试模式
)
```

### 6.2 CompiledGraph

```python
# 同步调用
result: State = graph.invoke(input: State, config: dict = None)

# 流式调用
for event in graph.stream(input: State, config: dict = None):
    # event: dict[str, Any] — 节点名 → 输出

# 异步调用
result = await graph.ainvoke(input: State)

# 获取当前状态
state = graph.get_state(config)

# 更新状态（用于 Human-in-the-Loop）
graph.update_state(config, values: dict)

# 获取状态历史
history = list(graph.get_state_history(config))
```

### 6.3 常用常量

```python
from langgraph.graph import START    # 图的起始标记
from langgraph.graph import END      # 图的终止标记
```

### 6.4 Checkpointer

```python
from langgraph.checkpoint.memory import MemorySaver    # 内存（测试用）
from langgraph.checkpoint.sqlite import SqliteSaver     # SQLite
from langgraph.checkpoint.postgres import PostgresSaver # PostgreSQL

# 使用
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 调用时传入 config（thread_id 标识一个会话）
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input, config=config)
```

---

# 第三部分：快速上手

---

## 7. 环境搭建与安装

### 7.1 安装依赖

```bash
# 核心依赖
pip install langgraph langchain langchain-openai

# 可选依赖
pip install langsmith        # 可观测性平台
pip install chromadb          # 向量数据库
pip install rank-bm25         # BM25 检索
pip install streamlit         # Web UI
```

### 7.2 验证安装

```python
import langgraph
print(langgraph.__version__)

from langgraph.graph import StateGraph, END
print("LangGraph 安装成功！")
```

---

## 8. 最小可运行示例

### 8.1 最简单的图（线性）

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. 定义状态
class State(TypedDict):
    message: str

# 2. 定义节点
def uppercase_node(state: State) -> dict:
    """将消息转为大写"""
    return {"message": state["message"].upper()}

def add_suffix_node(state: State) -> dict:
    """添加后缀"""
    return {"message": state["message"] + " !!!"}

# 3. 构建图
workflow = StateGraph(State)
workflow.add_node("uppercase", uppercase_node)
workflow.add_node("add_suffix", add_suffix_node)
workflow.set_entry_point("uppercase")
workflow.add_edge("uppercase", "add_suffix")
workflow.add_edge("add_suffix", END)

# 4. 编译并运行
graph = workflow.compile()
result = graph.invoke({"message": "hello world"})
print(result["message"])  # "HELLO WORLD !!!"
```

### 8.2 带 LLM 调用的图

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class State(TypedDict):
    topic: str
    outline: str
    content: str

llm = ChatOpenAI(model="qwen-plus", temperature=0)

def outline_node(state: State) -> dict:
    """生成大纲"""
    prompt = f"请为以下主题生成一个文章大纲：{state['topic']}"
    response = llm.invoke(prompt)
    return {"outline": response.content}

def content_node(state: State) -> dict:
    """根据大纲生成内容"""
    prompt = f"根据以下大纲写一篇文章：\n{state['outline']}"
    response = llm.invoke(prompt)
    return {"content": response.content}

workflow = StateGraph(State)
workflow.add_node("outline", outline_node)
workflow.add_node("content", content_node)
workflow.set_entry_point("outline")
workflow.add_edge("outline", "content")
workflow.add_edge("content", END)

graph = workflow.compile()
result = graph.invoke({"topic": "LangGraph 入门", "outline": "", "content": ""})
print(result["content"])
```

### 8.3 带条件分支的图

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    sentiment: str
    response: str

def analyze_sentiment(state: State) -> dict:
    """分析情感"""
    text = state["input"].lower()
    if any(w in text for w in ["好", "棒", "喜欢", "满意"]):
        return {"sentiment": "positive"}
    elif any(w in text for w in ["差", "烂", "失望", "投诉"]):
        return {"sentiment": "negative"}
    else:
        return {"sentiment": "neutral"}

def positive_response(state: State) -> dict:
    return {"response": "感谢您的好评！我们会继续努力！"}

def negative_response(state: State) -> dict:
    return {"response": "非常抱歉给您带来不好的体验，我们会尽快改进！"}

def neutral_response(state: State) -> dict:
    return {"response": "感谢您的反馈，我们会认真对待！"}

def route_by_sentiment(state: State) -> str:
    """根据情感路由"""
    return state["sentiment"]

workflow = StateGraph(State)
workflow.add_node("analyze", analyze_sentiment)
workflow.add_node("positive", positive_response)
workflow.add_node("negative", negative_response)
workflow.add_node("neutral", neutral_response)

workflow.set_entry_point("analyze")

# 条件边
workflow.add_conditional_edges(
    "analyze",
    route_by_sentiment,
    {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    }
)

workflow.add_edge("positive", END)
workflow.add_edge("negative", END)
workflow.add_edge("neutral", END)

graph = workflow.compile()

# 测试
print(graph.invoke({"input": "你们的产品太棒了！", "sentiment": "", "response": ""})["response"])
print(graph.invoke({"input": "太差了，要投诉", "sentiment": "", "response": ""})["response"])
```

---

## 9. 三种典型模式

### 9.1 模式一：线性管道

```
A → B → C → END
```

适用场景：简单的数据处理管道。

```python
workflow.add_edge("A", "B")
workflow.add_edge("B", "C")
workflow.add_edge("C", END)
```

### 9.2 模式二：条件路由

```
A → B → 判断 ─┬→ C → END
               └→ D → END
```

适用场景：根据中间结果选择不同处理路径。

```python
workflow.add_conditional_edges("judge", router, {"c": "C", "d": "D"})
```

### 9.3 模式三：反思循环（本项目使用）

```
A → B → C → 审查 ─┬→ END（通过）
                   └→ B（不通过，最多 N 次）
```

适用场景：需要质量保证，生成后审查，不通过则重做。

```python
workflow.add_conditional_edges(
    "reviewer",
    should_continue,
    {"retry": "generator", "end": END}
)
```

这是本项目的核心模式，也是 LangGraph 最有价值的使用场景之一。

---

# 第四部分：项目实战深度解析

---

## 10. 项目总览与架构设计

### 10.1 项目背景

构建一个**手机品类电商导购 Agent**，用户输入自然语言需求，系统输出个性化的手机推荐。

### 10.2 技术架构

```
┌──────────────────────────────────────────────────────────────┐
│                        用户界面层                             │
│                    Streamlit Chat UI                          │
│              (app.py - 聊天交互 + 结果展示)                    │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    Agent 编排层 (LangGraph)                    │
│                                                              │
│  ┌────────┐  ┌──────────┐  ┌──────────┐  ┌───────┐  ┌────┐ │
│  │Planner │→ │Retriever │→ │Generator │→ │Critic │→ │Pres│ │
│  │意图解析│  │混合检索   │  │推荐生成   │  │反思审查│  │呈现 │ │
│  └────────┘  └────┬─────┘  └──────────┘  └───┬───┘  └────┘ │
│                   │                          │              │
│                   │     ← 未通过时回退 ←       │              │
│                   └──────────────────────────┘              │
│                   (最多迭代 N 次)                             │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    RAG 检索层                                  │
│  ┌────────────┐  ┌────────────┐  ┌──────────┐               │
│  │ 向量检索    │  │ 元数据过滤  │  │ BM25     │               │
│  │ (ChromaDB) │  │ (硬性约束)  │  │ (关键词)  │               │
│  └─────┬──────┘  └─────┬──────┘  └────┬─────┘               │
│        └───────┬───────┘              │                      │
│                ▼                      │                      │
│        ┌────────────┐                 │                      │
│        │ RRF 融合排序│◀────────────────┘                      │
│        └────────────┘                                         │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    数据层                                      │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │ ChromaDB 向量库   │  │ data/products.json│                  │
│  │ (840 商品+907评价)│  │ (120款手机)       │                  │
│  └──────────────────┘  └──────────────────┘                  │
└──────────────────────────────────────────────────────────────┘
```

### 10.3 数据流详解

以用户输入 `"预算3000，送女朋友，主要拍照好，不要曲面屏"` 为例：

```
Step 1: Planner 接收原始 query
  输入: "预算3000，送女朋友，主要拍照好，不要曲面屏"
  输出: StructuredConstraints(
    budget_max=3000,
    scenario="送礼",
    core_needs=["拍照"],
    negative_constraints=["曲面屏"],
    semantic_query="拍照好的手机"
  )

Step 2: Retriever 根据约束检索
  输入: query + search_constraints(price<=3000, exclude_screen=["曲面"])
  输出: [vivo S18, OPPO Reno11, 小米Civi 4 Pro, ...] (8-10个候选)

Step 3: Generator 生成推荐
  输入: query + constraints + candidates
  输出: GeneratorOutput(
    recommendation_text="根据您的需求，推荐以下几款手机...",
    comparison_table="| 机型 | 价格 | 屏幕 | 处理器 | 摄像头 | ..."
  )

Step 4: Critic 审查
  输入: query + constraints + generator_output + candidates
  输出: CriticOutput(
    passed=True,  (或 False)
    checks=[6项检查结果],
    score=8.5,
    revision_notes="所有检查通过" (或具体修改意见)
  )

Step 5a: 如果 Critic 通过 → Presenter 格式化输出
Step 5b: 如果 Critic 未通过 → 回到 Step 2 重新检索（最多N次）
```

---

## 11. 状态设计详解

### 11.1 完整状态定义

```python
# agents/graph.py:37-86

class GraphState(BaseModel):
    """LangGraph 状态图的状态定义"""

    # ── 输入 ──────────────────────────────────────────
    query: str = Field(description="用户原始查询")
    # 作用: 整个流程的原始输入，所有节点都可能用到
    # 示例: "预算3000，送女朋友，主要拍照好，不要曲面屏"

    # ── Planner 输出 ─────────────────────────────────
    constraints: Optional[StructuredConstraints] = Field(
        default=None,
        description="Planner 解析的结构化约束"
    )
    # 作用: 供 Generator 生成推荐、Critic 审查时使用
    # 包含: budget_min, budget_max, scenario, core_needs,
    #       negative_constraints, brands, semantic_query 等

    search_constraints: Optional[SearchConstraints] = Field(
        default=None,
        description="转换后的检索约束"
    )
    # 作用: 供 Retriever 做元数据过滤
    # 包含: budget_max, brands, exclude_screen_keywords 等

    # ── Retriever 输出 ───────────────────────────────
    candidates: List[RetrieverResult] = Field(
        default_factory=list,
        description="检索到的候选商品列表"
    )
    # 作用: 供 Generator 生成推荐、Critic 审查
    # 每个 RetrieverResult 包含: name, price, brand, screen_type,
    #   processor, camera_main, battery, parent_text 等

    # ── Generator 输出 ───────────────────────────────
    generator_output: Optional[GeneratorOutput] = Field(
        default=None,
        description="Generator 生成的推荐文本和对比表格"
    )
    # 作用: 供 Critic 审查、Presenter 格式化
    # 包含: recommendation_text (推荐话术),
    #       comparison_table (Markdown 对比表格)

    # ── Critic 输出 ──────────────────────────────────
    critic_output: Optional[CriticOutput] = Field(
        default=None,
        description="Critic 审查结果"
    )
    # 作用: 供条件边 should_continue 决定流向
    # 包含: passed (bool), checks (6项检查),
    #       revision_notes (修改意见), score (0-10)

    # ── 迭代控制 ─────────────────────────────────────
    iteration: int = Field(default=0, description="当前迭代次数")
    # 作用: 防止无限循环，每经过 Critic 节点 +1

    max_iterations: int = Field(default=3, description="最大迭代次数")
    # 作用: 条件边中检查，达到最大值强制结束

    # ── 反思日志 ─────────────────────────────────────
    reflection_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="每轮迭代的反思日志"
    )
    # 作用: 记录每轮 Critic 的审查结果，用于可观测性
    # 格式: [{"iteration": 1, "passed": False, "score": 6.7,
    #         "checks": [...], "revision_notes": "..."}]

    # ── 最终输出 ─────────────────────────────────────
    final_output: str = Field(default="", description="最终格式化输出")
    # 作用: Presenter 格式化后的最终结果，返回给用户

    class Config:
        arbitrary_types_allowed = True  # 允许使用自定义类型
```

### 11.2 设计决策分析

**为什么将 Planner 输出拆成两个字段？**
```
constraints (StructuredConstraints)  → 给 Generator 和 Critic 用（人类可读的约束）
search_constraints (SearchConstraints) → 给 Retriever 用（机器可读的过滤条件）
```

**为什么 reflection_log 用 `List[Dict]` 而不是 `Annotated[List, add]`？**
```
因为 Critic 节点需要在返回前处理旧日志 + 新条目。
使用 Annotated 的话只能简单拼接，无法做条件处理。
```

**为什么 Config 设置 `arbitrary_types_allowed = True`？**
```
因为 State 中包含了自定义类型（StructuredConstraints、RetrieverResult 等），
Pydantic 默认不允许非标准类型，需要显式开启。
```

---

## 12. 五大节点逐行解析

### 12.1 Planner 节点

```python
# agents/graph.py:90-118

def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Planner 节点：解析用户意图

    输入: state.query (用户原始查询)
    输出: state.constraints, state.search_constraints
    """
    # 打印分隔符，便于调试观察流程
    print("\n" + "="*60)
    print("📝 [Planner] 解析用户意图...")
    print("="*60)

    # 初始化 Planner Agent
    # 注意：每次调用都 new 一个，因为节点应是无状态的
    # 生产环境可考虑使用闭包或全局单例
    planner = PlannerAgent()

    # 从状态读取输入
    query = state.query

    # 执行解析：query → StructuredConstraints
    constraints = planner.parse(query)
    # 内部实现：
    # 1. 将 query + format_instructions 填入 prompt 模板
    # 2. 调用 LLM（qwen-plus）
    # 3. 用 PydanticOutputParser 解析 LLM 输出为 StructuredConstraints

    # 转换为 Retriever 可用的格式
    search_constraints = planner.parse_to_search_constraints(query)
    # 内部实现：
    # 将 StructuredConstraints 的字段映射到 SearchConstraints
    # 例如: negative_constraints=["曲面屏"] → exclude_screen_keywords=["曲面屏"]

    # 打印解析结果（调试用）
    print(f"   预算: {constraints.budget_min} - {constraints.budget_max}")
    print(f"   场景: {constraints.scenario}")
    print(f"   核心需求: {constraints.core_needs}")
    print(f"   负向约束: {constraints.negative_constraints}")
    print(f"   品牌偏好: {constraints.brands}")
    print(f"   语义查询: {constraints.semantic_query}")

    # 返回状态更新（只返回需要更新的字段）
    return {
        "constraints": constraints,
        "search_constraints": search_constraints,
    }
    # LangGraph 会自动将这些字段合并到全局 State 中
```

### 12.2 Retriever 节点

```python
# agents/graph.py:121-146

def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    Retriever 节点：检索候选商品

    输入: state.query, state.search_constraints
    输出: state.candidates

    注意：这个节点可能被执行多次！
    第一次：Planner → Retriever
    回退时：Critic(fail) → Retriever
    """
    print("\n" + "="*60)
    print("🔍 [Retriever] 检索候选商品...")
    print("="*60)

    retriever = HybridRetriever()
    query = state.query
    search_constraints = state.search_constraints

    # 检索策略：
    # 1. 如果有 search_constraints，使用约束过滤
    # 2. 如果没有（理论上不会），使用纯语义检索
    if search_constraints:
        candidates = retriever.search(query, constraints=search_constraints)
        # 内部实现：
        # 1. 向量检索：ChromaDB similarity_search(query, k=20)
        # 2. 元数据过滤：price <= budget_max, brand in brands, ...
        # 3. BM25 检索：关键词匹配
        # 4. RRF 融合：三路结果排序
        # 5. 返回 Top-K
    else:
        candidates = retriever.search(query)

    print(f"   检索到 {len(candidates)} 个候选商品")

    return {
        "candidates": candidates,
    }
```

### 12.3 Generator 节点

```python
# agents/graph.py:149-173

def generator_node(state: GraphState) -> Dict[str, Any]:
    """
    Generator 节点：生成推荐

    输入: state.query, state.constraints, state.candidates
    输出: state.generator_output
    """
    print("\n" + "="*60)
    print("✍️  [Generator] 生成推荐...")
    print("="*60)

    generator = GeneratorAgent()
    query = state.query
    constraints = state.constraints
    candidates = state.candidates

    # 生成推荐
    generator_output = generator.generate(query, constraints, candidates)
    # 内部实现：
    # 1. 将候选商品格式化为文本列表
    # 2. 将约束条件格式化为文本
    # 3. 构建 prompt：用户需求 + 约束 + 候选列表
    # 4. 调用 LLM 生成推荐话术 + 对比表格
    # 5. 通过分隔符 ===RECOMMENDATION=== / ===TABLE=== 解析输出
    #    或自动识别 Markdown 表格

    print(f"   推荐文本长度: {len(generator_output.recommendation_text)} 字符")
    print(f"   对比表格长度: {len(generator_output.comparison_table)} 字符")

    return {
        "generator_output": generator_output,
    }
```

### 12.4 Critic 节点（核心亮点）

```python
# agents/graph.py:176-226

def critic_node(state: GraphState) -> Dict[str, Any]:
    """
    Critic 节点：审查推荐质量

    输入: state.query, state.constraints, state.generator_output,
          state.candidates, state.iteration, state.reflection_log
    输出: state.critic_output, state.reflection_log (追加), state.iteration (+1)

    这是整个项目的核心亮点！
    """
    print("\n" + "="*60)
    print("🔍 [Critic] 审查推荐质量...")
    print("="*60)

    critic = CriticAgent()
    query = state.query
    constraints = state.constraints
    generator_output = state.generator_output
    candidates = state.candidates
    iteration = state.iteration

    # 执行 6 项结构化审查
    critic_output = critic.review(query, constraints, generator_output, candidates)
    # 6 项检查：
    # 1. price_check — 价格是否超预算
    # 2. negative_constraint_check — 是否违反负向约束（如曲面屏）
    # 3. diversity_check — 同品牌是否过多（>3个）
    # 4. needs_coverage_check — 推荐理由是否覆盖核心需求
    # 5. accuracy_check — 对比表格数据是否与候选一致
    # 6. completeness_check — 推荐文本和表格是否完整

    # 构建反思日志条目
    reflection_entry = {
        "iteration": iteration + 1,
        "passed": critic_output.passed,
        "score": critic_output.score,
        "checks": [
            {
                "name": check.name,
                "passed": check.passed,
                "details": check.details,
            }
            for check in critic_output.checks
        ],
        "revision_notes": critic_output.revision_notes,
    }

    # ⚠️ 关键：追加到反思日志，而不是覆盖！
    # 使用列表拼接创建新列表（不可变数据模式）
    new_reflection_log = state.reflection_log + [reflection_entry]

    print(f"   审查结果: {'✅ 通过' if critic_output.passed else '❌ 未通过'}")
    print(f"   质量评分: {critic_output.score}/10")

    if not critic_output.passed:
        print(f"   修改意见: {critic_output.revision_notes[:100]}...")

    return {
        "critic_output": critic_output,
        "reflection_log": new_reflection_log,  # 包含所有历史记录
        "iteration": iteration + 1,             # 迭代计数 +1
    }
```

### 12.5 Presenter 节点

```python
# agents/graph.py:229-315

def presenter_node(state: GraphState) -> Dict[str, Any]:
    """
    Presenter 节点：格式化最终输出

    输入: 所有状态字段
    输出: state.final_output

    这个节点负责将所有结果整合为用户可读的 Markdown 格式。
    """
    print("\n" + "="*60)
    print("📋 [Presenter] 格式化最终输出...")
    print("="*60)

    query = state.query
    constraints = state.constraints
    generator_output = state.generator_output
    critic_output = state.critic_output
    reflection_log = state.reflection_log
    iteration = state.iteration

    # 使用列表收集输出部分，最后 join
    output_parts = []

    # ── 1. 用户需求摘要 ──
    output_parts.append("## 📱 手机推荐结果\n")
    output_parts.append(f"**用户需求**: {query}\n")

    # ── 2. 解析的约束条件 ──
    if constraints:
        output_parts.append("### 📝 解析的约束条件\n")
        if constraints.budget_max:
            output_parts.append(f"- 预算上限: {constraints.budget_max:.0f} 元")
        if constraints.budget_min:
            output_parts.append(f"- 预算下限: {constraints.budget_min:.0f} 元")
        if constraints.scenario:
            output_parts.append(f"- 使用场景: {constraints.scenario}")
        if constraints.core_needs:
            output_parts.append(f"- 核心需求: {', '.join(constraints.core_needs)}")
        if constraints.negative_constraints:
            output_parts.append(f"- 负向约束: {', '.join(constraints.negative_constraints)}")
        if constraints.brands:
            output_parts.append(f"- 品牌偏好: {', '.join(constraints.brands)}")
        output_parts.append("")

    # ── 3. 推荐内容 ──
    if generator_output:
        output_parts.append("### 🎯 个性化推荐\n")
        output_parts.append(generator_output.recommendation_text)
        output_parts.append("")

        output_parts.append("### 📊 对比表格\n")
        output_parts.append(generator_output.comparison_table)
        output_parts.append("")

    # ── 4. 审查结果 ──
    if critic_output:
        output_parts.append("### ✅ 质量审查\n")
        status = "通过" if critic_output.passed else "未通过"
        output_parts.append(f"- 审查结果: {status}")
        output_parts.append(f"- 质量评分: {critic_output.score}/10")
        output_parts.append("")

        output_parts.append("**检查详情**:\n")
        for check in critic_output.checks:
            icon = "✅" if check.passed else "❌"
            output_parts.append(f"- {icon} {check.name}: {check.details}")
        output_parts.append("")

    # ── 5. 反思日志（仅在有迭代时展示） ──
    if len(reflection_log) > 1:
        output_parts.append("### 🔄 反思迭代日志\n")
        output_parts.append(f"共经历 {iteration} 轮迭代:\n")

        for entry in reflection_log:
            status = "✅ 通过" if entry["passed"] else "❌ 未通过"
            output_parts.append(
                f"**第 {entry['iteration']} 轮**: {status} "
                f"(评分: {entry['score']}/10)"
            )
            if not entry["passed"] and entry.get("revision_notes"):
                output_parts.append(
                    f"  修改意见: {entry['revision_notes'][:200]}..."
                )
            output_parts.append("")

    # 拼接所有部分
    final_output = "\n".join(output_parts)

    print(f"   输出长度: {len(final_output)} 字符")

    return {
        "final_output": final_output,
    }
```

---

## 13. 条件边与回退机制

### 13.1 条件边函数

```python
# agents/graph.py:320-342

def should_continue(state: GraphState) -> str:
    """
    条件边：决定 Critic 之后的流向

    决策逻辑（优先级从高到低）：
    1. Critic 通过 → "presenter"
    2. 达到最大迭代 → "presenter"（强制结束，防止死循环）
    3. Critic 未通过 → "retriever"（回退重试）

    返回值必须在 add_conditional_edges 的映射表中
    """
    critic_output = state.critic_output
    iteration = state.iteration
    max_iterations = state.max_iterations

    # 情况 1: Critic 审查通过
    if critic_output.passed:
        print(f"\n✅ Critic 审查通过，进入 Presenter")
        return "presenter"

    # 情况 2: 达到最大迭代次数
    if iteration >= max_iterations:
        print(f"\n⚠️  已达最大迭代次数 ({max_iterations})，强制进入 Presenter")
        return "presenter"

    # 情况 3: 未通过，回退重试
    print(f"\n🔄 Critic 审查未通过，回退到 Retriever 重新执行 "
          f"(迭代 {iteration}/{max_iterations})")
    return "retriever"
```

### 13.2 回退时发生了什么

```
第 1 轮:
  Planner → Retriever → Generator → Critic(❌)
  Critic 输出修改意见: "推荐中包含曲面屏机型，违反负向约束"

回退到 Retriever:
  Retriever 重新检索（这次 search_constraints 中已包含 exclude_screen_keywords）
  → Generator 重新生成
  → Critic 再次审查

第 2 轮:
  Retriever → Generator → Critic(✅)
  → Presenter → END
```

### 13.3 反思日志的累积

```python
# Critic 节点中的关键代码
new_reflection_log = state.reflection_log + [reflection_entry]

# 第 1 轮后: reflection_log = [{iter 1, passed: False, ...}]
# 第 2 轮后: reflection_log = [{iter 1, ...}, {iter 2, passed: True, ...}]

# Presenter 可以展示完整的历史记录
```

---

## 14. Streamlit 前端集成

### 14.1 核心集成代码

```python
# app.py 的核心部分

from agents.graph import build_graph, GraphState

# 构建图（只需一次）
graph = build_graph()

# 用户输入
if prompt := st.chat_input("请描述你的手机需求..."):
    # 创建初始状态
    initial_state = GraphState(
        query=prompt,
        max_iterations=max_iterations,  # 从侧边栏滑块获取
    )

    # 执行图
    final_state = graph.invoke(initial_state)
    # final_state 是一个 dict，包含所有 GraphState 的字段
```

### 14.2 展示反思日志

```python
# 侧边栏展示反思日志
reflection_log = final_state.get("reflection_log", [])
for entry in reflection_log:
    with st.expander(f"第 {entry['iteration']} 轮 - "
                     f"{'✅ 通过' if entry['passed'] else '❌ 未通过'} "
                     f"(评分: {entry['score']}/10)"):
        for check in entry.get("checks", []):
            icon = "✅" if check["passed"] else "❌"
            st.markdown(f"{icon} **{check['name']}**")
            st.caption(check["details"])

        if not entry["passed"] and entry.get("revision_notes"):
            st.markdown("**修改意见:**")
            st.info(entry["revision_notes"])
```

---

# 第五部分：高级特性

---

## 15. Human-in-the-Loop

Human-in-the-Loop 让工作流在某个节点暂停，等待人工输入或审批后再继续。

### 15.1 基本用法

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    content: str
    approved: bool

def generate_node(state: State) -> dict:
    return {"content": "AI 生成的推荐结果..."}

def human_review(state: State) -> dict:
    """这个节点会暂停，等待人工输入"""
    # 实际应用中，这里可能调用 UI 组件
    # 暂时自动通过
    return {"approved": True}

def finalize(state: State) -> dict:
    return {"content": state["content"] + " [已审核]"}

workflow = StateGraph(State)
workflow.add_node("generate", generate_node)
workflow.add_node("review", human_review)
workflow.add_node("finalize", finalize)

workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_conditional_edges(
    "review",
    lambda s: "finalize" if s["approved"] else "generate",
    {"finalize": "finalize", "generate": "generate"}
)
workflow.add_edge("finalize", END)

# 使用 Checkpointer 保存状态
checkpointer = MemorySaver()
graph = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"],  # 在 review 之前暂停
)

# 执行
config = {"configurable": {"thread_id": "session-1"}}
result = graph.invoke({"content": "", "approved": False}, config)

# 获取当前状态（已暂停在 review 节点前）
current = graph.get_state(config)

# 人工审核后，更新状态并继续执行
graph.update_state(config, {"approved": True})
result = graph.invoke(None, config)  # 从暂停处继续
```

### 15.2 实际应用场景

| 场景 | 暂停点 | 人工操作 |
|------|--------|----------|
| 客服系统 | 回复生成后 | 客服确认/修改回复 |
| 代码审查 | PR 生成后 | 审核者批准/驳回 |
| 内容发布 | 内容生成后 | 编辑审核内容 |
| 数据处理 | 高风险操作前 | 管理员确认执行 |

---

## 16. Streaming 流式输出

### 16.1 节点级 Streaming

```python
# stream 返回每个节点的执行结果
for event in graph.stream(initial_state):
    for node_name, output in event.items():
        print(f"\n--- 节点 [{node_name}] 完成 ---")
        print(f"输出: {output}")
```

### 16.2 Token 级 Streaming

```python
# 对于 LLM 调用，可以实现 token 级流式
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="qwen-plus", streaming=True)

# 在节点中使用 stream
def generator_node(state: State) -> dict:
    chunks = []
    for chunk in llm.stream(prompt):
        chunks.append(chunk.content)
        yield chunk.content  # 实时输出每个 token
```

### 16.3 与 Streamlit 集成

```python
import streamlit as st

# 流式显示
placeholder = st.empty()
full_text = ""
for event in graph.stream(initial_state):
    for node_name, output in event.items():
        if "recommendation_text" in str(output):
            full_text = output["recommendation_text"]
            placeholder.markdown(full_text)
```

---

## 17. 并行执行与 Map-Reduce

### 17.1 并行执行

```python
workflow = StateGraph(State)

workflow.add_node("search_web", search_web_node)
workflow.add_node("search_db", search_db_node)
workflow.add_node("search_api", search_api_node)
workflow.add_node("merge", merge_node)

# 从 START 同时出发到三个搜索节点
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_db")
workflow.add_edge(START, "search_api")

# 三个搜索都完成后，进入 merge
workflow.add_edge(["search_web", "search_db", "search_api"], "merge")
workflow.add_edge("merge", END)
```

### 17.2 Map-Reduce 模式

```python
# Map: 对列表中每个元素并行处理
# Reduce: 合并所有结果

def map_node(state: State) -> dict:
    """将大任务拆分为小任务"""
    tasks = split_into_tasks(state["query"])
    return {"tasks": tasks}

def process_single(task: str) -> dict:
    """处理单个任务"""
    return llm.invoke(task)

# LangGraph 的 Send API 支持 Map
from langgraph.types import Send

def route_to_workers(state: State):
    """动态创建并行任务"""
    return [
        Send("worker", {"task": task})
        for task in state["tasks"]
    ]

workflow.add_conditional_edges("mapper", route_to_workers)
```

---

## 18. Subgraph 子图

### 18.1 为什么需要子图

当图变得复杂时，将其拆分为子图可以：
1. **模块化**：每个子图独立开发和测试
2. **复用**：同一个子图可以在多个地方使用
3. **可读性**：主图更简洁

### 18.2 使用方式

```python
# 定义搜索子图
def create_search_subgraph():
    sub = StateGraph(SearchState)
    sub.add_node("rewrite_query", rewrite_node)
    sub.add_node("vector_search", vector_search_node)
    sub.add_node("bm25_search", bm25_search_node)
    sub.add_node("merge", merge_node)

    sub.set_entry_point("rewrite_query")
    sub.add_edge("rewrite_query", "vector_search")
    sub.add_edge("rewrite_query", "bm25_search")
    sub.add_edge(["vector_search", "bm25_search"], "merge")
    sub.add_edge("merge", END)

    return sub.compile()

# 在主图中使用子图
search_graph = create_search_subgraph()

main_workflow = StateGraph(MainState)
main_workflow.add_node("planner", planner_node)
main_workflow.add_node("search", search_graph)  # 子图作为一个节点
main_workflow.add_node("generator", generator_node)

main_workflow.add_edge("planner", "search")
main_workflow.add_edge("search", "generator")
```

---

## 19. 持久化与 Checkpointing

### 19.1 为什么需要持久化

1. **故障恢复**：程序崩溃后可以从上次状态继续
2. **Human-in-the-Loop**：暂停后等待人工输入
3. **历史回溯**：查看工作流的执行历史
4. **多会话**：不同用户/会话的状态独立保存

### 19.2 Checkpointer 类型

```python
# 内存（测试用，重启后丢失）
from langgraph.checkpoint.memory import MemorySaver
checkpointer = MemorySaver()

# SQLite（轻量级，适合单机）
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

# PostgreSQL（生产级，支持并发）
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string("postgresql://...")
```

### 19.3 使用 Checkpointer

```python
# 编译时传入 checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# 执行时传入 config（thread_id 标识会话）
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input, config=config)

# 获取当前状态
state_snapshot = graph.get_state(config)
print(state_snapshot.values)    # 当前状态值
print(state_snapshot.next)      # 下一个要执行的节点

# 获取状态历史
for state in graph.get_state_history(config):
    print(f"节点: {state.metadata.get('source')}, 值: {state.values}")
```

### 19.4 状态回溯

```python
# 获取历史
history = list(graph.get_state_history(config))

# 回溯到某个历史状态（重新执行）
# 通过 to_replay 参数指定从哪个 checkpoint 开始
to_replay = history[-3]  # 倒数第 3 个状态
result = graph.invoke(None, config=to_replay.config)
```

---

# 第六部分：工程实践

---

## 20. 调试与可观测性

### 20.1 打印调试

最简单的方式：在节点中添加 `print` 语句。

```python
def my_node(state: State) -> dict:
    print(f"[my_node] 输入: {state['input'][:50]}...")
    result = process(state["input"])
    print(f"[my_node] 输出: {str(result)[:50]}...")
    return {"result": result}
```

### 20.2 LangSmith 追踪

LangSmith 是 LangChain 官方的可观测性平台。

```bash
pip install langsmith
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY=your-api-key
```

设置好环境变量后，所有 LangChain/LangGraph 调用都会自动追踪。

### 20.3 调试模式

```python
# 开启 debug 模式
graph = workflow.compile(debug=True)

# 或使用 verbose
import langchain
langchain.verbose = True
```

### 20.4 结构化日志

```python
import logging

logger = logging.getLogger(__name__)

def my_node(state: State) -> dict:
    logger.info(f"节点开始: 输入={state['input'][:100]}")
    try:
        result = process(state["input"])
        logger.info(f"节点成功: 输出长度={len(str(result))}")
        return {"result": result}
    except Exception as e:
        logger.error(f"节点失败: {e}", exc_info=True)
        return {"error": str(e)}
```

### 20.5 状态检查工具

```python
# 查看图的结构
print(graph.get_graph().draw_mermaid())

# 查看当前状态
config = {"configurable": {"thread_id": "test"}}
state = graph.get_state(config)
print(f"当前值: {state.values}")
print(f"下一个节点: {state.next}")
```

---

## 21. 测试策略

### 21.1 三层测试

```
┌─────────────────────────────────────┐
│         端到端测试 (E2E)             │  3. 测试完整图流程
│  ┌───────────────────────────────┐  │
│  │      集成测试                  │  │  2. 测试节点组合
│  │  ┌─────────────────────────┐  │  │
│  │  │     单元测试             │  │  │  1. 测试单个节点
│  │  └─────────────────────────┘  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
```

### 21.2 单元测试示例

```python
import pytest
from agents.graph import planner_node, GraphState

def test_planner_basic():
    """测试 Planner 基本功能"""
    state = GraphState(query="预算3000拍照好的手机")
    result = planner_node(state)

    assert "constraints" in result
    assert result["constraints"].budget_max == 3000
    assert "拍照" in result["constraints"].core_needs

def test_planner_no_budget():
    """测试无预算约束"""
    state = GraphState(query="打游戏用的手机")
    result = planner_node(state)

    assert result["constraints"].budget_max is None
    assert "游戏性能" in result["constraints"].core_needs

def test_planner_negative_constraint():
    """测试负向约束解析"""
    state = GraphState(query="不要曲面屏的手机")
    result = planner_node(state)

    assert "曲面屏" in result["constraints"].negative_constraints
```

### 21.3 集成测试示例

```python
def test_planner_to_retriever():
    """测试 Planner → Retriever 的数据流"""
    state = GraphState(query="3000元以内小米手机")

    # 执行 Planner
    planner_result = planner_node(state)

    # 模拟状态更新
    state = state.model_copy(update=planner_result)

    # 执行 Retriever
    retriever_result = retriever_node(state)

    # 验证
    assert len(retriever_result["candidates"]) > 0
    for candidate in retriever_result["candidates"]:
        assert candidate.price <= 3000
```

### 21.4 端到端测试示例

```python
def test_full_graph_normal_path():
    """测试正常路径: Planner → Retriever → Generator → Critic(pass) → Presenter"""
    graph = build_graph()
    result = graph.invoke({
        "query": "3000元以内拍照好的手机",
        "max_iterations": 3,
    })

    assert result["final_output"] != ""
    assert result["iteration"] >= 1
    assert len(result["reflection_log"]) >= 1
    assert result["reflection_log"][-1]["passed"] == True

def test_full_graph_retry_path():
    """测试回退路径: Critic 未通过 → 重新检索"""
    graph = build_graph()
    result = graph.invoke({
        "query": "预算1500，拍照好，不要曲面屏",
        "max_iterations": 3,
    })

    # 可能经历多轮迭代
    assert result["iteration"] >= 1
    # 最终应该有输出
    assert result["final_output"] != ""
```

### 21.5 Mock 测试

```python
from unittest.mock import patch, MagicMock

def test_generator_node_with_mock():
    """使用 Mock 测试 Generator 节点"""
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratorOutput(
        recommendation_text="推荐小米14",
        comparison_table="| 机型 | 价格 |\n| --- | --- |\n| 小米14 | 3999 |"
    )

    with patch("agents.graph.GeneratorAgent", return_value=mock_generator):
        state = GraphState(
            query="测试",
            constraints=StructuredConstraints(),
            candidates=[RetrieverResult(name="小米14", price=3999, ...)],
        )
        result = generator_node(state)

        assert "推荐小米14" in result["generator_output"].recommendation_text
        mock_generator.generate.assert_called_once()
```

---

## 22. 性能优化

### 22.1 LLM 调用优化

```python
# 1. 使用 streaming 减少首字延迟
llm = ChatOpenAI(model="qwen-plus", streaming=True)

# 2. 降低 temperature 提高确定性（减少重试）
llm = ChatOpenAI(temperature=0)

# 3. 使用更小的模型（对于简单任务）
# Planner 用 qwen-plus，Generator 可以用 qwen-turbo

# 4. 缓存 LLM 结果
from langchain.globals import set_llm_cache
from langchain.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

### 22.2 检索优化

```python
# 1. 减少检索数量（减少 LLM 输入 token）
# 从 Top-20 降到 Top-5

# 2. 先过滤再向量搜索（利用 ChromaDB 的 where 参数）
collection.query(
    query_texts=["拍照好的手机"],
    n_results=10,
    where={"price": {"$lte": 3000}},  # 先过滤
)

# 3. 使用 HNSW 索引（ChromaDB 默认支持）

# 4. 批量嵌入（而不是逐条）
embeddings.embed_documents(texts)  # 批量
```

### 22.3 图执行优化

```python
# 1. 并行执行独立节点
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_db")
# 两个搜索并行执行

# 2. 减少不必要的节点
# 如果某个节点的逻辑很简单，可以合并到其他节点

# 3. 提前退出
def early_exit(state: State) -> str:
    if not state["candidates"]:
        return "no_results"  # 直接跳到结束
    return "continue"
```

### 22.4 状态大小优化

```python
# 不要在 State 中存储大量数据
# ❌ 不好
class State(TypedDict):
    all_products: List[dict]  # 100 个商品的完整数据

# ✅ 好
class State(TypedDict):
    candidate_ids: List[str]  # 只存 ID，需要时再查
```

---

## 23. 常见错误与排坑

### 23.1 条件函数返回值不在映射表中

```python
# ❌ 错误：router 返回 "unknown"，但映射表中没有
def router(state):
    return "unknown"

workflow.add_conditional_edges("node", router, {
    "yes": "next",
    "no": "end",
})
# 报错: KeyError: "unknown"

# ✅ 正确：确保所有可能的返回值都在映射表中
def router(state):
    if state["score"] > 0.5:
        return "yes"
    else:
        return "no"  # 所有路径都有对应映射
```

### 23.2 状态字段类型不匹配

```python
# ❌ 错误：State 定义 results 为 List[str]，但节点返回 str
class State(TypedDict):
    results: List[str]

def bad_node(state):
    return {"results": "single string"}  # 应该是 ["single string"]

# ✅ 正确
def good_node(state):
    return {"results": ["single string"]}
```

### 23.3 忘记设置入口点

```python
# ❌ 错误
workflow = StateGraph(State)
workflow.add_node("a", node_a)
workflow.add_node("b", node_b)
workflow.add_edge("a", "b")
graph = workflow.compile()
graph.invoke({"input": "test"})
# 报错: Graph has no entry point

# ✅ 正确
workflow.set_entry_point("a")  # 设置入口
```

### 23.4 状态更新覆盖而非追加

```python
# ❌ 意外行为
class State(TypedDict):
    log: List[str]

def node_a(state):
    return {"log": ["a"]}  # 覆盖了原有 log！

# ✅ 正确：手动追加
def node_a(state):
    return {"log": state["log"] + ["a"]}  # 追加

# ✅ 或使用 Annotated
from typing import Annotated
from operator import add

class State(TypedDict):
    log: Annotated[List[str], add]  # 自动追加
```

### 23.5 节点有副作用

```python
# ❌ 不好：节点修改了外部状态
global_counter = 0

def bad_node(state):
    global_counter += 1  # 副作用！并行执行时会出问题
    return {"count": global_counter}

# ✅ 好：所有状态都在 State 中
class State(TypedDict):
    count: int

def good_node(state):
    return {"count": state["count"] + 1}
```

### 23.6 图没有终止边

```python
# ❌ 错误：最后一个节点没有连到 END
workflow.add_edge("a", "b")
# 执行完 b 后，图不知道该结束了

# ✅ 正确
workflow.add_edge("a", "b")
workflow.add_edge("b", END)
```

### 23.7 循环没有退出条件

```python
# ❌ 危险：无限循环
def should_continue(state):
    return "retry"  # 永远重试

workflow.add_conditional_edges("node", should_continue, {
    "retry": "node",  # 死循环！
})

# ✅ 正确：设置最大迭代次数
def should_continue(state):
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    if state["passed"]:
        return "end"
    return "retry"
```

---

## 24. 生产环境部署

### 24.1 部署架构

```
┌──────────────────────────────────────────────────┐
│                 生产环境架构                       │
│                                                  │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ Nginx   │──→│ FastAPI  │──→│ LangGraph    │ │
│  │ 负载均衡│   │ API 层   │   │ 图执行引擎   │ │
│  └─────────┘   └──────────┘   └──────┬───────┘ │
│                                      │         │
│       ┌──────────────────────────────┼──────┐  │
│       ▼              ▼               ▼      │  │
│  ┌────────┐   ┌──────────┐   ┌───────────┐ │  │
│  │ChromaDB│   │ Redis    │   │ PostgreSQL│ │  │
│  │向量库  │   │ 缓存     │   │Checkpoint │ │  │
│  └────────┘   └──────────┘   └───────────┘ │  │
└──────────────────────────────────────────────────┘
```

### 24.2 FastAPI 封装

```python
from fastapi import FastAPI
from pydantic import BaseModel
from agents.graph import build_graph, GraphState

app = FastAPI()
graph = build_graph()

class QueryRequest(BaseModel):
    query: str
    max_iterations: int = 3

class QueryResponse(BaseModel):
    result: str
    iteration: int
    reflection_log: list

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    initial_state = GraphState(
        query=request.query,
        max_iterations=request.max_iterations,
    )

    # 异步执行
    result = await graph.ainvoke(initial_state)

    return QueryResponse(
        result=result["final_output"],
        iteration=result["iteration"],
        reflection_log=result["reflection_log"],
    )
```

### 24.3 错误处理

```python
@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        result = await asyncio.wait_for(
            graph.ainvoke(initial_state),
            timeout=60.0,  # 超时 60 秒
        )
        return {"result": result["final_output"]}
    except asyncio.TimeoutError:
        raise HTTPException(504, "请求超时")
    except Exception as e:
        logger.error(f"执行失败: {e}", exc_info=True)
        raise HTTPException(500, f"执行失败: {str(e)}")
```

### 24.4 监控指标

```python
# 关键指标
- 请求延迟（P50/P95/P99）
- 迭代次数分布
- Critic 通过率
- LLM 调用次数和延迟
- 检索结果数量
- 错误率
```

### 24.5 成本控制

```python
# 1. 限制最大迭代次数（默认 3）
# 2. 使用更便宜的模型（qwen-turbo vs qwen-plus）
# 3. 缓存 LLM 结果（相同输入不重复调用）
# 4. 减少 prompt 长度（精简 system prompt）
# 5. 限制检索结果数量（Top-5 而非 Top-20）
```

---

# 第七部分：面试专区

---

## 25. Agent 基础概念题

> 本节覆盖 Agent 开发岗位最高频的基础概念面试题，包括 Agent 构成、RAG 核心模块、Function Calling、Think-Execute 机制，以及 LangGraph 核心概念。

### Q1: Agent 的构成与开发流程是什么？

**答**：Agent = **LLM（大脑）** + **工具（手脚）** + **记忆（状态）** + **规划（决策）**。

```
┌─────────────────────────────────────────────┐
│              Agent 构成四要素                 │
│                                             │
│  ┌─────────┐     ┌───────────┐             │
│  │  LLM    │────→│ 规划引擎   │             │
│  │ (大脑)  │     │ (决策)     │             │
│  └─────────┘     └─────┬─────┘             │
│                        │                    │
│         ┌──────────────┼──────┐             │
│         ▼              ▼      ▼             │
│     ┌──────┐     ┌──────┐ ┌──────┐         │
│     │搜索  │     │数据库│ │ API  │         │
│     └──────┘     └──────┘ └──────┘         │
│          工具 (Tools / 手脚)                 │
│                                             │
│  ┌─────────────────────────────────┐       │
│  │ 记忆 (Memory / 状态)            │       │
│  │ 短期记忆 + 工作记忆 + 长期记忆   │       │
│  └─────────────────────────────────┘       │
└─────────────────────────────────────────────┘
```

**标准开发流程**：
1. **需求分析**：明确 Agent 要解决的问题和使用场景
2. **架构设计**：选择单 Agent 或 Multi-Agent 架构
3. **工具定义**：定义 Agent 可调用的工具（搜索、数据库、API）
4. **Prompt 工程**：设计系统 Prompt，定义 Agent 的行为规范
5. **状态设计**：定义 Agent 的记忆结构（短期/长期记忆）
6. **编排实现**：使用 LangGraph 等框架实现工作流
7. **测试验证**：单元测试、集成测试、端到端测试
8. **部署监控**：部署到生产环境，监控性能和成本

**本项目映射**：Planner（规划）→ Retriever（工具）→ Generator（LLM 生成）→ Critic（反思审查）

### Q2: RAG 系统核心模块有哪些？

**答**：RAG 系统由三大核心模块组成：

| 模块 | 功能 | 关键技术 | 本项目实现 |
|------|------|----------|-----------|
| **文档切片** | 将长文档切分为可检索的小块 | 固定长度、按段落、语义切分、Parent-Child | Parent-Child（小块检索精准，大块上下文完整） |
| **检索策略** | 从向量库中找到最相关的文档 | 向量检索、BM25、元数据过滤、混合检索 + RRF | 向量 + BM25 + 元数据过滤 + RRF 融合 |
| **多轮对话治理** | 维护对话上下文，处理追问和澄清 | 对话历史管理、上下文压缩、意图追踪 | GraphState 维护状态 + 反思日志 |

**文档切片策略对比**：

| 策略 | 原理 | 适用场景 |
|------|------|----------|
| 固定长度 | 按字符/token 数切分 | 通用文本 |
| 按段落 | 按自然段落切分 | 文章、报告 |
| 语义切分 | 根据语义边界切分 | 长文档 |
| **Parent-Child** | 小块检索，大块提供上下文 | **电商、问答（本项目）** |

**检索策略对比**：

| 策略 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| 向量检索 | 语义相似度 | 理解同义词 | 可能漏掉精确关键词 |
| BM25 | 关键词匹配 | 精确匹配型号 | 不理解语义 |
| 元数据过滤 | 结构化条件过滤 | 精确过滤价格/品牌 | 无法处理模糊需求 |
| **混合检索 + RRF** | **融合多种策略** | **兼顾语义和精确** | **实现复杂** |

### Q3: Function Calling 底层实现原理是什么？

**答**：Function Calling 是 LLM 调用外部工具的核心机制：

```
┌──────────────────────────────────────────────────────────┐
│                Function Calling 完整流程                   │
│                                                          │
│  1. 工具注册：将工具的 JSON Schema 注册到 LLM              │
│     ┌─────────────────────────────────┐                  │
│     │ name: "search_products"         │                  │
│     │ description: "搜索手机商品"       │                  │
│     │ parameters: {query, max_price}  │                  │
│     └─────────────────────────────────┘                  │
│                                                          │
│  2. 用户输入 → LLM 分析意图 → 决定调用哪个工具              │
│                                                          │
│  3. LLM 生成工具调用参数（JSON 格式）                       │
│     {"name": "search_products",                          │
│      "arguments": {"query": "拍照手机", "max_price": 3000}}│
│                                                          │
│  4. 系统解析并执行工具，获取结果                            │
│                                                          │
│  5. 将工具返回值注入 LLM 上下文                            │
│                                                          │
│  6. LLM 基于工具结果生成最终回答                            │
└──────────────────────────────────────────────────────────┘
```

**底层机制**：
- LLM 的 Prompt 中包含工具的 JSON Schema 描述
- LLM 输出特殊格式的 token 序列表示工具调用（非普通文本）
- 系统解析这个序列，执行对应的工具函数
- 将结果作为新的消息注入对话历史，让 LLM 继续生成

**本项目中**：虽然没有直接使用 Function Calling API，但 Retriever 节点本质上就是 LLM 通过 Planner 输出的结构化约束"调用"检索工具。

### Q4: Think-Execute 机制如何设计？

**答**：Think-Execute = **思考（规划）** + **执行（行动）** 的循环：

```python
# Think 阶段：LLM 分析当前状态并规划下一步
def think(state):
    prompt = f"""
    当前状态: {state}
    可用工具: {tools}
    目标: {goal}
    请分析下一步应该做什么？输出 JSON: {{"action": "...", "params": {...}}}
    """
    plan = llm.invoke(prompt)
    return parse_plan(plan)

# Execute 阶段：执行规划
def execute(plan):
    if plan.action == "search":
        return search(plan.params.query)
    elif plan.action == "calculate":
        return calculate(plan.params.expression)
    elif plan.action == "respond":
        return plan.params.response

# 循环执行（Agent Loop）
while not done:
    plan = think(state)           # 思考
    result = execute(plan)        # 执行
    state = update(state, result) # 更新状态
    if should_stop(state):        # 判断是否结束
        done = True
```

**关键设计点**：
- **Prompt 工程**：清晰定义思考框架和输出格式（JSON Schema）
- **循环控制**：设置最大迭代次数防止死循环
- **状态管理**：维护执行历史，支持回溯和反思
- **错误处理**：工具调用失败时的重试和降级策略
- **终止条件**：明确什么情况下停止循环（达到目标/最大迭代/超时）

**本项目的 Think-Execute**：Planner（Think：解析意图）→ Retriever + Generator（Execute：检索+生成）→ Critic（判断是否停止）

### Q5: LangGraph 是什么？解决了什么问题？

**答**：LangGraph 是 LangChain 团队开发的框架，用于构建**有状态的多步骤 AI 应用**。它基于有向图模型，解决了传统 LangChain Chain 只能线性执行的问题。

| 能力 | LangChain Chain | LangGraph |
|------|----------------|-----------|
| 线性执行 | ✅ | ✅ |
| 条件分支 | ❌ | ✅ |
| 循环/重试 | ❌ | ✅ |
| 状态管理 | 有限 | ✅ 完整 |
| 并行执行 | ❌ | ✅ |
| 持久化 | ❌ | ✅ Checkpointing |
| Human-in-the-Loop | ❌ | ✅ |

**选 LangGraph 的理由**：需要精确控制每一步的逻辑、需要回退重试、需要状态持久化。

### Q6: LangGraph 的核心概念有哪些？

**答**：
| 概念 | 说明 | 本项目示例 |
|------|------|-----------|
| **State** | 在图中流转的数据对象，所有节点共享 | GraphState（query、constraints、candidates 等） |
| **Node** | 执行单元，接收 State，返回 State 的部分更新 | planner_node、retriever_node 等 |
| **Edge** | 节点间的连接，分为普通边和条件边 | should_continue 条件边 |
| **Graph** | 节点和边的集合，定义完整工作流 | build_graph() |
| **START/END** | 图的入口和出口标记 | START → planner → ... → END |
| **Checkpointer** | 状态持久化后端 | MemorySaver（测试）/ PostgresSaver（生产） |

### Q7: State 用 TypedDict 还是 BaseModel？有什么区别？

**答**：

| 维度 | TypedDict | BaseModel |
|------|-----------|-----------|
| 类型检查 | 运行时不检查 | 运行时检查 |
| 访问方式 | `state["key"]` | `state.key` |
| 默认值 | 不支持 | 支持 |
| 验证器 | 不支持 | 支持（`ge`, `le`, 自定义验证器） |
| 性能 | 更好 | 略慢 |
| 嵌套类型 | 有限支持 | 完整支持 |
| 适用场景 | 简单 State、原型 | 复杂 State、生产环境 |

**建议**：简单原型用 TypedDict，生产项目用 BaseModel。

**本项目选择 BaseModel**，因为需要复杂嵌套类型（`Optional[StructuredConstraints]`）、默认值（`default_factory=list`）和运行时类型校验。

### Q8: 条件边和普通边有什么区别？

**答**：
- **普通边**：`add_edge("A", "B")`，A 执行完必定执行 B，无条件
- **条件边**：`add_conditional_edges("A", func, mapping)`，A 执行完根据 `func` 的返回值选择下一个节点

```python
# 普通边：线性执行
workflow.add_edge("planner", "retriever")

# 条件边：根据 Critic 结果选择路径
def should_continue(state: GraphState) -> str:
    if state.critic_output.passed:
        return "presenter"      # 通过 → 输出
    if state.iteration >= state.max_iterations:
        return "presenter"      # 达到最大迭代 → 强制输出
    return "retriever"          # 未通过 → 回退重做

workflow.add_conditional_edges("critic", should_continue, {
    "presenter": "presenter",
    "retriever": "retriever",
})
```

**注意**：条件函数的返回值必须在映射表的 key 中，否则会报 KeyError。

### Q9: State 的合并机制是什么？

**答**：默认行为是**覆盖**——如果节点返回 `{"results": ["b"]}`，原 State 中的 `results` 会被完全替换为 `["b"]`，而不是追加。

```python
# 当前状态
current = {"query": "hello", "results": ["a"], "iteration": 0}

# 节点返回更新
update = {"results": ["b"], "iteration": 1}

# 合并后（覆盖）
merged = {"query": "hello", "results": ["b"], "iteration": 1}
# ⚠️ results 被整个覆盖，不是追加！
```

**如果想追加到列表**，有两种方式：
1. **手动拼接**：`return {"results": state["results"] + ["b"]}`
2. **使用 Annotated**：`results: Annotated[List[str], add]`（自动拼接）

**本项目使用手动拼接**（`state.reflection_log + [new_entry]`），因为需要在追加前做条件处理。

### Q10: 如何实现循环/重试机制？

**答**：使用条件边 + 迭代计数器：

```python
def should_continue(state: State) -> str:
    # 退出条件 1：审查通过
    if state["passed"]:
        return "end"
    # 退出条件 2：达到最大迭代次数（防死循环）
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    # 继续条件：未通过且未达上限
    return "retry"

workflow.add_conditional_edges("reviewer", should_continue, {
    "retry": "generator",  # 回退重做
    "end": END,            # 结束
})
```

**关键设计**：
- 必须有退出条件（通过 或 达到最大迭代），否则会死循环
- 最大迭代次数建议 2-5 次（太低可能质量不够，太高浪费成本）
- 每次迭代应记录反思日志，支持可观测性

### Q11: invoke 和 stream 有什么区别？

**答**：
- `invoke`：同步执行整个图，返回最终 State（适合后端 API）
- `stream`：逐步返回每个节点的执行结果，适合实时展示（适合前端 UI）

```python
# invoke：一次性返回
result = graph.invoke(input)

# stream：逐步返回
for event in graph.stream(input):
    for node_name, output in event.items():
        print(f"{node_name}: {output}")

# ainvoke / astream：异步版本
result = await graph.ainvoke(input)
async for event in graph.astream(input):
    ...
```

**本项目**：Streamlit 前端使用 `invoke`（简单场景），生产 API 推荐 `stream`（更好的用户体验）。

### Q12: LangGraph 和 AutoGen/CrewAI 的核心区别？

**答**：

| 维度 | LangGraph | AutoGen | CrewAI | Dify |
|------|-----------|---------|--------|------|
| 核心模型 | 有向图（状态机） | 对话 | 角色+任务 | DAG |
| 控制流 | 开发者完全控制 | Agent 自主决定 | 框架预设 | 可视化配置 |
| 循环/回退 | ✅ 原生 | ✅ 对话驱动 | ❌ | 部分 |
| 条件分支 | ✅ 原生 | ❌ | ❌ | ✅ |
| 持久化 | ✅ Checkpointing | ❌ | ❌ | ✅ |
| 可观测性 | 优秀（LangSmith） | 一般 | 一般 | 好 |
| 学习曲线 | 中等 | 低 | 低 | 低 |
| 灵活性 | 极高 | 高 | 中 | 中 |
| 适合场景 | 精细控制的复杂工作流 | 自由对话 | 简单任务分配 | 低代码搭建 |

**选 LangGraph 的理由**：需要精确控制每一步的逻辑、需要回退重试、需要状态持久化、需要生产级可靠性。

---

## 26. 技术实现与工具题

> 本节覆盖 Agent 开发中的技术实现细节，包括框架使用、工具治理、数据库选型和优化策略。

### Q13: LangChain 核心模块有哪些？各自作用是什么？

**答**：LangChain 核心模块：

| 模块 | 功能 | 本项目用法 |
|------|------|-----------|
| **ChatOpenAI** | 统一接口调用各种 LLM（OpenAI 兼容） | 调用 DashScope qwen-plus |
| **ChatPromptTemplate** | Prompt 模板管理，支持变量注入 | 定义 Planner/Critic 的系统 Prompt |
| **PydanticOutputParser** | 将 LLM 文本输出解析为 Pydantic 对象 | 解析输出为 StructuredConstraints |
| **LCEL** | 管道语法 `\|` 串联组件 | `prompt \| llm \| parser` |
| **Tools** | 工具定义和调用（Function Calling） | 检索器、数据库查询等 |
| **OutputParser** | 输出解析（JSON/Pydantic/列表） | 结构化输出 |

**LCEL 管道语法**（核心）：
```python
# 构建 Chain：Prompt → LLM → Parser
chain = prompt | llm | parser

# 调用
result = chain.invoke({"query": "3000元拍照手机"})
# result 是 StructuredConstraints 类型，不是字符串
```

### Q14: 异步工具调度如何实现？

**答**：异步工具调度的关键是**并发执行**和**结果收集**：

```python
import asyncio

# 方式 1: asyncio.gather 并行调用
async def parallel_tool_calls(tools: list, query: str):
    tasks = [tool.ainvoke(query) for tool in tools]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Tool {i} failed: {result}")
        else:
            successful.append(result)
    return successful

# 方式 2: LangGraph 并行边
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_db")
workflow.add_edge(["search_web", "search_db"], "merge")  # 等待两个都完成

# 方式 3: LangGraph Send API（动态并行）
from langgraph.types import Send

def route_to_workers(state):
    return [Send("worker", {"task": t}) for t in state["tasks"]]
```

### Q15: 工具失败重试机制如何设计？

**答**：多层重试策略：

```python
from tenacity import retry, stop_after_attempt, wait_exponential

# 1. 函数级重试（tenacity）
@retry(
    stop=stop_after_attempt(3),           # 最多重试 3 次
    wait=wait_exponential(min=1, max=10), # 指数退避 1s, 2s, 4s
    retry=retry_if_exception_type((TimeoutError, ConnectionError))
)
async def robust_tool_call(tool, query):
    return await tool.ainvoke(query)

# 2. 节点级重试（LangGraph 回退）
def should_retry(state):
    if state.get("error") and state["retry_count"] < 3:
        return "retry"
    return "continue"

# 3. 图级重试（Critic 反思）
# 本项目：Critic 审查不通过 → 回退 Retriever 重新执行
```

**重试策略要点**：
- **指数退避**：避免频繁重试导致服务过载（1s → 2s → 4s）
- **选择性重试**：只重试可恢复的错误（超时、网络错误），不重试参数错误
- **熔断机制**：连续失败达到阈值后停止重试，返回降级结果
- **最大次数**：通常 3 次，超过后记录错误并返回默认值

### Q16: 向量数据库如何选型？

**答**：

| 数据库 | 特点 | 适用场景 | 本项目 |
|--------|------|----------|--------|
| **ChromaDB** | 轻量级、嵌入式、Python 原生 | 原型开发、小规模 | ✅ 使用 |
| **FAISS** | Facebook 开源、高性能 | 大规模向量检索 | - |
| **Pinecone** | 全托管、易扩展 | 生产环境、不想运维 | - |
| **Weaviate** | 支持混合搜索（向量+关键词） | 需要混合检索 | - |
| **ES (Elasticsearch)** | 成熟生态、支持向量 | 已有 ES 集群 | - |
| **PostgreSQL + pgvector** | 关系型+向量一体化 | 需要事务+向量 | - |

**选型决策树**：
```
需要生产级？
  ├─ 是 → 已有 ES 集群？ → 是 → ES
  │                    → 否 → Pinecone/Weaviate
  └─ 否 → 数据量 < 100万？ → 是 → ChromaDB
                          → 否 → FAISS
```

**本项目选择 ChromaDB**：轻量级、Python 原生支持、嵌入式部署、适合 demo 项目。

### Q17: Redis 和 MySQL 在 Agent 系统中如何应用？

**答**：

| 维度 | Redis | MySQL |
|------|-------|-------|
| **数据结构** | Key-Value、Hash、List、Set | 关系型表格 |
| **读写性能** | 极高（内存，μs 级） | 中等（磁盘，ms 级） |
| **持久化** | 可选（RDB/AOF） | 必须 |
| **查询能力** | 简单查询 | 复杂 SQL |
| **适用场景** | 缓存、会话、消息队列 | 持久化、审计、复杂查询 |

**Agent 系统中的应用**：
- **Redis**：
  - 对话历史缓存（TTL 过期）
  - 工具调用结果缓存（避免重复调用）
  - 限流计数器（滑动窗口）
  - 实时状态（当前正在执行的节点）
- **MySQL**：
  - 用户数据持久化
  - 对话记录归档
  - 审计日志
  - 配置管理

```python
# Redis 缓存 LLM 调用结果
import redis, hashlib

r = redis.Redis()

def cached_llm_call(query: str):
    cache_key = f"llm:{hashlib.md5(query.encode()).hexdigest()}"
    cached = r.get(cache_key)
    if cached:
        return cached.decode()  # 缓存命中
    
    result = llm.invoke(query)
    r.setex(cache_key, 3600, result)  # 缓存 1 小时
    return result
```

### Q18: 上下文压缩策略有哪些？

**答**：上下文压缩解决 Token 限制问题（上下文窗口有限）：

| 策略 | 原理 | 适用场景 | 实现复杂度 |
|------|------|----------|-----------|
| **截断** | 保留最近 N 条消息 | 简单对话 | 低 |
| **摘要** | LLM 生成历史摘要 | 长对话 | 中 |
| **滑动窗口** | 保留最近 N 轮 + 系统 Prompt | 多轮对话 | 低 |
| **选择性保留** | 根据相关性筛选历史 | 复杂对话 | 高 |
| **分层记忆** | 短期（完整）+ 长期（摘要） | 生产系统 | 高 |

**本项目的压缩策略**：
```python
# 1. Planner 将自然语言压缩为结构化约束（减少 token）
constraints = planner.parse("预算3000，送女朋友，拍照好，不要曲面屏")
# → StructuredConstraints(budget_max=3000, scenario="送礼", ...)
# 原始 query 几十个字 → 结构化 JSON 几十个 token

# 2. Retriever 只返回 Top-K 候选（减少上下文）
candidates = retriever.search(query, k=5)  # 只取 5 个，不是全部

# 3. Generator 只接收约束 + 候选（不接收原始历史）
generator.generate(query, constraints, candidates)
```

### Q19: SSE 流式推送如何实现？

**答**：SSE（Server-Sent Events）实现服务端到客户端的实时推送：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

async def event_generator(query: str):
    """生成 SSE 事件流"""
    # 1. 发送开始事件
    yield f"data: {json.dumps({'type': 'start'})}\n\n"
    
    # 2. 流式执行图
    for event in graph.stream({"query": query}):
        for node_name, output in event.items():
            yield f"data: {json.dumps({'type': 'node', 'node': node_name, 'output': str(output)[:200]})}\n\n"
    
    # 3. 发送结束事件
    yield f"data: {json.dumps({'type': 'end'})}\n\n"

@app.get("/stream")
async def stream(query: str):
    return StreamingResponse(
        event_generator(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

**前端接收**：
```javascript
const es = new EventSource('/stream?query=推荐手机');
es.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.type === 'node') updateUI(data.node, data.output);
    if (data.type === 'end') es.close();
};
```

**与 WebSocket 的区别**：SSE 是单向（服务端→客户端），WebSocket 是双向。Agent 流式输出用 SSE 即可。

### Q20: Checkpoint 存储机制如何工作？

**答**：Checkpoint 实现状态持久化和恢复：

```python
from langgraph.checkpoint.memory import MemorySaver      # 内存（测试）
from langgraph.checkpoint.sqlite import SqliteSaver       # SQLite（单机）
from langgraph.checkpoint.postgres import PostgresSaver   # PostgreSQL（生产）

# 编译时传入 checkpointer
graph = workflow.compile(checkpointer=MemorySaver())

# 执行时传入 config（thread_id 标识会话）
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input, config=config)

# 获取当前状态
state = graph.get_state(config)
print(state.values)    # 当前状态值
print(state.next)      # 下一个要执行的节点

# 获取状态历史（回溯）
for state in graph.get_state_history(config):
    print(f"节点: {state.metadata.get('source')}, 值: {state.values}")
```

**Checkpoint 的四大作用**：
1. **故障恢复**：程序崩溃后可以从上次状态继续
2. **Human-in-the-Loop**：暂停后等待人工输入
3. **历史回溯**：查看工作流的执行历史
4. **多会话隔离**：不同用户/会话的状态独立保存

### Q21: 如何实现状态监控与可观测性？

**答**：多层监控体系：

```python
# 1. 结构化日志
import logging
logger = logging.getLogger(__name__)

def monitored_node(state: State) -> dict:
    start = time.time()
    logger.info(f"节点开始: input={state['input'][:100]}")
    try:
        result = process(state["input"])
        duration = time.time() - start
        logger.info(f"节点成功: duration={duration:.2f}s")
        return {"result": result}
    except Exception as e:
        logger.error(f"节点失败: {e}", exc_info=True)
        return {"error": str(e)}

# 2. LangSmith 追踪（LangChain 官方平台）
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
# 所有 LangChain/LangGraph 调用自动追踪

# 3. 关键监控指标
metrics = {
    "请求延迟": "P50/P95/P99",
    "迭代次数": "分布直方图",
    "Critic 通过率": "百分比",
    "LLM 调用次数": "每请求平均",
    "LLM 延迟": "P50/P95",
    "检索结果数": "Top-K 分布",
    "错误率": "百分比",
}
```

### Q22: Function Calling 工具如何注册？

**答**：工具注册让 LLM 知道有哪些工具可用：

```python
from langchain_core.tools import tool

# 方式 1: @tool 装饰器（简单）
@tool
def search_products(query: str, max_price: float = None) -> list:
    """搜索手机商品
    
    Args:
        query: 搜索关键词（如"拍照好的手机"）
        max_price: 最高价格限制（如 3000）
    """
    return retriever.search(query, max_price=max_price)

# 方式 2: StructuredTool（更灵活）
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    max_price: float = Field(default=None, description="最高价格")

search_tool = StructuredTool.from_function(
    func=search_products,
    name="search_products",
    description="搜索手机商品",
    args_schema=SearchInput
)

# 绑定到 LLM
llm_with_tools = llm.bind_tools([search_tool, calculate_tool, ...])

# LLM 会自动决定何时调用哪个工具
response = llm_with_tools.invoke("推荐 3000 元以内的拍照手机")
```

**工具描述的重要性**：LLM 根据工具的 `name` 和 `description` 决定是否调用，描述不清会导致调用错误。

---

## 27. 项目设计与架构题

> 本节覆盖 Agent 系统的架构设计、Multi-Agent 协作、内存治理和模型选型等高级设计题。

### Q23: Multi-Agent 协作流程如何设计？

**答**：Multi-Agent 协作的五种模式：

| 模式 | 说明 | 适用场景 | 本项目 |
|------|------|----------|--------|
| **串行** | A → B → C，前一个的输出是后一个的输入 | 流水线处理 | ✅ Planner → Retriever → Generator |
| **并行** | A、B 同时执行，C 合并结果 | 多路检索 | ✅ 向量 + BM25 + 元数据并行 |
| **层级** | Manager Agent 分配任务给 Worker Agent | 任务分解 | - |
| **辩论** | 多个 Agent 辩论，达成共识 | 多角度分析 | - |
| **反思** | 一个 Agent 生成，另一个 Agent 审查 | 质量保证 | ✅ Generator → Critic |

**本项目的「生成-批判」双 Agent 模式**：
```
Planner（规划）→ Retriever（检索）→ Generator（生成）→ Critic（审查）
                                                          ↓
                                                  通过 → Presenter
                                                  未通过 → 回退 Retriever（最多 N 次）
```

### Q24: 系统架构如何分层？

**答**：Agent 系统的四层架构：

```
┌─────────────────────────────────────────┐
│           用户界面层 (UI Layer)          │
│    Streamlit / Web / Mobile / API GW    │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│         编排层 (Orchestration Layer)      │
│    LangGraph / 状态机 / 工作流引擎       │
│    ┌────────┐ ┌────────┐ ┌────────┐    │
│    │Planner │ │Generator│ │Critic │    │
│    └────────┘ └────────┘ └────────┘    │
│    状态管理 · 条件路由 · 循环控制        │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           工具层 (Tool Layer)            │
│    检索器 / 数据库 / API / 搜索引擎      │
│    异步调度 · 重试 · 缓存 · 限流         │
└─────────────────┬───────────────────────┘
                  │
┌─────────────────▼───────────────────────┐
│           模型层 (Model Layer)           │
│    LLM / Embedding / Reranker           │
│    负载均衡 · 降级 · 成本控制             │
└─────────────────────────────────────────┘
```

**各层职责**：
- **UI 层**：用户交互、结果展示、状态可视化
- **编排层**：任务分解、节点调度、状态管理、条件路由
- **工具层**：数据检索、外部 API 调用、数据库操作
- **模型层**：LLM 推理、向量化、重排序

### Q25: 分层 Memory 设计如何实现？

**答**：Agent 的三层记忆系统：

```python
class AgentMemory:
    def __init__(self):
        # 短期记忆：当前对话的完整历史（最近 N 轮）
        self.short_term = []      # List[Message]
        
        # 工作记忆：当前任务的关键信息（中间状态）
        self.working = {}         # Dict[str, Any]
        
        # 长期记忆：历史对话的摘要（持久化）
        self.long_term = []       # List[Summary]
    
    def add_message(self, role: str, content: str):
        self.short_term.append({"role": role, "content": content})
        # 超过阈值时压缩到长期记忆
        if len(self.short_term) > 20:
            self._compress()
    
    def _compress(self):
        old = self.short_term[:10]
        summary = llm.invoke(f"总结：{old}")
        self.long_term.append(summary)
        self.short_term = self.short_term[10:]
    
    def get_context(self):
        context = []
        if self.working:
            context.append(f"当前任务：{self.working}")
        if self.long_term:
            context.append(f"历史摘要：{self.long_term[-1]}")
        context.extend(self.short_term[-6:])
        return context
```

**本项目的记忆实现**：
- **短期记忆**：GraphState 中的 `query`、`constraints`、`candidates`
- **工作记忆**：GraphState 中的 `iteration`、`reflection_log`
- **长期记忆**：Checkpoint 持久化（thread_id 隔离不同会话）

### Q26: 历史对话检索如何实现？

**答**：从历史对话中检索相关信息：

```python
from langchain.vectorstores import Chroma

class ConversationRetriever:
    def __init__(self):
        self.vectorstore = Chroma(collection_name="conversations")
    
    def index_conversation(self, thread_id: str, messages: list):
        """索引对话历史"""
        for msg in messages:
            self.vectorstore.add_texts(
                texts=[msg["content"]],
                metadatas=[{
                    "thread_id": thread_id,
                    "role": msg["role"],
                    "timestamp": msg.get("timestamp")
                }]
            )
    
    def search_history(self, query: str, thread_id: str = None, k: int = 5):
        """搜索历史对话"""
        filter_dict = {"thread_id": thread_id} if thread_id else None
        return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)

# 使用
retriever = ConversationRetriever()
relevant = retriever.search_history("之前推荐的手机", thread_id="user-123")
```

**关键设计**：
- 按 thread_id 隔离不同用户的对话
- 用向量相似度搜索找到相关历史
- 结合时间衰减（越近的历史权重越高）

### Q27: 模型选型标准是什么？

**答**：模型选型的四维评估框架：

| 维度 | 考量因素 | 权重 | 本项目选择 |
|------|----------|------|-----------|
| **成本** | Token 价格、调用频率、月度预算 | 高 | qwen-plus（¥0.8/千 token） |
| **性能** | 准确性、推理能力、上下文长度 | 高 | 128K 上下文 |
| **延迟** | 首字延迟、生成速度、并发能力 | 中 | 流式输出 |
| **可靠性** | API 稳定性、错误率、SLA | 高 | 阿里云 SLA |

**选型策略**：
- **复杂任务**（Planner/Critic）：用强模型（qwen-plus/GPT-4）
- **简单任务**（格式化/摘要）：用轻量模型（qwen-turbo/GPT-3.5）
- **嵌入模型**：专用嵌入模型（text-embedding-v4）比通用 LLM 更高效
- **成本控制**：缓存相同输入、限制最大迭代、精简 Prompt

### Q28: 如何设计一个带质量审查的推荐系统？

**答**（参考本项目）：
1. **Planner**：解析用户意图，输出结构化约束
2. **Retriever**：根据约束检索候选
3. **Generator**：根据候选生成推荐
4. **Critic**：用代码规则 + LLM 审查推荐质量（6 项检查）
5. **条件边**：Critic 通过 → 输出；不通过 → 回退 Retriever（最多 N 次）

**关键设计决策**：
- Critic 用**代码规则**（而非纯 LLM），更可靠、可解释
- 设置**最大迭代次数**，防止死循环（本项目默认 3 次）
- 记录**反思日志**，支持可观测性和调试
- 6 项检查：价格、负向约束、多样性、需求覆盖、准确性、完整性

### Q29: 如何设计一个 RAG + Agent 的系统？

**答**：
```
用户查询
  → Planner（意图解析 + 查询改写）
  → Retriever（混合检索：向量 + BM25 + 元数据过滤）
  → Reranker（重排序，可选）
  → Generator（基于检索结果生成回答）
  → Critic（验证回答是否基于检索结果，是否有幻觉）
  → 输出
```

**关键设计点**：
- Planner 的查询改写可以提升检索质量
- 混合检索（向量 + 关键词）比单一检索效果好
- Reranker 可以进一步提升相关性（但增加延迟）
- Critic 验证回答是否有幻觉（是否基于检索结果）

### Q30: 如何设计一个多轮对话 Agent？

**答**：
```python
class State(TypedDict):
    messages: List[dict]       # 对话历史
    current_intent: str        # 当前意图
    context: dict              # 上下文信息
    response: str              # 当前回复

# 使用 Checkpointer 保存对话状态
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input, config=config)

# 下一轮对话时，自动加载历史状态
result = graph.invoke(new_input, config=config)
```

**关键设计**：
- 使用 Checkpointer + thread_id 实现会话隔离
- 对话历史需要压缩（Token 限制）
- 意图识别需要考虑上下文（追问、澄清）
- 支持对话中断恢复（Checkpoint）

### Q31: 如何设计一个代码审查 Agent？

**答**：
```
PR 代码
  → Analyzer（静态分析 + LLM 理解）
  → Checker × 3（并行：安全性、性能、风格）
  → Merger（合并三路审查结果）
  → Judge（综合判断是否通过）
  → 条件边：
      通过 → Auto-approve
      未通过 → 生成修改建议
      严重问题 → Block PR
```

**关键设计**：
- 三路并行审查提升效率
- 每路审查独立，结果互不影响
- Judge 综合三路结果做最终决策
- 严重问题直接 Block，不走自动流程

### Q32: 如何设计一个 Map-Reduce 风格的文档处理系统？

**答**：
```
输入文档
  → Splitter（拆分为 N 个 chunk）
  → Worker × N（并行处理每个 chunk）
  → Reducer（合并所有结果）
  → 输出
```

```python
# 使用 LangGraph Send API 实现动态并行
from langgraph.types import Send

def route_to_workers(state):
    """动态创建并行任务"""
    return [Send("worker", {"chunk": chunk}) for chunk in state["chunks"]]

workflow.add_conditional_edges("splitter", route_to_workers)
workflow.add_edge("worker", "reducer")
```

**关键设计**：
- Splitter 按语义边界切分（不要切断句子）
- Worker 数量根据 chunk 数量动态调整
- Reducer 需要处理 Worker 失败的情况
- 考虑 chunk 之间的依赖关系（如果有）

---

## 28. 场景题与问题解决

> 本节覆盖 Agent 开发中的高频场景题，包括上下文爆炸、任务调度、对话中断恢复、重复触发处理等实际工程问题。

### Q33: 如何解决上下文爆炸问题？

**答**：上下文爆炸是指 Token 超出模型限制（如 128K），导致无法处理。

```python
# 方案 1: 分块检索（RAG，最常用）
# 不把所有文档放入上下文，只检索最相关的 Top-K
candidates = retriever.search(query, k=5)  # 只取 5 个

# 方案 2: 上下文压缩（摘要）
def compress_context(messages, max_tokens=4000):
    recent = messages[-6:]           # 保留最近 3 轮
    old = messages[:-6]
    if old:
        summary = llm.invoke(f"总结以下对话：{old}")
        return [{"role": "system", "content": f"历史摘要：{summary}"}] + recent
    return recent

# 方案 3: 滑动窗口
def sliding_window(messages, window_size=10):
    return messages[-window_size:]

# 方案 4: 选择性保留（根据相关性筛选）
def selective_retain(messages, query):
    return [m for m in messages if relevance_score(m, query) > 0.5]
```

**本项目的策略**（多层压缩）：
1. Planner 将自然语言压缩为**结构化约束**（几十字 → JSON）
2. Retriever 只返回 **Top-K 候选**（不传递全部商品）
3. Generator 只接收**约束 + 候选**（不接收原始历史）

### Q34: 如何处理耗时任务依赖调度（A→B 任务链）？

**答**：

```python
# 方案 1: 串行执行（简单但慢）
workflow.add_edge("A", "B")
workflow.add_edge("B", "C")

# 方案 2: 并行执行独立任务（推荐）
workflow.add_edge(START, "A1")       # A1 和 A2 并行
workflow.add_edge(START, "A2")
workflow.add_edge(["A1", "A2"], "B") # A1, A2 都完成后才执行 B

# 方案 3: 异步 + 超时控制
import asyncio

async def task_a(state):
    try:
        result = await asyncio.wait_for(long_running_task(), timeout=60.0)
        return {"a_result": result}
    except asyncio.TimeoutError:
        return {"a_result": None, "error": "Task A timeout"}

# 方案 4: 消息队列解耦（大规模场景）
# 使用 Redis/RabbitMQ 作为任务队列
redis.rpush("task_queue", json.dumps({"name": "task_a", "params": {...}}))
```

**关键设计**：
- 识别哪些任务可以并行（无依赖关系）
- 设置超时避免单个任务阻塞整个流程
- 大规模场景用消息队列解耦

### Q35: 多轮对话中断恢复如何实现？

**答**：使用 Checkpoint + thread_id 实现对话恢复：

```python
# 1. 保存对话状态（自动）
config = {"configurable": {"thread_id": "user-123"}}
result = graph.invoke(input, config=config)
# 状态自动保存到 Checkpointer

# 2. 用户中断后重新连接
# 获取上次状态
state_snapshot = graph.get_state(config)
print(f"上次执行到: {state_snapshot.next}")    # 下一个要执行的节点
print(f"状态值: {state_snapshot.values}")       # 当前状态

# 3. 从断点继续执行
result = graph.invoke(None, config=config)  # 传入 None 继续执行

# 4. 或者更新状态后继续（用户输入了新信息）
graph.update_state(config, {"new_input": "用户的新输入"})
result = graph.invoke(None, config=config)
```

**关键点**：
- 使用 **thread_id** 隔离不同用户的对话
- Checkpointer **自动保存**每个节点的执行状态
- 传入 `None` 可以从**上次中断处**继续
- `update_state` 可以在继续前**更新状态**（如注入用户新输入）

### Q36: 用户重复触发工具如何处理？

**答**：防重复触发策略：

```python
import hashlib
from datetime import datetime, timedelta

class ToolDeduplicator:
    def __init__(self, window_seconds=60):
        self.window = timedelta(seconds=window_seconds)
        self.recent_calls = {}  # key -> timestamp
    
    def should_execute(self, tool_name: str, params: dict) -> bool:
        # 生成调用的唯一 key
        call_key = hashlib.md5(
            f"{tool_name}:{json.dumps(params, sort_keys=True)}".encode()
        ).hexdigest()
        
        now = datetime.now()
        
        # 检查是否在窗口内重复调用
        if call_key in self.recent_calls:
            if now - self.recent_calls[call_key] < self.window:
                return False  # 重复调用，跳过
        
        self.recent_calls[call_key] = now
        self._cleanup(now)
        return True
    
    def _cleanup(self, now):
        expired = [k for k, v in self.recent_calls.items() 
                   if now - v > self.window * 2]
        for k in expired:
            del self.recent_calls[k]

# 使用
dedup = ToolDeduplicator(window_seconds=60)

def safe_tool_call(tool_name, params):
    if not dedup.should_execute(tool_name, params):
        return get_cached_result(tool_name, params)  # 返回缓存结果
    return execute_tool(tool_name, params)
```

**其他方案**：
- **Redis 分布式锁**：`redis.setnx(lock_key, 1, ex=60)`
- **幂等性设计**：工具本身支持幂等（相同输入返回相同结果）
- **前端防抖**：按钮点击后禁用 N 秒

### Q37: 如何处理 LLM 调用超时？

**答**：
```python
import asyncio

async def safe_llm_call(state: State) -> dict:
    try:
        result = await asyncio.wait_for(
            llm.ainvoke(state["query"]),
            timeout=30.0,  # 30 秒超时
        )
        return {"response": result.content, "error": None}
    except asyncio.TimeoutError:
        return {"response": None, "error": "LLM 调用超时"}
    except Exception as e:
        return {"response": None, "error": str(e)}

# 在条件边中检查 error
def route(state: State) -> str:
    if state.get("error"):
        return "error_handler"  # 错误处理节点
    return "continue"

# 错误处理节点
def error_handler(state: State) -> dict:
    return {"response": f"抱歉，处理出错了：{state['error']}。请稍后重试。"}
```

### Q38: 如何处理检索结果为空？

**答**：
```python
def retriever_node(state: State) -> dict:
    candidates = retriever.search(state["query"])
    if not candidates:
        return {"candidates": [], "no_results": True}
    return {"candidates": candidates, "no_results": False}

def route(state: State) -> str:
    if state.get("no_results"):
        return "fallback"    # 降级处理
    return "continue"        # 正常流程

# 降级策略
def fallback_node(state: State) -> dict:
    return {
        "response": "抱歉，没有找到完全匹配的商品。以下是热门推荐：",
        "candidates": get_popular_products(k=5),  # 返回热门商品
    }
```

### Q39: 如何实现会话级状态隔离？

**答**：使用 Checkpointer + thread_id：
```python
graph = workflow.compile(checkpointer=MemorySaver())

# 不同用户使用不同 thread_id
user1_config = {"configurable": {"thread_id": "user-1"}}
user2_config = {"configurable": {"thread_id": "user-2"}}

# 互不影响
graph.invoke(input1, config=user1_config)
graph.invoke(input2, config=user2_config)

# 获取用户历史
history1 = list(graph.get_state_history(user1_config))
history2 = list(graph.get_state_history(user2_config))
```

### Q40: 如何实现限流/熔断？

**答**：
```python
from datetime import datetime, timedelta

class RateLimiter:
    """滑动窗口限流"""
    def __init__(self, max_calls: int, period: timedelta):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def allow(self) -> bool:
        now = datetime.now()
        self.calls = [c for c in self.calls if now - c < self.period]
        if len(self.calls) >= self.max_calls:
            return False
        self.calls.append(now)
        return True

class CircuitBreaker:
    """熔断器"""
    def __init__(self, failure_threshold=5, recovery_timeout=60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.last_failure_time = None
        self.state = "closed"  # closed / open / half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if datetime.now() - self.last_failure_time > timedelta(seconds=self.recovery_timeout):
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            raise

# 使用
limiter = RateLimiter(max_calls=10, period=timedelta(minutes=1))
breaker = CircuitBreaker(failure_threshold=5)

def rate_limited_node(state: State) -> dict:
    if not limiter.allow():
        return {"error": "Rate limit exceeded", "retry_after": 60}
    try:
        return breaker.call(llm_call, state)
    except Exception:
        return {"error": "Service unavailable", "fallback": True}
```

### Q41: 如何实现 A/B 测试？

**答**：
```python
import hashlib

def ab_test_router(state: State) -> str:
    """使用用户 ID 的 hash 决定分组，保证同一用户始终在同一组"""
    user_id = state.get("user_id", "anonymous")
    group = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 2
    return "group_a" if group == 0 else "group_b"

workflow.add_conditional_edges("start", ab_test_router, {
    "group_a": "prompt_v1",   # 对照组
    "group_b": "prompt_v2",   # 实验组
})

# 记录分组结果用于分析
def record_assignment(state: State) -> dict:
    return {"ab_group": state.get("ab_group"), "timestamp": datetime.now()}
```

### Q42: 如何实现结果缓存？

**答**：
```python
import hashlib
import json
import redis

# 方式 1: 内存缓存（简单）
memory_cache = {}

def cached_node(state: State) -> dict:
    cache_key = hashlib.md5(state["query"].encode()).hexdigest()
    if cache_key in memory_cache:
        return memory_cache[cache_key]
    
    result = expensive_operation(state)
    memory_cache[cache_key] = result
    return result

# 方式 2: Redis 缓存（生产级，支持分布式）
r = redis.Redis()

def cached_node_redis(state: State) -> dict:
    cache_key = f"agent:{hashlib.md5(state['query'].encode()).hexdigest()}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)
    
    result = expensive_operation(state)
    r.setex(cache_key, 3600, json.dumps(result))  # 缓存 1 小时
    return result
```

---

## 29. 代码题

### Q43: 实现一个简单的线性图

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    input: str
    step1_output: str
    step2_output: str

def step1(state: State) -> dict:
    return {"step1_output": state["input"].upper()}

def step2(state: State) -> dict:
    return {"step2_output": state["step1_output"] + " DONE"}

workflow = StateGraph(State)
workflow.add_node("step1", step1)
workflow.add_node("step2", step2)
workflow.set_entry_point("step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

graph = workflow.compile()
result = graph.invoke({"input": "hello", "step1_output": "", "step2_output": ""})
assert result["step2_output"] == "HELLO DONE"
```

### Q44: 实现一个带条件分支的图

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class State(TypedDict):
    number: int
    result: str

def classify(state: State) -> dict:
    return {"result": "even" if state["number"] % 2 == 0 else "odd"}

def handle_even(state: State) -> dict:
    return {"result": f"{state['number']} is even"}

def handle_odd(state: State) -> dict:
    return {"result": f"{state['number']} is odd"}

def route(state: State) -> str:
    return state["result"]

workflow = StateGraph(State)
workflow.add_node("classify", classify)
workflow.add_node("even", handle_even)
workflow.add_node("odd", handle_odd)
workflow.set_entry_point("classify")
workflow.add_conditional_edges("classify", route, {"even": "even", "odd": "odd"})
workflow.add_edge("even", END)
workflow.add_edge("odd", END)

graph = workflow.compile()
assert graph.invoke({"number": 4, "result": ""})["result"] == "4 is even"
assert graph.invoke({"number": 3, "result": ""})["result"] == "3 is odd"
```

### Q45: 实现一个带重试的图（最多 3 次）

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
import random

class State(TypedDict):
    attempt: int
    max_attempts: int
    success: bool
    result: str

def try_operation(state: State) -> dict:
    """模拟一个可能失败的操作"""
    success = random.random() > 0.5  # 50% 成功率
    return {
        "attempt": state["attempt"] + 1,
        "success": success,
        "result": f"Attempt {state['attempt'] + 1}: {'success' if success else 'failed'}"
    }

def should_retry(state: State) -> str:
    if state["success"]:
        return "end"
    if state["attempt"] >= state["max_attempts"]:
        return "end"  # 达到最大重试次数
    return "retry"

workflow = StateGraph(State)
workflow.add_node("try", try_operation)
workflow.set_entry_point("try")
workflow.add_conditional_edges("try", should_retry, {
    "retry": "try",  # 回到自身，形成循环
    "end": END,
})

graph = workflow.compile()
result = graph.invoke({"attempt": 0, "max_attempts": 3, "success": False, "result": ""})
```

### Q46: 实现一个带并行执行的图

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END, START

class State(TypedDict):
    query: str
    web_results: List[str]
    db_results: List[str]
    merged_results: List[str]

def search_web(state: State) -> dict:
    return {"web_results": [f"web: {state['query']} result 1", "web: result 2"]}

def search_db(state: State) -> dict:
    return {"db_results": [f"db: {state['query']} result 1"]}

def merge(state: State) -> dict:
    return {"merged_results": state["web_results"] + state["db_results"]}

workflow = StateGraph(State)
workflow.add_node("search_web", search_web)
workflow.add_node("search_db", search_db)
workflow.add_node("merge", merge)

# 从 START 同时出发（并行）
workflow.add_edge(START, "search_web")
workflow.add_edge(START, "search_db")

# 两个搜索完成后合并
workflow.add_edge(["search_web", "search_db"], "merge")
workflow.add_edge("merge", END)

graph = workflow.compile()
result = graph.invoke({"query": "test", "web_results": [], "db_results": [], "merged_results": []})
assert len(result["merged_results"]) == 3
```

### Q47: 使用 Pydantic BaseModel 定义 State 并验证

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from langgraph.graph import StateGraph, END

class State(BaseModel):
    query: str = Field(description="用户输入")
    budget: Optional[float] = Field(default=None, ge=0, le=100000)
    results: List[str] = Field(default_factory=list)
    iteration: int = Field(default=0, ge=0)

def parse_budget(state: State) -> dict:
    import re
    match = re.search(r"(\d+)", state.query)
    if match:
        return {"budget": float(match.group(1))}
    return {}

def search(state: State) -> dict:
    return {
        "results": [f"商品A (预算{state.budget})", f"商品B (预算{state.budget})"],
        "iteration": state.iteration + 1,
    }

workflow = StateGraph(State)
workflow.add_node("parse", parse_budget)
workflow.add_node("search", search)
workflow.set_entry_point("parse")
workflow.add_edge("parse", "search")
workflow.add_edge("search", END)

graph = workflow.compile()
result = graph.invoke(State(query="3000元的手机"))
assert result.budget == 3000.0
assert len(result.results) == 2
```

### Q48: 实现反思循环（本项目简化版）

```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, END

class State(TypedDict):
    content: str
    score: float
    iteration: int
    max_iterations: int
    history: List[dict]

def generate(state: State) -> dict:
    """生成内容（迭代次数越多，质量越高）"""
    score = min(0.4 + state["iteration"] * 0.2, 1.0)
    content = f"Generated v{state['iteration'] + 1} (score={score:.1f})"
    return {
        "content": content,
        "score": score,
        "iteration": state["iteration"] + 1,
    }

def review(state: State) -> dict:
    """审查并记录历史"""
    entry = {"iteration": state["iteration"], "score": state["score"],
             "passed": state["score"] >= 0.8}
    return {"history": state["history"] + [entry]}

def should_continue(state: State) -> str:
    if state["score"] >= 0.8:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"
    return "retry"

workflow = StateGraph(State)
workflow.add_node("generate", generate)
workflow.add_node("review", review)
workflow.set_entry_point("generate")
workflow.add_edge("generate", "review")
workflow.add_conditional_edges("review", should_continue, {
    "retry": "generate",
    "end": END,
})

graph = workflow.compile()
result = graph.invoke({
    "content": "", "score": 0, "iteration": 0,
    "max_iterations": 5, "history": []
})
print(f"最终迭代: {result['iteration']}, 最终分数: {result['score']}")
print(f"历史: {result['history']}")
```

---

## 30. 面试准备清单

### Agent 基础概念（必须掌握）

- [ ] 能用一句话解释 Agent 是什么（LLM + 工具 + 记忆 + 规划）
- [ ] 能解释 RAG 系统的三大核心模块（文档切片、检索策略、多轮对话治理）
- [ ] 能解释 Function Calling 的底层实现原理
- [ ] 能解释 Think-Execute 机制的设计思路
- [ ] 能用一句话解释 LangGraph 是什么
- [ ] 能画出 LangGraph 的核心概念图（State、Node、Edge、Graph）
- [ ] 能解释 TypedDict 和 BaseModel 的区别
- [ ] 能解释普通边和条件边的区别
- [ ] 能解释 State 的合并机制（覆盖 vs 追加）
- [ ] 能解释 START 和 END 的作用

### 技术实现（加分项）

- [ ] 能解释 LangChain 核心模块及其作用（ChatOpenAI/PromptTemplate/OutputParser/LCEL）
- [ ] 能解释异步工具调度的实现方式（asyncio.gather / 并行边 / Send API）
- [ ] 能解释工具失败重试机制的设计（指数退避 / 熔断 / 降级）
- [ ] 能解释向量数据库的选型依据（ChromaDB/FAISS/Pinecone/ES/pgvector）
- [ ] 能解释 Redis 和 MySQL 在 Agent 系统中的应用
- [ ] 能解释上下文压缩的几种策略（截断/摘要/滑动窗口/分层记忆）
- [ ] 能解释 SSE 流式推送的实现原理
- [ ] 能解释 Checkpoint 存储机制的工作原理
- [ ] 能解释如何实现状态监控与可观测性
- [ ] 能解释 Function Calling 工具注册方式（@tool / StructuredTool）

### 项目设计（高级项）

- [ ] 能画出本项目的架构图并解释数据流
- [ ] 能解释 Multi-Agent 协作的几种模式（串行/并行/层级/辩论/反思）
- [ ] 能解释系统架构的四层设计（UI/编排/工具/模型）
- [ ] 能解释分层 Memory 设计的实现方式（短期/工作/长期）
- [ ] 能解释历史对话检索的实现方式
- [ ] 能解释模型选型的四维评估标准（成本/性能/延迟/可靠性）
- [ ] 能设计一个带质量审查的推荐系统
- [ ] 能设计一个 RAG + Agent 的系统
- [ ] 能设计一个多轮对话 Agent
- [ ] 能设计一个代码审查 Agent

### 场景题（高级项）

- [ ] 能解释上下文爆炸的解决方案（RAG 分块检索 / 上下文压缩）
- [ ] 能解释耗时任务依赖调度的实现方式（并行/异步/消息队列）
- [ ] 能解释多轮对话中断恢复的实现方式（Checkpoint + thread_id）
- [ ] 能解释用户重复触发工具的处理策略（去重/幂等/防抖）
- [ ] 能解释如何处理 LLM 调用超时
- [ ] 能解释如何处理检索结果为空（降级策略）
- [ ] 能解释如何实现会话级状态隔离
- [ ] 能解释如何实现限流/熔断
- [ ] 能解释如何实现 A/B 测试
- [ ] 能解释如何实现结果缓存（内存/Redis）

### 代码能力（必须掌握）

- [ ] 能实现一个简单的线性图
- [ ] 能实现一个带条件分支的图
- [ ] 能实现一个带循环/重试的图（防死循环）
- [ ] 能实现一个带并行执行的图
- [ ] 能使用 Pydantic BaseModel 定义 State
- [ ] 能实现反思循环（本项目核心模式）

### 工程能力（高级项）

- [ ] 能解释如何测试 LangGraph 应用（单元/集成/端到端）
- [ ] 能解释如何调试和追踪 LangGraph 执行（LangSmith）
- [ ] 能解释如何部署 LangGraph 到生产环境（FastAPI + Checkpoint）
- [ ] 能解释如何处理错误和超时
- [ ] 能解释如何实现 Human-in-the-Loop
- [ ] 能解释如何实现状态持久化

### 面试讲解框架

**5 分钟版**（快速介绍）：
> 本项目是一个基于 LangGraph 的电商导购 Agent，使用「生成-批判」双 Agent 架构。Planner 解析用户意图，Retriever 混合检索商品，Generator 生成推荐，Critic 执行 6 项结构化审查。核心亮点是 Critic 审查不通过时可以自动回退重做，最多迭代 N 次，实现了 Agent 的自我纠错。

**15 分钟版**（深入讲解）：
> 1. 项目背景和目标（1 分钟）
> 2. 技术架构图（2 分钟）
> 3. LangGraph 状态图设计（3 分钟）
> 4. Critic 反思审查机制（3 分钟）
> 5. RAG 混合检索实现（3 分钟）
> 6. 技术亮点和挑战（2 分钟）
> 7. 如果有更多时间，我会...（1 分钟）

---

# 附录

---

## 31. 项目代码索引

| 文件 | 行数 | 作用 | 关键知识点 |
|------|------|------|-----------|
| `agents/graph.py` | ~450 | LangGraph 状态图组装 | StateGraph, add_node, add_conditional_edges, 条件回退 |
| `agents/planner.py` | ~250 | 意图解析 Agent | PydanticOutputParser, ChatPromptTemplate, LCEL |
| `agents/critic.py` | ~590 | 反思审查 Agent | 结构化检查, CheckResult, 修改意见生成 |
| `agents/generator.py` | ~200 | 推荐生成 Agent | LLM 调用, 分隔符解析, Markdown 表格生成 |
| `rag/retriever.py` | ~400 | 混合检索器 | ChromaDB, BM25, RRF 融合, 元数据过滤 |
| `rag/indexer.py` | ~200 | 数据入库 pipeline | Parent-Child 切分, DashScope Embedding |
| `config.py` | ~30 | 全局配置 | dotenv, API Key 管理 |
| `app.py` | ~240 | Streamlit 前端 | graph.invoke(), session_state, 反思日志展示 |
| `data_generator/` | ~300 | AI 数据生成 | Pydantic Schema, JSON 输出 |
| `tests/` | ~800 | 测试套件 | 48 项测试, pytest |

### 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| GraphState 定义 | `agents/graph.py` | 37-86 |
| planner_node | `agents/graph.py` | 90-118 |
| retriever_node | `agents/graph.py` | 121-146 |
| generator_node | `agents/graph.py` | 149-173 |
| critic_node | `agents/graph.py` | 176-226 |
| presenter_node | `agents/graph.py` | 229-315 |
| should_continue 条件边 | `agents/graph.py` | 320-342 |
| build_graph | `agents/graph.py` | 347-388 |
| StructuredConstraints | `agents/planner.py` | 31-82 |
| PlannerAgent.parse | `agents/planner.py` | 179-200 |
| CriticAgent.review | `agents/critic.py` | 398-452 |
| 6 项检查函数 | `agents/critic.py` | 52-310 |
| HybridRetriever.search | `rag/retriever.py` | ~150 |
| RRF 融合排序 | `rag/retriever.py` | ~200 |

---

## 32. 学习资源

### 官方资源

| 资源 | 链接 |
|------|------|
| LangGraph 文档 | https://langchain-ai.github.io/langgraph/ |
| LangGraph GitHub | https://github.com/langchain-ai/langgraph |
| LangGraph 教程 | https://langchain-ai.github.io/langgraph/tutorials/ |
| LangChain 文档 | https://python.langchain.com/docs/ |
| LangSmith（可观测性） | https://smith.langchain.com/ |

### 推荐学习路径

```
Week 1: LangChain 基础
  → 理解 ChatOpenAI, PromptTemplate, OutputParser, LCEL
  → 完成一个简单的 Chain 应用

Week 2: RAG 基础
  → 理解 Embedding, Vector Store, Retriever
  → 完成一个简单的 RAG 问答系统

Week 3: LangGraph 核心
  → 理解 State, Node, Edge, Graph
  → 完成线性图、条件图、循环图各一个

Week 4: 项目实战
  → 阅读并运行本项目代码
  → 理解 Critic 反思审查机制
  → 尝试修改和扩展

Week 5: 高级特性
  → 学习 Human-in-the-Loop, Checkpointing
  → 学习并行执行、Subgraph
  → 尝试部署到生产环境

Week 6: 面试准备
  → 完成本文档所有面试题
  → 准备项目讲解（5 分钟版 + 15 分钟版）
  → 准备系统设计题的回答框架
```

### 面试讲解框架

**5 分钟版**（快速介绍）：
> 本项目是一个基于 LangGraph 的电商导购 Agent，使用「生成-批判」双 Agent 架构。Planner 解析用户意图，Retriever 混合检索商品，Generator 生成推荐，Critic 执行 6 项结构化审查。核心亮点是 Critic 审查不通过时可以自动回退重做，最多迭代 N 次，实现了 Agent 的自我纠错。

**15 分钟版**（深入讲解）：
> 1. 项目背景和目标（1 分钟）
> 2. 技术架构图（2 分钟）
> 3. LangGraph 状态图设计（3 分钟）
> 4. Critic 反思审查机制（3 分钟）
> 5. RAG 混合检索实现（3 分钟）
> 6. 技术亮点和挑战（2 分钟）
> 7. 如果有更多时间，我会...（1 分钟）

---

**文档版本**：v3.0
**最后更新**：2026-06-21
**适用项目**：电商导购 Agent（RAG__learn）
**文档字数**：约 4250+ 行
**面试题总数**：48 题（概念 12 + 技术 10 + 设计 10 + 场景 10 + 代码 6）
