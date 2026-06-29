"""
Planner Agent — 用户意图解析

将自然语言 query 解析为结构化约束条件，包括：
- 预算范围（budget_min, budget_max）
- 使用场景（scenario）
- 核心需求（core_needs）
- 负向约束（negative_constraints）
- 品牌偏好（brands）
- 其他约束

输出符合 StructuredConstraints schema，供下游 Retriever 和 Generator 使用。
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


# ── 结构化输出 Schema ──────────────────────────────────────

class StructuredConstraints(BaseModel):
    """Planner 解析后的结构化约束条件"""

    budget_min: Optional[float] = Field(
        default=None,
        description="最低预算（元），未指定则为 None"
    )
    budget_max: Optional[float] = Field(
        default=None,
        description="最高预算（元），未指定则为 None"
    )
    scenario: Optional[str] = Field(
        default=None,
        description="使用场景，如：送礼、自用、办公、学生等"
    )
    core_needs: List[str] = Field(
        default_factory=list,
        description="核心需求列表，如：['拍照', '游戏性能', '续航', '轻薄']"
    )
    negative_constraints: List[str] = Field(
        default_factory=list,
        description="负向约束列表，如：['曲面屏', '大屏', '苹果']"
    )
    brands: List[str] = Field(
        default_factory=list,
        description="偏好品牌列表，如：['华为', '小米']"
    )
    exclude_brands: List[str] = Field(
        default_factory=list,
        description="排除品牌列表"
    )
    screen_type_keywords: List[str] = Field(
        default_factory=list,
        description="屏幕类型要求，如：['直屏', 'AMOLED']"
    )
    processor_keywords: List[str] = Field(
        default_factory=list,
        description="处理器要求，如：['骁龙8', '天玑9']"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="其他标签，如：['轻薄', '长续航', '快充']"
    )
    semantic_query: str = Field(
        default="",
        description="去除约束后的纯语义查询，用于向量检索"
    )
    reasoning: str = Field(
        default="",
        description="解析推理过程，说明如何从原始 query 得出约束"
    )


# ── Prompt 模板 ──────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT = """你是一个手机推荐系统的意图解析专家。
你的任务是将用户的自然语言需求解析为结构化的约束条件。

## 解析规则

1. **预算解析**：
   - "3000以内" → budget_max=3000
   - "2000-4000" → budget_min=2000, budget_max=4000
   - "预算不限" / "不差钱" → budget_min=None, budget_max=None
   - "便宜的" → budget_max=2000
   - "旗舰" → budget_min=5000

2. **场景识别**：
   - "送女朋友/男朋友" → scenario="送礼"
   - "自己用" → scenario="自用"
   - "给父母" → scenario="送礼"
   - "学生用" → scenario="学生"
   - "办公用" → scenario="办公"

3. **核心需求提取**：
   - "拍照好" → core_needs=["拍照"]
   - "打游戏" / "游戏性能" → core_needs=["游戏性能"]
   - "续航强" / "电池大" → core_needs=["续航"]
   - "轻薄" → core_needs=["轻薄"]
   - "屏幕好" → core_needs=["屏幕质量"]

4. **负向约束识别**：
   - "不要曲面屏" → negative_constraints=["曲面屏"]
   - "不要苹果" → negative_constraints=["苹果"]
   - "不要太大的" → negative_constraints=["大屏"]

5. **品牌识别**：
   - "华为手机" → brands=["华为"]
   - "小米或OPPO" → brands=["小米", "OPPO"]
   - "除了苹果都行" → exclude_brands=["苹果"]

6. **语义查询生成**：
   - 去除约束条件后的核心语义部分
   - "预算3000，拍照好的手机" → semantic_query="拍照好的手机"
   - "打游戏用的，预算不限" → semantic_query="打游戏用的手机"

7. **结合多轮对话上下文**：
   - 用户的当前查询可能是在历史对话基础上的补充或修改。
   - 你需要继承历史对话中的核心约束（如之前的核心需求、品牌偏好等）。
   - 如果当前查询与历史约束冲突（如提出了新的预算、新排除的品牌），用当前查询的条件**覆盖**旧条件。

## 输出格式

{format_instructions}
"""

PLANNER_HUMAN_PROMPT = """{chat_history_section}

## 当前用户需求
"{query}"
"""


# ── Planner Agent ──────────────────────────────────────────────

class PlannerAgent:
    """
    Planner Agent

    将自然语言 query 解析为 StructuredConstraints。
    使用 LLM 进行意图理解和结构化提取。
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0):
        """
        初始化 Planner Agent

        Args:
            model_name: 模型名称，默认使用 config 中的 LLM_MODEL
            temperature: 温度参数，0 表示确定性输出
        """
        import os
        self.model_name = model_name or os.getenv("LLM_MODEL", LLM_MODEL)
        self.temperature = temperature

        import os
        api_key = os.getenv("LLM_API_KEY", LLM_API_KEY)
        if not api_key:
            raise ValueError("未配置 LLM_API_KEY")

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=api_key,
            base_url=os.getenv("LLM_API_BASE", LLM_API_BASE),
        )

        # 初始化输出解析器
        self.output_parser = PydanticOutputParser(pydantic_object=StructuredConstraints)

        # 构建 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", PLANNER_SYSTEM_PROMPT),
            ("human", PLANNER_HUMAN_PROMPT),
        ])

        # 构建 chain
        self.chain = self.prompt | self.llm | self.output_parser

    def parse(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> StructuredConstraints:
        """
        解析用户 query 为结构化约束条件

        Args:
            query: 用户自然语言查询
            chat_history: 历史对话记录，格式如 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns:
            StructuredConstraints: 结构化约束条件
        """
        chat_history_section = ""
        if chat_history:
            history_lines = []
            # 只取最近的 4 条上下文（2轮）
            for msg in chat_history[-4:]:
                role = "用户" if msg["role"] == "user" else "导购"
                history_lines.append(f"{role}: {msg['content']}")
            chat_history_section = "## 历史对话记录\n" + "\n".join(history_lines) + "\n\n"

        try:
            result = self.chain.invoke({
                "query": query,
                "chat_history_section": chat_history_section,
                "format_instructions": self.output_parser.get_format_instructions(),
            })
            return result
        except Exception as e:
            # 解析失败时返回基础约束
            return StructuredConstraints(
                semantic_query=query,
                reasoning=f"解析失败: {str(e)}"
            )

    def parse_to_search_constraints(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None):
        """
        解析用户 query 并转换为 Retriever 的 SearchConstraints 格式

        Args:
            query: 用户自然语言查询
            chat_history: 历史对话记录

        Returns:
            SearchConstraints: 检索器约束条件
        """
        from rag.retriever import SearchConstraints

        structured = self.parse(query, chat_history=chat_history)

        return SearchConstraints(
            budget_max=structured.budget_max,
            budget_min=structured.budget_min,
            brands=structured.brands,
            exclude_brands=structured.exclude_brands,
            screen_type_keywords=structured.screen_type_keywords,
            exclude_screen_keywords=structured.negative_constraints,
            processor_keywords=structured.processor_keywords,
            tags=structured.tags,
            exclude_tags=[],
        )


# ── 测试入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    # 简单测试
    planner = PlannerAgent()

    test_queries = [
        "预算3000，送女朋友，主要拍照好，不要曲面屏",
        "打游戏用的，预算不限",
        "2000-4000元小米手机",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        result = planner.parse(query)
        print(f"预算范围: {result.budget_min} - {result.budget_max}")
        print(f"场景: {result.scenario}")
        print(f"核心需求: {result.core_needs}")
        print(f"负向约束: {result.negative_constraints}")
        print(f"品牌: {result.brands}")
        print(f"语义查询: {result.semantic_query}")
        print(f"推理: {result.reasoning}")
