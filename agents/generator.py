"""
Generator Agent — 推荐生成

根据候选商品列表和用户约束条件，生成：
1. 个性化推荐话术（推荐理由绑定用户具体需求）
2. Markdown 对比表格（机型名、价格、关键参数）

输入：
- 用户原始 query
- Planner 解析的 StructuredConstraints
- Retriever 返回的候选商品列表（RetrieverResult）

输出：
- recommendation_text: 推荐文本
- comparison_table: Markdown 对比表格
"""
import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from agents.planner import StructuredConstraints
from rag.retriever import RetrieverResult


# ── 输出 Schema ──────────────────────────────────────────────

class GeneratorOutput(BaseModel):
    """Generator Agent 的输出"""
    recommendation_text: str = Field(
        description="个性化推荐文本，包含推荐理由和购买建议"
    )
    comparison_table: str = Field(
        description="Markdown 格式的对比表格，包含机型名、价格、关键参数"
    )
    recommended_product_names: List[str] = Field(
        default_factory=list,
        description="推荐的机型名称列表"
    )


# ── 候选商品序列化 ──────────────────────────────────────────────

def _format_candidates(candidates: List[RetrieverResult]) -> str:
    """将候选商品列表格式化为 LLM 可读的文本"""
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"【候选 {i}】{c.name}")
        lines.append(f"  品牌: {c.brand}")
        lines.append(f"  价格: {c.price:.0f} 元")
        lines.append(f"  屏幕: {c.screen_type}")
        lines.append(f"  处理器: {c.processor}")
        lines.append(f"  商品详情: {c.parent_text[:500]}")
        lines.append("")
    return "\n".join(lines)


def _format_constraints(constraints: StructuredConstraints) -> str:
    """将约束条件格式化为 LLM 可读的文本"""
    parts = []
    if constraints.budget_min is not None or constraints.budget_max is not None:
        budget = ""
        if constraints.budget_min is not None and constraints.budget_max is not None:
            budget = f"{constraints.budget_min:.0f}-{constraints.budget_max:.0f} 元"
        elif constraints.budget_max is not None:
            budget = f"{constraints.budget_max:.0f} 元以内"
        else:
            budget = f"{constraints.budget_min:.0f} 元以上"
        parts.append(f"预算: {budget}")
    if constraints.scenario:
        parts.append(f"使用场景: {constraints.scenario}")
    if constraints.core_needs:
        parts.append(f"核心需求: {', '.join(constraints.core_needs)}")
    if constraints.negative_constraints:
        parts.append(f"不想要的: {', '.join(constraints.negative_constraints)}")
    if constraints.brands:
        parts.append(f"偏好品牌: {', '.join(constraints.brands)}")
    if constraints.exclude_brands:
        parts.append(f"排除品牌: {', '.join(constraints.exclude_brands)}")
    if constraints.tags:
        parts.append(f"其他标签: {', '.join(constraints.tags)}")
    return "\n".join(parts) if parts else "无特殊约束"


# ── Prompt 模板 ──────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = """你是一个专业的手机推荐顾问。
你的任务是根据用户需求和候选商品列表，生成个性化推荐。

## 输出要求

你需要输出三部分：

### 第一部分：推荐商品标识
- 用逗号分隔的你实际推荐的机型名称列表
- 名称必须与候选列表中的名称完全一致

### 第二部分：推荐文本
- 开头简要总结用户需求（1-2 句话）
- 从候选列表中选出最值得推荐的 3-5 款手机
- 每款推荐都要给出**具体理由**，理由必须绑定到用户提到的具体需求
  - 好的例子："这款手机搭载 5000 万像素主摄，拍照能力出色，完全满足您对拍照的需求"
  - 差的例子："这款手机很不错，值得购买"（太泛泛）
- 如果用户有负向约束（如不要曲面屏），说明推荐的商品如何满足了这一要求
- 最后给出总结性购买建议

### 第三部分：对比表格
- 使用 Markdown 表格格式
- 必须包含列：机型名、价格、屏幕类型、处理器、主摄像头、电池容量
- 根据用户核心需求，额外增加对应列：
  - 如果关心拍照：增加"拍照亮点"列
  - 如果关心游戏：增加"游戏性能"列
  - 如果关心续航：增加"充电规格"列
- 表格数据必须准确，来自候选商品的真实参数
- 表格中推荐的机型用 ✓ 标记

## 格式

用以下分隔符分隔三部分：

===RECOMMENDED_PRODUCTS===
（机型1, 机型2, ...）

===RECOMMENDATION===
（推荐文本）

===TABLE===
（Markdown 对比表格）
"""

GENERATOR_HUMAN_PROMPT = """## 用户原始需求
"{query}"

## 解析后的约束条件
{constraints}

## 候选商品列表（共 {n_candidates} 款）
{candidates}
{feedback_section}
请根据以上信息生成推荐。
"""


# ── Generator Agent ──────────────────────────────────────────────

class GeneratorAgent:
    """
    Generator Agent

    根据候选商品和约束条件，生成个性化推荐话术和 Markdown 对比表格。
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.3):
        """
        初始化 Generator Agent

        Args:
            model_name: 模型名称，默认使用 config 中的 LLM_MODEL
            temperature: 温度参数，0.3 允许适度创造性
        """
        self.model_name = model_name or LLM_MODEL
        self.temperature = temperature

        # 初始化 LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE,
        )

        # 构建 prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", GENERATOR_SYSTEM_PROMPT),
            ("human", GENERATOR_HUMAN_PROMPT),
        ])

        # 构建 chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(
        self,
        query: str,
        constraints: StructuredConstraints,
        candidates: List[RetrieverResult],
        feedback: Optional[str] = None,
    ) -> GeneratorOutput:
        """
        生成推荐文本和对比表格

        Args:
            query: 用户原始查询
            constraints: Planner 解析的结构化约束
            candidates: Retriever 返回的候选商品列表
            feedback: 上一轮 Critic 的审查意见

        Returns:
            GeneratorOutput: 包含推荐文本和对比表格
        """
        # 格式化输入
        constraints_text = _format_constraints(constraints)
        candidates_text = _format_candidates(candidates)

        feedback_section = ""
        if feedback:
            feedback_section = f"\n## 之前的审查意见（你上一轮的生成存在以下问题，请务必在本次生成中修正）\n{feedback}\n"

        try:
            # 调用 LLM
            raw_output = self.chain.invoke({
                "query": query,
                "constraints": constraints_text,
                "n_candidates": len(candidates),
                "candidates": candidates_text,
                "feedback_section": feedback_section,
            })

            # 解析输出
            return self._parse_output(raw_output)

        except Exception as e:
            return GeneratorOutput(
                recommendation_text=f"推荐生成失败: {str(e)}",
                comparison_table="| 机型 | 价格 |\n|------|------|\n| 暂无数据 | - |",
                recommended_product_names=[],
            )

    def _parse_output(self, raw_output: str) -> GeneratorOutput:
        """解析 LLM 输出为结构化结果"""
        import re
        
        recommendation_text = ""
        comparison_table = ""
        recommended_product_names = []
        
        # 提取 RECOMMENDED_PRODUCTS
        products_match = re.search(r'===RECOMMENDED_PRODUCTS===\n(.*?)\n(?:===|$)', raw_output, re.DOTALL)
        if products_match:
            products_str = products_match.group(1).strip()
            recommended_product_names = [p.strip() for p in products_str.split(',') if p.strip()]
            
        # 提取 RECOMMENDATION
        rec_match = re.search(r'===RECOMMENDATION===\n(.*?)\n(?:===|$)', raw_output, re.DOTALL)
        if rec_match:
            recommendation_text = rec_match.group(1).strip()
            
        # 提取 TABLE
        table_match = re.search(r'===TABLE===\n(.*)', raw_output, re.DOTALL)
        if table_match:
            comparison_table = table_match.group(1).strip()
            
        # 兼容旧格式（如果没有 PRODUCTS 分隔符）
        if not recommendation_text and not comparison_table:
            if "===RECOMMENDATION===" in raw_output and "===TABLE===" in raw_output:
                parts = raw_output.split("===TABLE===")
                recommendation_text = parts[0].replace("===RECOMMENDATION===", "").strip()
                comparison_table = parts[1].strip()
            elif "===TABLE===" in raw_output:
                parts = raw_output.split("===TABLE===")
                recommendation_text = parts[0].strip()
                comparison_table = parts[1].strip()
            else:
                # 没有分隔符时，尝试识别 Markdown 表格
                lines = raw_output.strip().split("\n")
                table_start = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("|") and "---" in line:
                        table_start = i - 1
                        break

                if table_start >= 0:
                    recommendation_text = "\n".join(lines[:table_start]).strip()
                    comparison_table = "\n".join(lines[table_start:]).strip()
                else:
                    recommendation_text = raw_output.strip()
                    comparison_table = "| 机型 | 价格 |\n|------|------|\n| 暂无对比数据 | - |"

        return GeneratorOutput(
            recommendation_text=recommendation_text,
            comparison_table=comparison_table,
            recommended_product_names=recommended_product_names,
        )


# ── 测试入口 ──────────────────────────────────────────────────

if __name__ == "__main__":
    from agents.planner import PlannerAgent
    from rag.retriever import HybridRetriever

    print("=" * 60)
    print("Generator Agent 测试")
    print("=" * 60)

    # 初始化组件
    planner = PlannerAgent()
    retriever = HybridRetriever()
    generator = GeneratorAgent()

    # 测试 query
    test_query = "预算3000，送女朋友，主要拍照好，不要曲面屏"

    # Step 1: Planner 解析
    print(f"\n📝 用户需求: {test_query}")
    constraints = planner.parse(test_query)
    print(f"解析结果: 预算 {constraints.budget_max}, 需求 {constraints.core_needs}")

    # Step 2: Retriever 检索
    search_constraints = planner.parse_to_search_constraints(test_query)
    candidates = retriever.search(test_query, constraints=search_constraints)
    print(f"检索到 {len(candidates)} 个候选商品")

    # Step 3: Generator 生成
    output = generator.generate(test_query, constraints, candidates)

    print(f"\n{'='*60}")
    print("推荐文本:")
    print("="*60)
    print(output.recommendation_text)

    print(f"\n{'='*60}")
    print("对比表格:")
    print("="*60)
    print(output.comparison_table)
