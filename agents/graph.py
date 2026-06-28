"""
LangGraph 状态图组装与条件回退

实现"生成-批判"双 Agent 工作流：
1. Planner → Retriever → Generator → Critic
2. Critic 未通过时，回退到 Retriever 重新执行
3. 最大迭代次数限制，防止无限循环
4. reflection_log 记录每轮迭代的审查结果

状态图结构：
- planner_node: 解析用户意图
- retriever_node: 检索候选商品
- generator_node: 生成推荐
- critic_node: 审查推荐质量
- presenter_node: 格式化最终输出
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Annotated
from dataclasses import dataclass, field
from operator import add

from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerAgent, StructuredConstraints
from agents.generator import GeneratorAgent, GeneratorOutput
from agents.critic import CriticAgent, CriticOutput
from rag.retriever import HybridRetriever, RetrieverResult, SearchConstraints


# ── 状态定义 ──────────────────────────────────────────────

class GraphState(BaseModel):
    """LangGraph 状态图的状态定义"""

    # 输入
    query: str = Field(description="用户原始查询")

    # Planner 输出
    constraints: Optional[StructuredConstraints] = Field(
        default=None,
        description="Planner 解析的结构化约束"
    )
    search_constraints: Optional[SearchConstraints] = Field(
        default=None,
        description="转换后的检索约束"
    )

    # Retriever 输出
    candidates: List[RetrieverResult] = Field(
        default_factory=list,
        description="检索到的候选商品列表"
    )

    # Generator 输出
    generator_output: Optional[GeneratorOutput] = Field(
        default=None,
        description="Generator 生成的推荐文本和对比表格"
    )

    # Critic 输出
    critic_output: Optional[CriticOutput] = Field(
        default=None,
        description="Critic 审查结果"
    )

    # 迭代控制
    iteration: int = Field(default=0, description="当前迭代次数")
    max_iterations: int = Field(default=3, description="最大迭代次数")

    # 反思日志
    reflection_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="每轮迭代的反思日志"
    )

    # 最终输出
    final_output: str = Field(default="", description="最终格式化输出")

    class Config:
        arbitrary_types_allowed = True


# ── 节点函数 ──────────────────────────────────────────────

def planner_node(state: GraphState) -> Dict[str, Any]:
    """
    Planner 节点：解析用户意图

    输入：query
    输出：constraints, search_constraints
    """
    print("\n" + "="*60)
    print("📝 [Planner] 解析用户意图...")
    print("="*60)

    planner = PlannerAgent()
    query = state.query

    # 解析约束
    constraints = planner.parse(query)
    search_constraints = planner.parse_to_search_constraints(query)

    print(f"   预算: {constraints.budget_min} - {constraints.budget_max}")
    print(f"   场景: {constraints.scenario}")
    print(f"   核心需求: {constraints.core_needs}")
    print(f"   负向约束: {constraints.negative_constraints}")
    print(f"   品牌偏好: {constraints.brands}")
    print(f"   语义查询: {constraints.semantic_query}")

    return {
        "constraints": constraints,
        "search_constraints": search_constraints,
    }


def retriever_node(state: GraphState) -> Dict[str, Any]:
    """
    Retriever 节点：检索候选商品

    输入：query, search_constraints
    输出：candidates
    """
    print("\n" + "="*60)
    print("🔍 [Retriever] 检索候选商品...")
    print("="*60)

    retriever = HybridRetriever()
    query = state.query
    search_constraints = state.search_constraints

    # 检索
    if search_constraints:
        candidates = retriever.search(query, constraints=search_constraints)
    else:
        candidates = retriever.search(query)

    print(f"   检索到 {len(candidates)} 个候选商品")

    return {
        "candidates": candidates,
    }


def generator_node(state: GraphState) -> Dict[str, Any]:
    """
    Generator 节点：生成推荐

    输入：query, constraints, candidates, reflection_log
    输出：generator_output
    """
    print("\n" + "="*60)
    print("✍️  [Generator] 生成推荐...")
    print("="*60)

    generator = GeneratorAgent()
    query = state.query
    constraints = state.constraints
    candidates = state.candidates

    # 获取上一轮的反思意见
    feedback = None
    if state.iteration > 0 and state.reflection_log:
        last_reflection = state.reflection_log[-1]
        if not last_reflection.get("passed"):
            feedback = last_reflection.get("revision_notes")
            print(f"   [接收到反思意见] {feedback[:100]}...")

    # 生成推荐
    generator_output = generator.generate(query, constraints, candidates, feedback=feedback)

    print(f"   推荐文本长度: {len(generator_output.recommendation_text)} 字符")
    print(f"   对比表格长度: {len(generator_output.comparison_table)} 字符")
    if hasattr(generator_output, 'recommended_product_names') and generator_output.recommended_product_names:
        print(f"   推荐商品数: {len(generator_output.recommended_product_names)}")

    return {
        "generator_output": generator_output,
    }


def critic_node(state: GraphState) -> Dict[str, Any]:
    """
    Critic 节点：审查推荐质量

    输入：query, constraints, generator_output, candidates
    输出：critic_output, reflection_log (追加)
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

    # 审查
    critic_output = critic.review(query, constraints, generator_output, candidates)

    # 记录反思日志
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

    # 追加到反思日志
    new_reflection_log = state.reflection_log + [reflection_entry]

    print(f"   审查结果: {'✅ 通过' if critic_output.passed else '❌ 未通过'}")
    print(f"   质量评分: {critic_output.score}/10")

    if not critic_output.passed:
        print(f"   修改意见: {critic_output.revision_notes[:100]}...")

    return {
        "critic_output": critic_output,
        "reflection_log": new_reflection_log,
        "iteration": iteration + 1,
    }


def presenter_node(state: GraphState) -> Dict[str, Any]:
    """
    Presenter 节点：格式化最终输出

    输入：query, constraints, generator_output, critic_output, reflection_log
    输出：final_output
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

    # 构建最终输出
    output_parts = []

    # 1. 用户需求摘要
    output_parts.append("## 📱 手机推荐结果\n")
    output_parts.append(f"**用户需求**: {query}\n")

    # 2. 约束条件
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

    # 3. 推荐内容
    if generator_output:
        output_parts.append("### 🎯 个性化推荐\n")
        output_parts.append(generator_output.recommendation_text)
        output_parts.append("")

        output_parts.append("### 📊 对比表格\n")
        output_parts.append(generator_output.comparison_table)
        output_parts.append("")

    # 4. 审查结果
    if critic_output:
        output_parts.append("### ✅ 质量审查\n")
        status = "通过" if critic_output.passed else "未通过"
        output_parts.append(f"- 审查结果: {status}")
        output_parts.append(f"- 质量评分: {critic_output.score}/10")
        output_parts.append("")

        # 检查详情
        output_parts.append("**检查详情**:\n")
        for check in critic_output.checks:
            icon = "✅" if check.passed else "❌"
            output_parts.append(f"- {icon} {check.name}: {check.details}")
        output_parts.append("")

    # 5. 反思日志
    if len(reflection_log) > 1:
        output_parts.append("### 🔄 反思迭代日志\n")
        output_parts.append(f"共经历 {iteration} 轮迭代:\n")

        for entry in reflection_log:
            status = "✅ 通过" if entry["passed"] else "❌ 未通过"
            output_parts.append(f"**第 {entry['iteration']} 轮**: {status} (评分: {entry['score']}/10)")

            if not entry["passed"] and entry.get("revision_notes"):
                output_parts.append(f"  修改意见: {entry['revision_notes'][:200]}...")
            output_parts.append("")

    final_output = "\n".join(output_parts)

    print(f"   输出长度: {len(final_output)} 字符")

    return {
        "final_output": final_output,
    }


# ── 条件边函数 ──────────────────────────────────────────────

def should_continue(state: GraphState) -> str:
    """
    条件边：决定 Critic 之后的流向

    逻辑：
    - 如果 Critic 通过 → presenter_node
    - 如果 Critic 未通过 且 iteration < max_iterations → retriever_node (回退)
    - 如果 Critic 未通过 且 iteration >= max_iterations → presenter_node (强制结束)
    """
    critic_output = state.critic_output
    iteration = state.iteration
    max_iterations = state.max_iterations

    if critic_output.passed:
        print(f"\n✅ Critic 审查通过，进入 Presenter")
        return "presenter"

    if iteration >= max_iterations:
        print(f"\n⚠️  已达最大迭代次数 ({max_iterations})，强制进入 Presenter")
        return "presenter"

    print(f"\n🔄 Critic 审查未通过，回退到 Retriever 重新执行 (迭代 {iteration}/{max_iterations})")
    return "retriever"


# ── 状态图构建 ──────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    构建 LangGraph 状态图

    节点：planner → retriever → generator → critic → presenter
    条件边：critic → (retriever | presenter)
    """
    # 创建状态图
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("presenter", presenter_node)

    # 设置入口
    workflow.set_entry_point("planner")

    # 添加边
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "critic")

    # 添加条件边：critic 之后的分支
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "retriever": "retriever",
            "presenter": "presenter",
        },
    )

    # presenter 到 END
    workflow.add_edge("presenter", END)

    # 编译图
    graph = workflow.compile()

    return graph


# ── 便捷函数 ──────────────────────────────────────────────

def run_graph(query: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    运行状态图的便捷函数

    Args:
        query: 用户查询
        max_iterations: 最大迭代次数

    Returns:
        包含 final_output 和 reflection_log 的字典
    """
    # 构建图
    graph = build_graph()

    # 初始状态
    initial_state = GraphState(
        query=query,
        max_iterations=max_iterations,
    )

    # 运行图
    print("\n" + "="*60)
    print("🚀 开始运行 LangGraph 状态图")
    print("="*60)
    print(f"用户查询: {query}")
    print(f"最大迭代: {max_iterations}")

    # 执行图
    final_state = graph.invoke(initial_state)

    print("\n" + "="*60)
    print("✅ 状态图执行完成")
    print("="*60)

    return {
        "final_output": final_state["final_output"],
        "reflection_log": final_state["reflection_log"],
        "iteration": final_state["iteration"],
    }


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    # 测试状态图
    test_query = "预算3000，送女朋友，主要拍照好，不要曲面屏"

    result = run_graph(test_query, max_iterations=2)

    print("\n" + "="*60)
    print("📊 最终输出")
    print("="*60)
    print(result["final_output"])

    print("\n" + "="*60)
    print("📝 反思日志")
    print("="*60)
    for entry in result["reflection_log"]:
        print(f"第 {entry['iteration']} 轮: {'通过' if entry['passed'] else '未通过'} (评分: {entry['score']}/10)")
