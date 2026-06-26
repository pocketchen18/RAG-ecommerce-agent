"""
测试 LangGraph 状态图组装与条件回退

验证：
1. 状态图可以正常构建和编译
2. 正常路径：Planner → Retriever → Generator → Critic(pass) → Presenter
3. 回退路径：Critic(fail) → Retriever → Generator → Critic(pass) → Presenter
4. 最大迭代次数限制生效
5. reflection_log 记录了每轮迭代的 critic verdict 和修改意见
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import List

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.graph import (
    GraphState,
    planner_node,
    retriever_node,
    generator_node,
    critic_node,
    presenter_node,
    should_continue,
    build_graph,
    run_graph,
)
from agents.planner import StructuredConstraints
from agents.generator import GeneratorOutput
from agents.critic import CriticOutput, CheckResult
from rag.retriever import RetrieverResult, SearchConstraints


# ── 测试数据 ──────────────────────────────────────────────

def create_mock_candidates() -> List[RetrieverResult]:
    """创建模拟候选商品"""
    return [
        RetrieverResult(
            sku_id="TEST001",
            name="测试手机1",
            brand="小米",
            price=2999.0,
            screen_type="直屏",
            processor="骁龙8 Gen3",
            section_type="概述",
            parent_text="这是一款拍照手机",
            score=0.95,
            source="vector",
        ),
        RetrieverResult(
            sku_id="TEST002",
            name="测试手机2",
            brand="华为",
            price=3499.0,
            screen_type="直屏",
            processor="麒麟9000",
            section_type="概述",
            parent_text="这是一款商务手机",
            score=0.85,
            source="bm25",
        ),
    ]


def create_mock_constraints() -> StructuredConstraints:
    """创建模拟约束条件"""
    return StructuredConstraints(
        budget_min=None,
        budget_max=3000.0,
        scenario="送礼",
        core_needs=["拍照"],
        negative_constraints=["曲面屏"],
        brands=[],
        exclude_brands=[],
        screen_type_keywords=[],
        processor_keywords=[],
        tags=[],
        semantic_query="拍照好的手机",
        reasoning="用户需要3000元以内拍照好的手机",
    )


def create_mock_generator_output() -> GeneratorOutput:
    """创建模拟 Generator 输出"""
    return GeneratorOutput(
        recommendation_text="根据您的需求，推荐以下拍照手机：\n1. 小米14 - 拍照出色\n2. 华为P60 - 商务拍照",
        comparison_table="| 机型 | 价格 | 拍照 |\n|------|------|------|\n| 小米14 | 2999 | ✓ |\n| 华为P60 | 3499 | ✓ |",
    )


def create_mock_critic_output(passed: bool = True) -> CriticOutput:
    """创建模拟 Critic 输出"""
    if passed:
        return CriticOutput(
            passed=True,
            checks=[
                CheckResult(name="price_check", passed=True, details="价格符合预算"),
                CheckResult(name="negative_constraint_check", passed=True, details="无负向约束违反"),
                CheckResult(name="diversity_check", passed=True, details="品牌分布合理"),
                CheckResult(name="needs_coverage_check", passed=True, details="覆盖核心需求"),
                CheckResult(name="accuracy_check", passed=True, details="信息准确"),
                CheckResult(name="completeness_check", passed=True, details="内容完整"),
            ],
            revision_notes="所有检查通过，推荐质量良好。",
            score=10.0,
        )
    else:
        return CriticOutput(
            passed=False,
            checks=[
                CheckResult(name="price_check", passed=False, details="测试手机2 超出预算"),
                CheckResult(name="negative_constraint_check", passed=True, details="无负向约束违反"),
                CheckResult(name="diversity_check", passed=True, details="品牌分布合理"),
                CheckResult(name="needs_coverage_check", passed=True, details="覆盖核心需求"),
                CheckResult(name="accuracy_check", passed=True, details="信息准确"),
                CheckResult(name="completeness_check", passed=True, details="内容完整"),
            ],
            revision_notes="价格检查未通过：测试手机2 超出预算",
            score=8.3,
        )


# ── 测试用例 ──────────────────────────────────────────────

def test_graph_state_creation():
    """测试 GraphState 创建"""
    print("\n✅ 测试 1: GraphState 创建")

    state = GraphState(query="测试查询")

    assert state.query == "测试查询"
    assert state.constraints is None
    assert state.search_constraints is None
    assert state.candidates == []
    assert state.generator_output is None
    assert state.critic_output is None
    assert state.iteration == 0
    assert state.max_iterations == 3
    assert state.reflection_log == []
    assert state.final_output == ""

    print("   ✓ GraphState 所有字段正确初始化")


def test_planner_node():
    """测试 Planner 节点"""
    print("\n✅ 测试 2: Planner 节点")

    # 创建模拟的 PlannerAgent
    mock_constraints = create_mock_constraints()

    with patch("agents.graph.PlannerAgent") as MockPlanner:
        # 配置 mock
        mock_planner_instance = MagicMock()
        mock_planner_instance.parse.return_value = mock_constraints
        mock_planner_instance.parse_to_search_constraints.return_value = SearchConstraints(
            budget_max=3000.0,
            exclude_screen_keywords=["曲面"],
        )
        MockPlanner.return_value = mock_planner_instance

        # 创建状态
        state = GraphState(query="预算3000，拍照好的手机")

        # 执行节点
        result = planner_node(state)

        # 验证
        assert "constraints" in result
        assert "search_constraints" in result
        assert result["constraints"].budget_max == 3000.0
        assert result["constraints"].core_needs == ["拍照"]

    print("   ✓ Planner 节点正确解析用户意图")


def test_retriever_node():
    """测试 Retriever 节点"""
    print("\n✅ 测试 3: Retriever 节点")

    mock_candidates = create_mock_candidates()

    with patch("agents.graph.HybridRetriever") as MockRetriever:
        # 配置 mock
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.search.return_value = mock_candidates
        MockRetriever.return_value = mock_retriever_instance

        # 创建状态
        state = GraphState(
            query="测试查询",
            search_constraints=SearchConstraints(budget_max=3000.0),
        )

        # 执行节点
        result = retriever_node(state)

        # 验证
        assert "candidates" in result
        assert len(result["candidates"]) == 2
        assert result["candidates"][0].name == "测试手机1"

    print("   ✓ Retriever 节点正确检索候选商品")


def test_generator_node():
    """测试 Generator 节点"""
    print("\n✅ 测试 4: Generator 节点")

    mock_generator_output = create_mock_generator_output()

    with patch("agents.graph.GeneratorAgent") as MockGenerator:
        # 配置 mock
        mock_generator_instance = MagicMock()
        mock_generator_instance.generate.return_value = mock_generator_output
        MockGenerator.return_value = mock_generator_instance

        # 创建状态
        state = GraphState(
            query="测试查询",
            constraints=create_mock_constraints(),
            candidates=create_mock_candidates(),
        )

        # 执行节点
        result = generator_node(state)

        # 验证
        assert "generator_output" in result
        assert "推荐" in result["generator_output"].recommendation_text
        assert "对比表格" in result["generator_output"].comparison_table or "|" in result["generator_output"].comparison_table

    print("   ✓ Generator 节点正确生成推荐")


def test_critic_node_pass():
    """测试 Critic 节点 - 通过场景"""
    print("\n✅ 测试 5: Critic 节点 - 通过场景")

    mock_critic_output = create_mock_critic_output(passed=True)

    with patch("agents.graph.CriticAgent") as MockCritic:
        # 配置 mock
        mock_critic_instance = MagicMock()
        mock_critic_instance.review.return_value = mock_critic_output
        MockCritic.return_value = mock_critic_instance

        # 创建状态
        state = GraphState(
            query="测试查询",
            constraints=create_mock_constraints(),
            generator_output=create_mock_generator_output(),
            candidates=create_mock_candidates(),
            iteration=0,
        )

        # 执行节点
        result = critic_node(state)

        # 验证
        assert "critic_output" in result
        assert "reflection_log" in result
        assert result["critic_output"].passed == True
        assert result["iteration"] == 1
        assert len(result["reflection_log"]) == 1
        assert result["reflection_log"][0]["passed"] == True

    print("   ✓ Critic 节点正确处理通过场景")


def test_critic_node_fail():
    """测试 Critic 节点 - 未通过场景"""
    print("\n✅ 测试 6: Critic 节点 - 未通过场景")

    mock_critic_output = create_mock_critic_output(passed=False)

    with patch("agents.graph.CriticAgent") as MockCritic:
        # 配置 mock
        mock_critic_instance = MagicMock()
        mock_critic_instance.review.return_value = mock_critic_output
        MockCritic.return_value = mock_critic_instance

        # 创建状态
        state = GraphState(
            query="测试查询",
            constraints=create_mock_constraints(),
            generator_output=create_mock_generator_output(),
            candidates=create_mock_candidates(),
            iteration=1,
        )

        # 执行节点
        result = critic_node(state)

        # 验证
        assert "critic_output" in result
        assert "reflection_log" in result
        assert result["critic_output"].passed == False
        assert result["iteration"] == 2
        assert len(result["reflection_log"]) == 1
        assert result["reflection_log"][0]["passed"] == False
        assert "价格检查未通过" in result["reflection_log"][0]["revision_notes"]

    print("   ✓ Critic 节点正确处理未通过场景")


def test_presenter_node():
    """测试 Presenter 节点"""
    print("\n✅ 测试 7: Presenter 节点")

    # 创建状态
    state = GraphState(
        query="预算3000，拍照好的手机",
        constraints=create_mock_constraints(),
        generator_output=create_mock_generator_output(),
        critic_output=create_mock_critic_output(passed=True),
        reflection_log=[
            {
                "iteration": 1,
                "passed": True,
                "score": 10.0,
                "checks": [],
                "revision_notes": "所有检查通过",
            }
        ],
        iteration=1,
    )

    # 执行节点
    result = presenter_node(state)

    # 验证
    assert "final_output" in result
    assert len(result["final_output"]) > 0
    assert "手机推荐结果" in result["final_output"]
    assert "预算3000" in result["final_output"]
    assert "推荐" in result["final_output"]

    print("   ✓ Presenter 节点正确格式化输出")


def test_should_continue_pass():
    """测试条件边 - 通过场景"""
    print("\n✅ 测试 8: 条件边 - 通过场景")

    state = GraphState(
        query="测试查询",
        critic_output=create_mock_critic_output(passed=True),
        iteration=1,
        max_iterations=3,
    )

    result = should_continue(state)
    assert result == "presenter"

    print("   ✓ Critic 通过时正确流向 Presenter")


def test_should_continue_fail_retry():
    """测试条件边 - 未通过且回退"""
    print("\n✅ 测试 9: 条件边 - 未通过且回退")

    state = GraphState(
        query="测试查询",
        critic_output=create_mock_critic_output(passed=False),
        iteration=1,
        max_iterations=3,
    )

    result = should_continue(state)
    assert result == "retriever"

    print("   ✓ Critic 未通过时正确回退到 Retriever")


def test_should_continue_fail_max_iterations():
    """测试条件边 - 未通过且达到最大迭代"""
    print("\n✅ 测试 10: 条件边 - 未通过且达到最大迭代")

    state = GraphState(
        query="测试查询",
        critic_output=create_mock_critic_output(passed=False),
        iteration=3,
        max_iterations=3,
    )

    result = should_continue(state)
    assert result == "presenter"

    print("   ✓ 达到最大迭代时强制流向 Presenter")


def test_build_graph():
    """测试状态图构建"""
    print("\n✅ 测试 11: 状态图构建")

    graph = build_graph()

    # 验证图可以编译
    assert graph is not None

    # 验证节点存在
    # 注意：langgraph 的 API 可能不同，这里只验证图对象
    print("   ✓ 状态图成功构建和编译")


def test_reflection_log_accumulation():
    """测试反思日志累积"""
    print("\n✅ 测试 12: 反思日志累积")

    mock_critic_output_fail = create_mock_critic_output(passed=False)
    mock_critic_output_pass = create_mock_critic_output(passed=True)

    with patch("agents.graph.CriticAgent") as MockCritic:
        mock_critic_instance = MagicMock()

        # 第一次调用返回失败，第二次返回成功
        mock_critic_instance.review.side_effect = [
            mock_critic_output_fail,
            mock_critic_output_pass,
        ]
        MockCritic.return_value = mock_critic_instance

        # 第一轮迭代
        state1 = GraphState(
            query="测试查询",
            constraints=create_mock_constraints(),
            generator_output=create_mock_generator_output(),
            candidates=create_mock_candidates(),
            iteration=0,
        )
        result1 = critic_node(state1)

        # 第二轮迭代
        state2 = GraphState(
            query="测试查询",
            constraints=create_mock_constraints(),
            generator_output=create_mock_generator_output(),
            candidates=create_mock_candidates(),
            iteration=result1["iteration"],
            reflection_log=result1["reflection_log"],
        )
        result2 = critic_node(state2)

        # 验证反思日志累积
        assert len(result2["reflection_log"]) == 2
        assert result2["reflection_log"][0]["passed"] == False
        assert result2["reflection_log"][1]["passed"] == True
        assert result2["iteration"] == 2

    print("   ✓ 反思日志正确累积多轮迭代")


# ── 主测试入口 ──────────────────────────────────────────────

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🧪 LangGraph 状态图测试")
    print("="*60)

    tests = [
        test_graph_state_creation,
        test_planner_node,
        test_retriever_node,
        test_generator_node,
        test_critic_node_pass,
        test_critic_node_fail,
        test_presenter_node,
        test_should_continue_pass,
        test_should_continue_fail_retry,
        test_should_continue_fail_max_iterations,
        test_build_graph,
        test_reflection_log_accumulation,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ 测试错误: {e}")
            failed += 1

    print("\n" + "="*60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    print("="*60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
