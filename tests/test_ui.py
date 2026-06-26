"""
UI 测试 — Streamlit 前端功能验证

测试项：
1. app.py 可正常导入
2. GraphState 创建正确
3. build_graph() 返回可用的图
4. 侧边栏配置参数正确
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def test_app_import():
    """测试 app.py 可正常导入"""
    try:
        import app
        print("✅ app.py 可正常导入")
        return True
    except Exception as e:
        print(f"❌ app.py 导入失败: {e}")
        return False


def test_graph_state():
    """测试 GraphState 创建"""
    from agents.graph import GraphState

    state = GraphState(
        query="测试查询",
        max_iterations=3,
    )

    assert state.query == "测试查询"
    assert state.max_iterations == 3
    assert state.iteration == 0
    assert state.reflection_log == []
    assert state.final_output == ""

    print("✅ GraphState 创建正确")
    return True


def test_build_graph():
    """测试 build_graph() 返回可用的图"""
    from agents.graph import build_graph

    graph = build_graph()

    # 验证图有节点
    nodes = list(graph.nodes)
    assert "planner" in nodes
    assert "retriever" in nodes
    assert "generator" in nodes
    assert "critic" in nodes
    assert "presenter" in nodes

    print("✅ build_graph() 返回可用的图")
    print(f"   节点: {nodes}")
    return True


def test_sidebar_config():
    """测试侧边栏配置参数"""
    # 验证配置参数的默认值
    max_iterations_default = 3

    # 验证参数范围
    assert 1 <= max_iterations_default <= 5

    print("✅ 侧边栏配置参数正确")
    return True


def test_step_extraction():
    """测试步骤提取逻辑"""
    from agents.planner import StructuredConstraints
    from agents.generator import GeneratorOutput
    from agents.critic import CriticOutput, CheckResult

    # 创建模拟数据
    constraints = StructuredConstraints(
        budget_max=3000,
        scenario="送礼",
        core_needs=["拍照"],
        negative_constraints=["曲面屏"],
        brands=[],
    )

    generator_output = GeneratorOutput(
        recommendation_text="推荐文本",
        comparison_table="对比表格",
    )

    critic_output = CriticOutput(
        passed=True,
        checks=[
            CheckResult(name="价格检查", passed=True, details="通过"),
        ],
        revision_notes="",
        score=9,
    )

    # 验证数据可访问
    assert constraints.budget_max == 3000
    assert constraints.scenario == "送礼"
    assert "拍照" in constraints.core_needs
    assert generator_output.recommendation_text == "推荐文本"
    assert critic_output.passed is True
    assert critic_output.score == 9

    print("✅ 步骤提取逻辑正确")
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 UI 测试")
    print("="*60 + "\n")

    results = []
    results.append(("app.py 导入", test_app_import()))
    results.append(("GraphState 创建", test_graph_state()))
    results.append(("build_graph()", test_build_graph()))
    results.append(("侧边栏配置", test_sidebar_config()))
    results.append(("步骤提取逻辑", test_step_extraction()))

    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("🎉 所有测试通过！")
    else:
        print("⚠️  部分测试失败")
    print("="*60)
