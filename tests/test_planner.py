"""
Planner Agent 测试套件

验证 Planner 能将自然语言 query 解析为结构化约束条件。
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerAgent, StructuredConstraints


def test_basic_parsing():
    """测试基本解析功能"""
    planner = PlannerAgent()

    # 测试 case 1: 预算 + 场景 + 核心需求 + 负向约束
    query1 = "预算3000，送女朋友，主要拍照好，不要曲面屏"
    result1 = planner.parse(query1)

    print(f"\n{'='*60}")
    print(f"测试 1: {query1}")
    print(f"{'='*60}")
    print(f"解析结果:")
    print(f"  budget_max: {result1.budget_max}")
    print(f"  scenario: {result1.scenario}")
    print(f"  core_needs: {result1.core_needs}")
    print(f"  negative_constraints: {result1.negative_constraints}")
    print(f"  semantic_query: {result1.semantic_query}")

    # 验证
    assert result1.budget_max == 3000, f"预算应为 3000，实际为 {result1.budget_max}"
    assert result1.scenario is not None and "送" in result1.scenario, f"场景应包含'送'，实际为 {result1.scenario}"
    assert any("拍照" in need for need in result1.core_needs), f"核心需求应包含'拍照'，实际为 {result1.core_needs}"
    assert any("曲面" in c for c in result1.negative_constraints), f"负向约束应包含'曲面'，实际为 {result1.negative_constraints}"

    print("✅ 测试 1 通过")


def test_unlimited_budget():
    """测试预算不限的情况"""
    planner = PlannerAgent()

    query2 = "打游戏用的，预算不限"
    result2 = planner.parse(query2)

    print(f"\n{'='*60}")
    print(f"测试 2: {query2}")
    print(f"{'='*60}")
    print(f"解析结果:")
    print(f"  budget_max: {result2.budget_max}")
    print(f"  core_needs: {result2.core_needs}")
    print(f"  semantic_query: {result2.semantic_query}")

    # 验证
    assert result2.budget_max is None, f"预算应为 None，实际为 {result2.budget_max}"
    assert any("游戏" in need for need in result2.core_needs), f"核心需求应包含'游戏'，实际为 {result2.core_needs}"

    print("✅ 测试 2 通过")


def test_brand_extraction():
    """测试品牌提取"""
    planner = PlannerAgent()

    query3 = "2000-4000元小米手机"
    result3 = planner.parse(query3)

    print(f"\n{'='*60}")
    print(f"测试 3: {query3}")
    print(f"{'='*60}")
    print(f"解析结果:")
    print(f"  budget_min: {result3.budget_min}")
    print(f"  budget_max: {result3.budget_max}")
    print(f"  brands: {result3.brands}")
    print(f"  semantic_query: {result3.semantic_query}")

    # 验证
    assert result3.budget_min == 2000, f"最低预算应为 2000，实际为 {result3.budget_min}"
    assert result3.budget_max == 4000, f"最高预算应为 4000，实际为 {result3.budget_max}"
    assert any("小米" in b for b in result3.brands), f"品牌应包含'小米'，实际为 {result3.brands}"

    print("✅ 测试 3 通过")


def test_complex_scenario():
    """测试复杂场景"""
    planner = PlannerAgent()

    query4 = "给父母买个手机，2000左右，屏幕大一点，续航好"
    result4 = planner.parse(query4)

    print(f"\n{'='*60}")
    print(f"测试 4: {query4}")
    print(f"{'='*60}")
    print(f"解析结果:")
    print(f"  budget_min: {result4.budget_min}")
    print(f"  budget_max: {result4.budget_max}")
    print(f"  scenario: {result4.scenario}")
    print(f"  core_needs: {result4.core_needs}")
    print(f"  tags: {result4.tags}")

    # 验证（宽松验证，因为 LLM 解析可能有差异）
    assert result4.budget_max is not None, "应有预算上限"
    assert result4.budget_max <= 3000, f"预算上限应 <= 3000，实际为 {result4.budget_max}"
    assert any("续航" in need for need in result4.core_needs) or any("续航" in t for t in result4.tags), \
        "应包含续航需求"

    print("✅ 测试 4 通过")


def test_structured_constraints_schema():
    """测试 StructuredConstraints schema 完整性"""
    print(f"\n{'='*60}")
    print(f"测试 5: StructuredConstraints Schema 验证")
    print(f"{'='*60}")

    # 创建一个完整的约束条件
    constraints = StructuredConstraints(
        budget_min=2000,
        budget_max=5000,
        scenario="送礼",
        core_needs=["拍照", "续航"],
        negative_constraints=["曲面屏"],
        brands=["华为", "小米"],
        exclude_brands=["苹果"],
        screen_type_keywords=["AMOLED"],
        processor_keywords=["骁龙8"],
        tags=["轻薄"],
        semantic_query="拍照续航好的手机",
        reasoning="用户需要送礼，预算2000-5000，偏好华为小米，需要拍照和续航，不要曲面屏"
    )

    # 验证所有字段都可访问
    assert constraints.budget_min == 2000
    assert constraints.budget_max == 5000
    assert constraints.scenario == "送礼"
    assert "拍照" in constraints.core_needs
    assert "曲面屏" in constraints.negative_constraints
    assert "华为" in constraints.brands
    assert "苹果" in constraints.exclude_brands
    assert "AMOLED" in constraints.screen_type_keywords
    assert "骁龙8" in constraints.processor_keywords
    assert "轻薄" in constraints.tags
    assert constraints.semantic_query == "拍照续航好的手机"

    # 验证可以转换为 dict
    constraints_dict = constraints.model_dump()
    assert isinstance(constraints_dict, dict)
    assert "budget_max" in constraints_dict

    print("✅ 测试 5 通过")
    print(f"Schema 字段: {list(constraints_dict.keys())}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Planner Agent 测试套件")
    print("=" * 60)

    tests = [
        test_structured_constraints_schema,
        test_basic_parsing,
        test_unlimited_budget,
        test_brand_extraction,
        test_complex_scenario,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
