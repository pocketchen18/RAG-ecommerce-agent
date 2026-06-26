"""
Critic Agent 测试

验证 6 项结构化检查：
1. 价格检查：推荐中超预算的商品能被检出
2. 负向约束检查：推荐中出现曲面屏能被检出
3. 多样性检查：3 款同品牌推荐能被检出
4. 需求覆盖检查：推荐理由是否覆盖核心需求
5. 信息准确性检查：推荐信息是否与候选商品一致
6. 推荐完整性检查：推荐文本是否完整
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.critic import (
    CriticAgent,
    CriticOutput,
    CheckResult,
    _check_price_constraints,
    _check_negative_constraints,
    _check_diversity,
    _check_needs_coverage,
    _check_accuracy,
    _check_completeness,
)
from agents.planner import StructuredConstraints
from agents.generator import GeneratorOutput
from rag.retriever import RetrieverResult


def create_mock_candidates(brand_prices=None, screen_types=None):
    """创建模拟候选商品列表"""
    if brand_prices is None:
        brand_prices = [("小米", 2500), ("华为", 3500), ("vivo", 2800)]
    if screen_types is None:
        screen_types = ["直屏", "直屏", "直屏"]

    candidates = []
    for i, ((brand, price), screen) in enumerate(zip(brand_prices, screen_types)):
        candidates.append(RetrieverResult(
            sku_id=f"sku_{i}",
            name=f"{brand}手机{i+1}",
            brand=brand,
            price=price,
            screen_type=screen,
            processor="骁龙8 Gen3",
            section_type="product",
            parent_text=f"{brand}手机{i+1}的详细描述",
            score=0.9,
            source="test",
        ))
    return candidates


def create_mock_constraints(
    budget_max=None,
    budget_min=None,
    core_needs=None,
    negative_constraints=None,
    scenario=None,
):
    """创建模拟约束条件"""
    return StructuredConstraints(
        budget_max=budget_max,
        budget_min=budget_min,
        core_needs=core_needs or [],
        negative_constraints=negative_constraints or [],
        scenario=scenario,
    )


def create_mock_generator_output(recommendation_text=None, comparison_table=None):
    """创建模拟推荐输出"""
    if recommendation_text is None:
        recommendation_text = "这是一款拍照出色的手机，适合送给女朋友。"
    if comparison_table is None:
        comparison_table = "| 机型 | 价格 | 拍照 |\n|------|------|------|\n| 小米手机1 | 2500元 | ✓ |"
    return GeneratorOutput(
        recommendation_text=recommendation_text,
        comparison_table=comparison_table,
    )


def test_schema_validation():
    """测试 1: Schema 验证 - CriticOutput 包含所有必需字段"""
    print("测试 1: Schema 验证...")

    # 创建 CriticOutput 实例
    output = CriticOutput(
        passed=True,
        checks=[
            CheckResult(name="test_check", passed=True, details="测试通过"),
        ],
        revision_notes="无修改意见",
        score=10.0,
    )

    # 验证字段存在
    assert hasattr(output, 'passed'), "CriticOutput 缺少 passed 字段"
    assert hasattr(output, 'checks'), "CriticOutput 缺少 checks 字段"
    assert hasattr(output, 'revision_notes'), "CriticOutput 缺少 revision_notes 字段"
    assert hasattr(output, 'score'), "CriticOutput 缺少 score 字段"

    # 验证字段类型
    assert isinstance(output.passed, bool), "passed 应为 bool 类型"
    assert isinstance(output.checks, list), "checks 应为 list 类型"
    assert isinstance(output.revision_notes, str), "revision_notes 应为 str 类型"
    assert isinstance(output.score, float), "score 应为 float 类型"

    # 验证 CheckResult 字段
    check = output.checks[0]
    assert hasattr(check, 'name'), "CheckResult 缺少 name 字段"
    assert hasattr(check, 'passed'), "CheckResult 缺少 passed 字段"
    assert hasattr(check, 'details'), "CheckResult 缺少 details 字段"

    print("✅ 测试 1 通过: Schema 验证正确")


def test_price_check_over_budget():
    """测试 2: 价格检查 - 推荐中超预算的商品能被检出"""
    print("\n测试 2: 价格检查（超预算）...")

    # 创建约束：预算 3000 元
    constraints = create_mock_constraints(budget_max=3000)

    # 创建候选：包含 3500 元的商品（超预算）
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
    )

    # 执行价格检查
    result = _check_price_constraints(candidates, constraints, "推荐文本")

    assert result.name == "price_check", "检查名称应为 price_check"
    assert result.passed == False, "超预算商品应导致检查失败"
    assert "3500" in result.details, "应提到超预算的价格"
    assert "3000" in result.details, "应提到预算上限"

    print("✅ 测试 2 通过: 超预算商品被正确检出")


def test_price_check_within_budget():
    """测试 2b: 价格检查 - 所有商品在预算内"""
    print("\n测试 2b: 价格检查（预算内）...")

    constraints = create_mock_constraints(budget_max=4000)
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
    )

    result = _check_price_constraints(candidates, constraints, "推荐文本")

    assert result.passed == True, "所有商品在预算内应通过检查"

    print("✅ 测试 2b 通过: 预算内商品正确通过检查")


def test_negative_constraint_check():
    """测试 3: 负向约束检查 - 推荐中出现曲面屏能被检出"""
    print("\n测试 3: 负向约束检查...")

    # 创建约束：不要曲面屏
    constraints = create_mock_constraints(negative_constraints=["曲面屏"])

    # 创建候选：包含曲面屏商品
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
        screen_types=["直屏", "曲面屏", "直屏"],
    )

    # 执行负向约束检查
    result = _check_negative_constraints(candidates, constraints)

    assert result.name == "negative_constraint_check", "检查名称应为 negative_constraint_check"
    assert result.passed == False, "曲面屏商品应导致检查失败"
    assert "曲面屏" in result.details, "应提到违反的约束"

    print("✅ 测试 3 通过: 曲面屏商品被正确检出")


def test_negative_constraint_no_violation():
    """测试 3b: 负向约束检查 - 无违反"""
    print("\n测试 3b: 负向约束检查（无违反）...")

    constraints = create_mock_constraints(negative_constraints=["曲面屏"])
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
        screen_types=["直屏", "直屏", "直屏"],
    )

    result = _check_negative_constraints(candidates, constraints)

    assert result.passed == True, "无曲面屏商品应通过检查"

    print("✅ 测试 3b 通过: 无违反时正确通过检查")


def test_diversity_check_same_brand():
    """测试 4: 多样性检查 - 4 款同品牌推荐能被检出（超过阈值 3）"""
    print("\n测试 4: 多样性检查（同品牌过多）...")

    # 创建候选：4 款小米手机（超过默认阈值 3）
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("小米", 3000), ("小米", 3500), ("小米", 4000)],
        screen_types=["直屏", "直屏", "直屏", "直屏"],
    )

    # 执行多样性检查（默认阈值 3）
    result = _check_diversity(candidates, max_same_brand=3)

    assert result.name == "diversity_check", "检查名称应为 diversity_check"
    assert result.passed == False, "4 款同品牌应导致检查失败"
    assert "小米" in result.details, "应提到过多的品牌"
    assert "4" in result.details, "应提到实际数量"

    print("✅ 测试 4 通过: 同品牌过多被正确检出")


def test_diversity_check_good():
    """测试 4b: 多样性检查 - 品牌分布合理"""
    print("\n测试 4b: 多样性检查（品牌分布合理）...")

    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
    )

    result = _check_diversity(candidates, max_same_brand=3)

    assert result.passed == True, "品牌分布合理应通过检查"
    assert "小米" in result.details, "应包含品牌分布信息"
    assert "华为" in result.details, "应包含品牌分布信息"
    assert "vivo" in result.details, "应包含品牌分布信息"

    print("✅ 测试 4b 通过: 品牌分布合理时正确通过检查")


def test_needs_coverage_check():
    """测试 5: 需求覆盖检查 - 推荐理由是否覆盖核心需求"""
    print("\n测试 5: 需求覆盖检查...")

    # 创建约束：核心需求为拍照
    constraints = create_mock_constraints(core_needs=["拍照"])

    # 推荐文本未提到"拍照"
    recommendation_text = "这是一款性能出色的手机，适合日常使用。"

    result = _check_needs_coverage(constraints, recommendation_text)

    assert result.name == "needs_coverage_check", "检查名称应为 needs_coverage_check"
    assert result.passed == False, "未覆盖核心需求应导致检查失败"
    assert "拍照" in result.details, "应提到未覆盖的需求"

    print("✅ 测试 5 通过: 未覆盖核心需求被正确检出")


def test_needs_coverage_check_pass():
    """测试 5b: 需求覆盖检查 - 已覆盖"""
    print("\n测试 5b: 需求覆盖检查（已覆盖）...")

    constraints = create_mock_constraints(core_needs=["拍照"])
    recommendation_text = "这款手机拍照能力出色，完全满足您对拍照的需求。"

    result = _check_needs_coverage(constraints, recommendation_text)

    assert result.passed == True, "已覆盖核心需求应通过检查"

    print("✅ 测试 5b 通过: 已覆盖核心需求时正确通过检查")


def test_accuracy_check():
    """测试 6: 信息准确性检查"""
    print("\n测试 6: 信息准确性检查...")

    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500)],
    )

    # 对比表格包含机型名
    comparison_table = "| 机型 | 价格 |\n|------|------|\n| 小米手机1 | 2500元 |\n| 华为手机2 | 3500元 |"

    result = _check_accuracy(candidates, comparison_table)

    assert result.name == "accuracy_check", "检查名称应为 accuracy_check"
    assert result.passed == True, "有效表格应通过检查"
    assert "2" in result.details, "应包含机型数量"

    print("✅ 测试 6 通过: 信息准确性检查正确")


def test_completeness_check():
    """测试 7: 推荐完整性检查"""
    print("\n测试 7: 推荐完整性检查...")

    # 推荐文本过短
    recommendation_text = "推荐这款手机。"
    comparison_table = "| 机型 | 价格 |\n|------|------|\n| 暂无数据 | - |"

    result = _check_completeness(recommendation_text, comparison_table)

    assert result.name == "completeness_check", "检查名称应为 completeness_check"
    assert result.passed == False, "过短文本和暂无数据应导致检查失败"

    print("✅ 测试 7 通过: 推荐完整性检查正确")


def test_critic_pass_scenario():
    """测试 8: Critic 通过场景 - passed=True 且 checks 全部 passed"""
    print("\n测试 8: Critic 通过场景...")

    # 创建符合条件的数据
    constraints = create_mock_constraints(
        budget_max=4000,
        core_needs=["拍照"],
        negative_constraints=[],
    )

    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800)],
        screen_types=["直屏", "直屏", "直屏"],
    )

    # 使用足够长的推荐文本（超过 100 字符）
    recommendation_text = (
        "根据您对拍照功能的需求，我为您精心挑选了以下几款手机。"
        "这些手机都搭载了高像素主摄像头和先进的图像处理算法，"
        "无论是白天还是夜晚，都能拍出清晰细腻的照片。"
        "同时，它们的价格都在您的预算范围内，性价比非常高。"
        "特别推荐小米手机1，它的一亿像素主摄在这个价位段表现出色。"
    )

    generator_output = create_mock_generator_output(
        recommendation_text=recommendation_text,
        comparison_table="| 机型 | 价格 | 拍照 |\n|------|------|------|\n| 小米手机1 | 2500元 | ✓ |\n| 华为手机2 | 3500元 | ✓ |\n| vivo手机3 | 2800元 | ✓ |",
    )

    # 创建 Critic Agent（不使用 LLM）
    critic = CriticAgent(use_llm=False)

    # 执行审查
    output = critic.review("拍照好的手机", constraints, generator_output, candidates)

    # 验证整体通过
    assert output.passed == True, "符合条件的推荐应通过审查"
    assert output.score == 10.0, f"分数应为 10.0，实际为 {output.score}"
    assert output.revision_notes == "所有检查通过，推荐质量良好。", "通过时应有正面反馈"

    # 验证所有检查通过
    for check in output.checks:
        assert check.passed == True, f"检查 {check.name} 应通过"

    print("✅ 测试 8 通过: 符合条件的推荐正确通过审查")


def test_critic_fail_scenario():
    """测试 9: Critic 未通过场景 - revision_notes 非空且包含具体修改意见"""
    print("\n测试 9: Critic 未通过场景...")

    # 创建不符合条件的数据
    constraints = create_mock_constraints(
        budget_max=3000,
        core_needs=["拍照"],
        negative_constraints=["曲面屏"],
    )

    # 候选商品：包含超预算和曲面屏
    candidates = create_mock_candidates(
        brand_prices=[("小米", 2500), ("华为", 3500), ("vivo", 2800), ("小米", 2600)],
        screen_types=["直屏", "曲面屏", "直屏", "直屏"],
    )

    # 推荐文本未提到"拍照"
    generator_output = create_mock_generator_output(
        recommendation_text="这是一款性能出色的手机，适合日常使用。",
        comparison_table="| 机型 | 价格 |\n|------|------|\n| 小米手机1 | 2500元 |",
    )

    critic = CriticAgent(use_llm=False)
    output = critic.review("拍照好的手机", constraints, generator_output, candidates)

    # 验证整体未通过
    assert output.passed == False, "不符合条件的推荐应未通过审查"
    assert output.score < 10.0, "分数应低于 10.0"

    # 验证修改意见非空且包含具体内容
    assert len(output.revision_notes) > 0, "修改意见不应为空"
    assert "价格" in output.revision_notes or "price" in output.revision_notes.lower(), "应包含价格相关意见"
    assert "曲面屏" in output.revision_notes or "negative" in output.revision_notes.lower(), "应包含负向约束相关意见"
    assert "拍照" in output.revision_notes or "needs" in output.revision_notes.lower(), "应包含需求覆盖相关意见"

    # 验证有失败的检查
    failed_checks = [c for c in output.checks if not c.passed]
    assert len(failed_checks) > 0, "应有失败的检查项"

    print("✅ 测试 9 通过: 不符合条件的推荐正确未通过审查")
    print(f"   失败检查: {[c.name for c in failed_checks]}")
    print(f"   修改意见:\n{output.revision_notes[:200]}...")


def test_critic_agent_initialization():
    """测试 10: CriticAgent 初始化"""
    print("\n测试 10: CriticAgent 初始化...")

    # 测试默认初始化
    critic = CriticAgent()
    assert critic.max_same_brand == 3, "默认同品牌阈值应为 3"
    assert critic.use_llm == False, "默认不使用 LLM"

    # 测试自定义初始化
    critic_custom = CriticAgent(max_same_brand=2, use_llm=False)
    assert critic_custom.max_same_brand == 2, "自定义同品牌阈值应为 2"

    print("✅ 测试 10 通过: CriticAgent 初始化正确")


# ── 运行所有测试 ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Critic Agent 测试套件")
    print("=" * 60)

    tests = [
        test_schema_validation,
        test_price_check_over_budget,
        test_price_check_within_budget,
        test_negative_constraint_check,
        test_negative_constraint_no_violation,
        test_diversity_check_same_brand,
        test_diversity_check_good,
        test_needs_coverage_check,
        test_needs_coverage_check_pass,
        test_accuracy_check,
        test_completeness_check,
        test_critic_pass_scenario,
        test_critic_fail_scenario,
        test_critic_agent_initialization,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ 测试失败: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ 测试错误: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)
