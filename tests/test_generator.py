"""
Generator Agent 测试套件

验证：
1. Generator 输出包含推荐文本和对比表格
2. 推荐文本中提到了用户的核心需求
3. 对比表格包含机型名、价格、关键参数列
4. 推荐理由绑定到用户具体需求
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.planner import PlannerAgent, StructuredConstraints
from agents.generator import GeneratorAgent, GeneratorOutput, _format_candidates, _format_constraints
from rag.retriever import HybridRetriever, RetrieverResult


def make_mock_candidates():
    """创建模拟候选商品列表（避免依赖真实检索）"""
    return [
        RetrieverResult(
            sku_id="TEST-001",
            name="OPPO Reno 21",
            brand="OPPO",
            price=2999,
            screen_type="直屏",
            processor="天玑9200",
            section_type="概述",
            parent_text="OPPO Reno 21 搭载 5000 万像素主摄，支持 OIS 光学防抖，拍照表现出色。6.7 英寸 AMOLED 直屏，120Hz 刷新率。4500mAh 电池，67W 快充。",
            score=0.015,
            source="both",
        ),
        RetrieverResult(
            sku_id="TEST-002",
            name="小米 POCO 62",
            brand="小米",
            price=1299,
            screen_type="直屏",
            processor="骁龙7 Gen3",
            section_type="概述",
            parent_text="小米 POCO 62 性价比之选，4800 万像素主摄，6.67 英寸 LCD 直屏，5000mAh 大电池，33W 快充。",
            score=0.012,
            source="vector",
        ),
        RetrieverResult(
            sku_id="TEST-003",
            name="荣耀 Magic 47 Pro",
            brand="荣耀",
            price=1800,
            screen_type="直屏",
            processor="天玑8300",
            section_type="概述",
            parent_text="荣耀 Magic 47 Pro 配备 1.6 亿像素主摄，支持 AI 影像增强。6.78 英寸 OLED 直屏，高频 PWM 调光护眼。4800mAh 电池，100W 超级快充。",
            score=0.011,
            source="bm25",
        ),
        RetrieverResult(
            sku_id="TEST-004",
            name="三星 Galaxy A 56",
            brand="三星",
            price=2699,
            screen_type="直屏",
            processor="Exynos 1480",
            section_type="概述",
            parent_text="三星 Galaxy A 56 搭载 5000 万像素 OIS 主摄，夜景模式表现出色。6.7 英寸 Super AMOLED 直屏，Vision Booster 技术。5000mAh 电池，25W 快充。",
            score=0.010,
            source="vector",
        ),
        RetrieverResult(
            sku_id="TEST-005",
            name="vivo iQOO 46",
            brand="vivo",
            price=2999,
            screen_type="直屏",
            processor="骁龙8 Gen2",
            section_type="概述",
            parent_text="vivo iQOO 46 游戏性能旗舰，5000 万像素主摄，6.78 英寸 AMOLED 直屏，144Hz 刷新率。5000mAh 电池，120W 超快闪充。",
            score=0.009,
            source="both",
        ),
    ]


def make_mock_constraints():
    """创建模拟约束条件"""
    return StructuredConstraints(
        budget_max=3000.0,
        scenario="送礼",
        core_needs=["拍照"],
        negative_constraints=["曲面屏"],
        semantic_query="拍照好的手机",
        reasoning="用户预算3000元，送女朋友，需要拍照好，不要曲面屏",
    )


# ── 测试函数 ──────────────────────────────────────────────

def test_output_schema():
    """测试 1: Generator 输出 schema 验证"""
    print("=" * 60)
    print("测试 1: Generator 输出 Schema 验证")
    print("=" * 60)

    output = GeneratorOutput(
        recommendation_text="这是一段推荐文本",
        comparison_table="| 机型 | 价格 |\n|------|------|\n| 测试 | 1000 |",
    )

    assert isinstance(output.recommendation_text, str), "recommendation_text 应为 str"
    assert isinstance(output.comparison_table, str), "comparison_table 应为 str"
    assert len(output.recommendation_text) > 0, "recommendation_text 不应为空"
    assert "|" in output.comparison_table, "comparison_table 应包含 Markdown 表格标记"
    print("✅ 测试 1 通过: 输出 schema 正确")


def test_format_candidates():
    """测试 2: 候选商品格式化"""
    print("\n" + "=" * 60)
    print("测试 2: 候选商品格式化")
    print("=" * 60)

    candidates = make_mock_candidates()
    formatted = _format_candidates(candidates)

    assert "OPPO Reno 21" in formatted, "应包含机型名"
    assert "2999" in formatted, "应包含价格"
    assert "直屏" in formatted, "应包含屏幕类型"
    assert "天玑9200" in formatted, "应包含处理器"
    assert len(formatted) > 200, "格式化文本应有足够长度"
    print(f"✅ 测试 2 通过: 格式化文本长度 {len(formatted)} 字符")


def test_format_constraints():
    """测试 3: 约束条件格式化"""
    print("\n" + "=" * 60)
    print("测试 3: 约束条件格式化")
    print("=" * 60)

    constraints = make_mock_constraints()
    formatted = _format_constraints(constraints)

    assert "3000" in formatted, "应包含预算"
    assert "送礼" in formatted, "应包含场景"
    assert "拍照" in formatted, "应包含核心需求"
    assert "曲面屏" in formatted, "应包含负向约束"
    print(f"✅ 测试 3 通过: 约束条件格式化正确")
    print(f"   内容: {formatted}")


def test_generate_with_mock():
    """测试 4: 使用模拟数据测试 Generator 生成"""
    print("\n" + "=" * 60)
    print("测试 4: Generator 生成（模拟数据）")
    print("=" * 60)

    generator = GeneratorAgent()
    query = "预算3000，送女朋友，主要拍照好，不要曲面屏"
    constraints = make_mock_constraints()
    candidates = make_mock_candidates()

    output = generator.generate(query, constraints, candidates)

    # 验证推荐文本
    assert len(output.recommendation_text) > 50, \
        f"推荐文本过短 ({len(output.recommendation_text)} 字符)"
    print(f"   推荐文本长度: {len(output.recommendation_text)} 字符")

    # 验证对比表格
    assert "|" in output.comparison_table, "对比表格应包含 Markdown 表格标记"
    assert "---" in output.comparison_table, "对比表格应包含分隔行"
    print(f"   对比表格长度: {len(output.comparison_table)} 字符")

    print("✅ 测试 4 通过: Generator 输出格式正确")


def test_recommendation_mentions_needs():
    """测试 5: 推荐文本提到用户核心需求"""
    print("\n" + "=" * 60)
    print("测试 5: 推荐文本提到用户核心需求")
    print("=" * 60)

    generator = GeneratorAgent()
    query = "预算3000，送女朋友，主要拍照好，不要曲面屏"
    constraints = make_mock_constraints()
    candidates = make_mock_candidates()

    output = generator.generate(query, constraints, candidates)

    # 推荐文本应提到核心需求
    text = output.recommendation_text
    mentions_photo = any(kw in text for kw in ["拍照", "摄影", "摄像", "像素", "相机"])
    assert mentions_photo, f"推荐文本未提到拍照需求。文本前 200 字: {text[:200]}"

    # 推荐文本应提到预算相关
    mentions_budget = any(kw in text for kw in ["3000", "预算", "价位", "价格"])
    assert mentions_budget, f"推荐文本未提到预算。文本前 200 字: {text[:200]}"

    print(f"   提到拍照: {mentions_photo}")
    print(f"   提到预算: {mentions_budget}")
    print("✅ 测试 5 通过: 推荐文本绑定了用户需求")


def test_comparison_table_structure():
    """测试 6: 对比表格结构验证"""
    print("\n" + "=" * 60)
    print("测试 6: 对比表格结构验证")
    print("=" * 60)

    generator = GeneratorAgent()
    query = "预算3000，送女朋友，主要拍照好，不要曲面屏"
    constraints = make_mock_constraints()
    candidates = make_mock_candidates()

    output = generator.generate(query, constraints, candidates)

    table = output.comparison_table
    lines = [line.strip() for line in table.split("\n") if line.strip()]

    # 表格至少应有表头、分隔行、数据行
    assert len(lines) >= 3, f"表格行数不足: {len(lines)} 行"

    # 表头应包含关键列
    header = lines[0].lower()
    has_name = any(kw in header for kw in ["机型", "名称", "手机", "型号", "product", "name"])
    has_price = any(kw in header for kw in ["价格", "售价", "price", "元"])
    assert has_name, f"表头缺少机型列: {lines[0]}"
    assert has_price, f"表头缺少价格列: {lines[0]}"

    print(f"   表格行数: {len(lines)}")
    print(f"   表头: {lines[0]}")
    print("✅ 测试 6 通过: 对比表格结构正确")


def test_tool_calling_integration():
    """测试 7: Generator 成功触发 search_realtime_price 工具调用"""
    print("\n" + "=" * 60)
    print("测试 7: Generator 工具调用集成测试")
    print("=" * 60)

    from unittest.mock import patch
    generator = GeneratorAgent()
    query = "请联网查一下小米14的最新价格，并告诉我"
    constraints = make_mock_constraints()
    candidates = make_mock_candidates()

    with patch("agents.tools.DuckDuckGoSearchRun.invoke") as mock_search:
        mock_search.return_value = "小米 14 最新售价约为 2999 元"
        output = generator.generate(query, constraints, candidates)
        
        assert mock_search.called, "应该调用了联网查价工具"
        print("✅ 成功触发联网搜索工具！")
        assert len(output.recommendation_text) > 0


# ── 主入口 ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Generator Agent 测试套件")
    print("=" * 60)

    results = []
    results.append(("Schema 验证", test_output_schema()))
    results.append(("候选格式化", test_format_candidates()))
    results.append(("约束格式化", test_format_constraints()))
    results.append(("Generator 生成", test_generate_with_mock()))
    results.append(("需求绑定", test_recommendation_mentions_needs()))
    results.append(("表格结构", test_comparison_table_structure()))
    results.append(("工具调用集成", test_tool_calling_integration()))

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    if failed > 0:
        print("\n失败的测试:")
        for name, ok in results:
            if not ok:
                print(f"  ❌ {name}")
        sys.exit(1)
    else:
        print("\n✅ 所有测试通过!")
        sys.exit(0)


if __name__ == "__main__":
    main()
