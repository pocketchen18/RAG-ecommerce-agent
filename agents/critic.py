"""
Critic Agent — 反思审查（核心亮点）

对推荐结果执行 6 项结构化检查，确保推荐质量：

1. 价格检查（price_check）：推荐商品是否符合预算约束
2. 负向约束检查（negative_constraint_check）：是否违反负向约束（如曲面屏）
3. 多样性检查（diversity_check）：推荐是否过于集中（如同品牌过多）
4. 需求覆盖检查（needs_coverage_check）：推荐理由是否覆盖用户核心需求
5. 信息准确性检查（accuracy_check）：推荐信息是否与候选商品一致
6. 推荐完整性检查（completeness_check）：推荐文本是否完整

未通过时输出修改意见（revision_notes），可触发重新检索和生成。
"""
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from agents.planner import StructuredConstraints
from agents.generator import GeneratorOutput
from rag.retriever import RetrieverResult


# ── 输出 Schema ──────────────────────────────────────────────

class CheckResult(BaseModel):
    """单项检查结果"""
    name: str = Field(description="检查名称")
    passed: bool = Field(description="是否通过")
    details: str = Field(description="详细说明")


class CriticOutput(BaseModel):
    """Critic Agent 的输出"""
    passed: bool = Field(description="整体是否通过（所有检查均通过）")
    checks: List[CheckResult] = Field(description="6 项检查结果列表")
    revision_notes: str = Field(description="修改意见（未通过时给出具体建议）")
    score: float = Field(description="质量评分（0-10）")


# ── 规则检查函数 ──────────────────────────────────────────────

def _check_price_constraints(
    candidates: List[RetrieverResult],
    constraints: StructuredConstraints,
    recommendation_text: str,
) -> CheckResult:
    """
    价格检查：推荐商品是否符合预算约束

    检查逻辑：
    1. 从推荐文本中提取提到的机型名
    2. 在候选列表中找到对应商品
    3. 检查价格是否超出预算
    """
    budget_min = constraints.budget_min
    budget_max = constraints.budget_max

    # 如果没有预算约束，直接通过
    if budget_min is None and budget_max is None:
        return CheckResult(
            name="price_check",
            passed=True,
            details="未设置预算约束，跳过价格检查",
        )

    # 检查候选商品中是否有超预算的
    violations = []
    for c in candidates:
        if budget_max is not None and c.price > budget_max:
            violations.append(f"{c.name}（{c.price:.0f}元）超出最高预算 {budget_max:.0f}元")
        if budget_min is not None and c.price < budget_min:
            violations.append(f"{c.name}（{c.price:.0f}元）低于最低预算 {budget_min:.0f}元")

    if violations:
        return CheckResult(
            name="price_check",
            passed=False,
            details=f"发现 {len(violations)} 个价格违规: {'; '.join(violations[:3])}",
        )

    return CheckResult(
        name="price_check",
        passed=True,
        details=f"所有候选商品价格符合预算约束（{budget_min or 0}-{budget_max or '不限'}元）",
    )


def _check_negative_constraints(
    candidates: List[RetrieverResult],
    constraints: StructuredConstraints,
) -> CheckResult:
    """
    负向约束检查：推荐商品是否违反负向约束

    检查逻辑：
    1. 获取负向约束列表（如 '曲面屏'）
    2. 检查候选商品的 screen_type 是否包含负向关键词
    """
    negative_constraints = constraints.negative_constraints

    if not negative_constraints:
        return CheckResult(
            name="negative_constraint_check",
            passed=True,
            details="未设置负向约束，跳过检查",
        )

    violations = []
    for c in candidates:
        for constraint in negative_constraints:
            # 检查屏幕类型
            if constraint in c.screen_type:
                violations.append(f"{c.name}（{c.screen_type}）违反约束 '{constraint}'")
            # 检查商品名称和详情
            if constraint in c.name or constraint in c.parent_text:
                violations.append(f"{c.name} 包含 '{constraint}'")

    if violations:
        return CheckResult(
            name="negative_constraint_check",
            passed=False,
            details=f"发现 {len(violations)} 个负向约束违规: {'; '.join(violations[:3])}",
        )

    return CheckResult(
        name="negative_constraint_check",
        passed=True,
        details=f"所有候选商品符合负向约束（{', '.join(negative_constraints)}）",
    )


def _check_diversity(
    candidates: List[RetrieverResult],
    max_same_brand: int = 3,
) -> CheckResult:
    """
    多样性检查：推荐是否过于集中

    检查逻辑：
    1. 统计每个品牌的候选数量
    2. 如果某品牌超过 max_same_brand 个，视为多样性不足
    """
    if len(candidates) < 2:
        return CheckResult(
            name="diversity_check",
            passed=True,
            details="候选商品少于 2 个，跳过多样性检查",
        )

    # 统计品牌分布
    brand_counts: Dict[str, int] = {}
    for c in candidates:
        brand_counts[c.brand] = brand_counts.get(c.brand, 0) + 1

    # 检查是否有品牌过多
    violations = []
    for brand, count in brand_counts.items():
        if count > max_same_brand:
            violations.append(f"{brand} 品牌有 {count} 款，超过限制 {max_same_brand}")

    if violations:
        return CheckResult(
            name="diversity_check",
            passed=False,
            details=f"多样性不足: {'; '.join(violations)}",
        )

    # 品牌分布信息
    brand_info = ", ".join([f"{b}: {c}款" for b, c in brand_counts.items()])
    return CheckResult(
        name="diversity_check",
        passed=True,
        details=f"品牌分布合理（{brand_info}）",
    )


def _check_needs_coverage(
    constraints: StructuredConstraints,
    recommendation_text: str,
) -> CheckResult:
    """
    需求覆盖检查：推荐理由是否覆盖用户核心需求

    检查逻辑：
    1. 获取用户核心需求列表（如 ['拍照', '游戏性能']）
    2. 检查推荐文本中是否提到了这些需求关键词
    """
    core_needs = constraints.core_needs

    if not core_needs:
        return CheckResult(
            name="needs_coverage_check",
            passed=True,
            details="未设置核心需求，跳过检查",
        )

    # 检查推荐文本是否提到核心需求
    uncovered = []
    for need in core_needs:
        if need not in recommendation_text:
            uncovered.append(need)

    if uncovered:
        return CheckResult(
            name="needs_coverage_check",
            passed=False,
            details=f"推荐文本未覆盖核心需求: {', '.join(uncovered)}",
        )

    return CheckResult(
        name="needs_coverage_check",
        passed=True,
        details=f"推荐文本覆盖了所有核心需求（{', '.join(core_needs)}）",
    )


def _check_accuracy(
    candidates: List[RetrieverResult],
    comparison_table: str,
) -> CheckResult:
    """
    信息准确性检查：对比表格中的数据是否与候选商品一致

    检查逻辑：
    1. 从对比表格中提取机型名和价格
    2. 与候选商品列表对比
    """
    if not comparison_table or "暂无" in comparison_table:
        return CheckResult(
            name="accuracy_check",
            passed=False,
            details="对比表格为空或无数据",
        )

    # 从表格中提取机型名
    table_names = set()
    for line in comparison_table.split("\n"):
        if "|" in line and "---" not in line:
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts and parts[0] not in ["机型", "型号", "名称", ""]:
                table_names.add(parts[0])

    # 从候选列表中提取机型名
    candidate_names = {c.name for c in candidates}

    # 检查表格中的机型是否在候选列表中
    if not table_names:
        return CheckResult(
            name="accuracy_check",
            passed=False,
            details="对比表格中未找到有效机型名",
        )

    # 允许部分匹配（表格可能只包含部分候选）
    return CheckResult(
        name="accuracy_check",
        passed=True,
        details=f"对比表格包含 {len(table_names)} 款机型",
    )


def _check_completeness(
    recommendation_text: str,
    comparison_table: str,
) -> CheckResult:
    """
    推荐完整性检查：推荐文本和表格是否完整

    检查逻辑：
    1. 推荐文本长度是否足够（> 100 字符）
    2. 对比表格是否包含有效内容
    """
    issues = []

    # 检查推荐文本
    if len(recommendation_text) < 100:
        issues.append(f"推荐文本过短（{len(recommendation_text)} 字符）")

    if "失败" in recommendation_text or "错误" in recommendation_text:
        issues.append("推荐文本包含错误信息")

    # 检查对比表格
    if not comparison_table or len(comparison_table) < 50:
        issues.append("对比表格过短或为空")

    if "暂无" in comparison_table:
        issues.append("对比表格无有效数据")

    if issues:
        return CheckResult(
            name="completeness_check",
            passed=False,
            details=f"完整性问题: {'; '.join(issues)}",
        )

    return CheckResult(
        name="completeness_check",
        passed=True,
        details=f"推荐文本 {len(recommendation_text)} 字符，对比表格 {len(comparison_table)} 字符",
    )


# ── Prompt 模板（用于 LLM 辅助审查） ──────────────────────

CRITIC_SYSTEM_PROMPT = """你是一个推荐质量审查专家。
你的任务是对推荐结果进行审查，判断是否满足用户需求。

## 审查要点

1. **相关性**：推荐的商品是否符合用户的使用场景和核心需求
2. **理由质量**：推荐理由是否具体、有说服力，而非泛泛而谈
3. **遗漏检查**：是否有明显更适合的商品被遗漏
4. **改进建议**：如果推荐质量不佳，给出具体的改进方向

## 输出格式

请用以下格式输出：

VERDICT: [PASS/FAIL]
SCORE: [0-10 的数字]
ISSUES: [列出主要问题，用分号分隔]
SUGGESTIONS: [给出改进建议]
"""

CRITIC_HUMAN_PROMPT = """## 用户需求
"{query}"

## 约束条件
{constraints}

## 推荐文本
{recommendation_text}

## 对比表格
{comparison_table}

请审查以上推荐结果的质量。
"""


# ── Prompt 模板 ──────────────────────────────────────────────

class CriticAgent:
    """
    Critic Agent

    对推荐结果执行 6 项结构化检查，确保推荐质量。
    未通过时输出修改意见，可触发重新检索和生成。
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_same_brand: int = 3,
        use_llm: bool = False,
    ):
        """
        初始化 Critic Agent

        Args:
            model_name: 模型名称，默认使用 config 中的 LLM_MODEL
            temperature: 温度参数，0 表示确定性输出
            max_same_brand: 同品牌最大数量（多样性检查阈值）
            use_llm: 是否使用 LLM 进行辅助审查（默认关闭，使用规则检查）
        """
        self.model_name = model_name or LLM_MODEL
        self.temperature = temperature
        self.max_same_brand = max_same_brand
        self.use_llm = use_llm

        # 初始化 LLM（如果启用）
        if self.use_llm:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=LLM_API_KEY,
                base_url=LLM_API_BASE,
            )

            self.prompt = ChatPromptTemplate.from_messages([
                ("system", CRITIC_SYSTEM_PROMPT),
                ("human", CRITIC_HUMAN_PROMPT),
            ])

            self.chain = self.prompt | self.llm | StrOutputParser()

    def review(
        self,
        query: str,
        constraints: StructuredConstraints,
        generator_output: GeneratorOutput,
        candidates: List[RetrieverResult],
    ) -> CriticOutput:
        """
        审查推荐结果

        Args:
            query: 用户原始查询
            constraints: Planner 解析的结构化约束
            generator_output: Generator 生成的推荐文本和对比表格
            candidates: Retriever 返回的候选商品列表

        Returns:
            CriticOutput: 审查结果，包含 6 项检查和修改意见
        """
        # 找出实际推荐的商品
        recommended_candidates = []
        if hasattr(generator_output, 'recommended_product_names') and generator_output.recommended_product_names:
            for c in candidates:
                # 使用部分匹配增强鲁棒性
                if any(c.name in name or name in c.name for name in generator_output.recommended_product_names):
                    if c not in recommended_candidates:
                        recommended_candidates.append(c)
        
        # 确保推荐列表不为空，如果是空的可能是解析错误，退化回全部候选
        if not recommended_candidates:
            recommended_candidates = candidates

        # 执行 6 项规则检查
        checks = [
            _check_price_constraints(recommended_candidates, constraints, generator_output.recommendation_text),
            _check_negative_constraints(recommended_candidates, constraints),
            _check_diversity(recommended_candidates, self.max_same_brand),
            _check_needs_coverage(constraints, generator_output.recommendation_text),
            _check_accuracy(candidates, generator_output.comparison_table),
            _check_completeness(generator_output.recommendation_text, generator_output.comparison_table),
        ]

        # 判断整体是否通过
        all_passed = all(check.passed for check in checks)

        # 计算分数
        passed_count = sum(1 for check in checks if check.passed)
        score = round((passed_count / len(checks)) * 10, 1)

        # 生成修改意见
        revision_notes = self._generate_revision_notes(checks, constraints)

        # 如果启用 LLM，进行辅助审查
        llm_issues = []
        if self.use_llm:
            llm_issues = self._llm_review(
                query, constraints, generator_output
            )
            if llm_issues:
                all_passed = False
                revision_notes += "\n\nLLM 审查意见:\n" + "\n".join(llm_issues)

        return CriticOutput(
            passed=all_passed,
            checks=checks,
            revision_notes=revision_notes,
            score=score,
        )

    def _generate_revision_notes(
        self,
        checks: List[CheckResult],
        constraints: StructuredConstraints,
    ) -> str:
        """根据检查结果生成修改意见"""
        failed_checks = [c for c in checks if not c.passed]

        if not failed_checks:
            return "所有检查通过，推荐质量良好。"

        notes = []
        notes.append(f"发现 {len(failed_checks)} 项问题需要修复：")

        for i, check in enumerate(failed_checks, 1):
            notes.append(f"\n{i}. [{check.name}] {check.details}")

            # 给出具体修改建议
            if check.name == "price_check":
                if constraints.budget_max:
                    notes.append(f"   建议：筛选价格在 {constraints.budget_max:.0f} 元以内的商品")
            elif check.name == "negative_constraint_check":
                notes.append(f"   建议：排除包含 {', '.join(constraints.negative_constraints)} 的商品")
            elif check.name == "diversity_check":
                notes.append(f"   建议：增加品牌多样性，避免超过 {self.max_same_brand} 款同品牌商品")
            elif check.name == "needs_coverage_check":
                notes.append(f"   建议：在推荐理由中明确提到 {', '.join(constraints.core_needs)}")
            elif check.name == "completeness_check":
                notes.append("   建议：丰富推荐内容，确保推荐文本和对比表格完整")

        return "\n".join(notes)

    def _llm_review(
        self,
        query: str,
        constraints: StructuredConstraints,
        generator_output: GeneratorOutput,
    ) -> List[str]:
        """使用 LLM 进行辅助审查"""
        try:
            # 格式化约束
            constraints_text = self._format_constraints(constraints)

            # 调用 LLM
            raw_output = self.chain.invoke({
                "query": query,
                "constraints": constraints_text,
                "recommendation_text": generator_output.recommendation_text[:2000],
                "comparison_table": generator_output.comparison_table,
            })

            # 解析输出
            return self._parse_llm_output(raw_output)

        except Exception as e:
            return [f"LLM 审查失败: {str(e)}"]

    def _format_constraints(self, constraints: StructuredConstraints) -> str:
        """格式化约束条件"""
        parts = []
        if constraints.budget_max:
            parts.append(f"预算上限: {constraints.budget_max:.0f}元")
        if constraints.budget_min:
            parts.append(f"预算下限: {constraints.budget_min:.0f}元")
        if constraints.scenario:
            parts.append(f"场景: {constraints.scenario}")
        if constraints.core_needs:
            parts.append(f"核心需求: {', '.join(constraints.core_needs)}")
        if constraints.negative_constraints:
            parts.append(f"负向约束: {', '.join(constraints.negative_constraints)}")
        return "\n".join(parts) if parts else "无特殊约束"

    def _parse_llm_output(self, raw_output: str) -> List[str]:
        """解析 LLM 审查输出"""
        issues = []

        # 检查 VERDICT
        if "VERDICT: FAIL" in raw_output:
            # 提取 ISSUES
            for line in raw_output.split("\n"):
                if line.startswith("ISSUES:"):
                    issues_text = line.replace("ISSUES:", "").strip()
                    if issues_text:
                        issues.extend([i.strip() for i in issues_text.split(";") if i.strip()])
                elif line.startswith("SUGGESTIONS:"):
                    suggestion = line.replace("SUGGESTIONS:", "").strip()
                    if suggestion:
                        issues.append(f"建议: {suggestion}")

        return issues


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    from agents.planner import PlannerAgent
    from agents.generator import GeneratorAgent
    from rag.retriever import HybridRetriever

    print("=" * 60)
    print("Critic Agent 测试")
    print("=" * 60)

    # 初始化组件
    planner = PlannerAgent()
    retriever = HybridRetriever()
    generator = GeneratorAgent()
    critic = CriticAgent()

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
    generator_output = generator.generate(test_query, constraints, candidates)
    print(f"推荐文本长度: {len(generator_output.recommendation_text)} 字符")

    # Step 4: Critic 审查
    print("\n" + "=" * 60)
    print("开始 Critic 审查...")
    print("=" * 60)

    critic_output = critic.review(test_query, constraints, generator_output, candidates)

    print(f"\n✅ 整体结果: {'通过' if critic_output.passed else '未通过'}")
    print(f"📊 质量评分: {critic_output.score}/10")

    print("\n📋 检查详情:")
    for check in critic_output.checks:
        status = "✅" if check.passed else "❌"
        print(f"  {status} {check.name}: {check.details}")

    if not critic_output.passed:
        print("\n📝 修改意见:")
        print(critic_output.revision_notes)
