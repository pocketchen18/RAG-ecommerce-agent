"""
Streamlit 前端 — 聊天交互与推荐展示

功能：
- 用户在聊天框输入需求
- Agent 返回推荐结果
- 界面展示推荐卡片、对比表格和反思日志
"""
import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
import os


# ── 页面配置 ──────────────────────────────────────────────

st.set_page_config(
    page_title="📱 手机导购 Agent",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义样式 ──────────────────────────────────────────────

st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .check-pass {
        color: #28a745;
    }
    .check-fail {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


# ── 初始化 session state ──────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph_results" not in st.session_state:
    st.session_state.graph_results = []


# ── 侧边栏 ──────────────────────────────────────────────

with st.sidebar:
    st.header("🔑 API 配置")
    
    # 初始化 session state 用于存储 keys
    if "llm_key" not in st.session_state:
        st.session_state.llm_key = os.getenv("LLM_API_KEY", "")
    if "emb_key" not in st.session_state:
        st.session_state.emb_key = os.getenv("EMBEDDING_API_KEY", "")
        
    llm_key = st.text_input("LLM API Key", type="password", value=st.session_state.llm_key)
    emb_key = st.text_input("Embedding API Key", type="password", value=st.session_state.emb_key)
    
    if llm_key:
        os.environ["LLM_API_KEY"] = llm_key
        st.session_state.llm_key = llm_key
    if emb_key:
        os.environ["EMBEDDING_API_KEY"] = emb_key
        st.session_state.emb_key = emb_key
        
    st.divider()

    st.header("⚙️ 参数配置")

    max_iterations = st.slider(
        "最大迭代次数",
        min_value=1,
        max_value=5,
        value=3,
        help="Critic 未通过时的最大重试次数",
    )

    st.divider()

    st.header("📖 使用说明")
    st.markdown("""
    1. 在聊天框输入你的手机需求
    2. 等待 Agent 分析和推荐
    3. 查看推荐结果和对比表格
    4. 侧边栏展示反思日志

    **示例查询**:
    - 预算3000，送女朋友，主要拍照好
    - 打游戏用的，预算不限
    - 给父母买个手机，2000左右，续航好
    - 4000元以内小米手机，不要曲面屏
    """)

    st.divider()

    st.header("🔄 反思日志")
    if st.session_state.graph_results:
        latest = st.session_state.graph_results[-1]
        reflection_log = latest.get("reflection_log", [])

        if reflection_log:
            for entry in reflection_log:
                with st.expander(
                    f"第 {entry['iteration']} 轮 - "
                    f"{'✅ 通过' if entry['passed'] else '❌ 未通过'} "
                    f"(评分: {entry['score']}/10)"
                ):
                    for check in entry.get("checks", []):
                        icon = "✅" if check["passed"] else "❌"
                        st.markdown(f"{icon} **{check['name']}**")
                        st.caption(check["details"])

                    if not entry["passed"] and entry.get("revision_notes"):
                        st.markdown("**修改意见:**")
                        st.info(entry["revision_notes"])
        else:
            st.info("暂无反思日志")
    else:
        st.info("请先发送查询")


# ── 主界面 ──────────────────────────────────────────────

st.title("📱 手机导购 Agent")
st.markdown("基于 LangGraph 的「生成-批判」双 Agent 电商导购系统")

# 检查 API Key
if not st.session_state.llm_key or not st.session_state.emb_key:
    st.warning("⚠️ 请先在左侧配置 LLM 和 Embedding 的 API Key。")
    st.stop()

# 延迟导入，确保在读取了最新的 os.environ 后再进行初始化
from agents.graph import build_graph, GraphState

# 显示历史消息
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # 如果是 assistant 消息，显示步骤详情
        if message["role"] == "assistant" and i < len(st.session_state.graph_results):
            result = st.session_state.graph_results[i]

            # 显示步骤详情
            with st.expander("🔍 查看 Agent 思考过程", expanded=False):
                # Planner 步骤
                constraints = result.get("constraints")
                if constraints:
                    st.markdown("**📝 Planner 解析的约束:**")
                    cols = st.columns(2)
                    with cols[0]:
                        if constraints.get("budget_max"):
                            st.metric("预算上限", f"{constraints['budget_max']:.0f} 元")
                        if constraints.get("scenario"):
                            st.metric("使用场景", constraints["scenario"])
                    with cols[1]:
                        if constraints.get("core_needs"):
                            st.metric("核心需求", ", ".join(constraints["core_needs"]))
                        if constraints.get("brands"):
                            st.metric("品牌偏好", ", ".join(constraints["brands"]))

                # Retriever 步骤
                candidates = result.get("candidates", [])
                if candidates:
                    st.markdown(f"**🔍 Retriever 检索的候选:** {len(candidates)} 个商品")
                    for j, cand in enumerate(candidates[:5]):  # 只显示前5个
                        st.caption(f"{j+1}. {cand.get('name', 'N/A')} - ¥{cand.get('price', 'N/A')}")

# 聊天输入
if prompt := st.chat_input("请描述你的手机需求..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # 运行 Agent
    with st.chat_message("assistant"):
        try:
            # 构建图
            graph = build_graph()

            # 初始状态
            initial_state = GraphState(
                query=prompt,
                chat_history=st.session_state.messages[:-1],  # 传入历史上下文，排除当前的新 query
                max_iterations=max_iterations,
            )

            final_state = initial_state.dict() if hasattr(initial_state, "dict") else dict(initial_state)

            # 状态可视化
            with st.status("🧠 开始解析用户意图...", expanded=True) as status:
                for output in graph.stream(initial_state):
                    for node_name, node_update in output.items():
                        # 更新当前状态
                        if isinstance(node_update, dict):
                            final_state.update(node_update)
                        
                        # 根据节点更新状态文案
                        if node_name == "planner":
                            status.update(label="🔍 正在检索候选商品...")
                            st.write("✅ 意图解析完成")
                        elif node_name == "retriever":
                            status.update(label="✍️ 正在生成个性化推荐...")
                            cands = node_update.get("candidates", [])
                            st.write(f"✅ 检索到 {len(cands)} 款候选商品")
                        elif node_name == "generator":
                            status.update(label="🤔 Critic 正在严格审查...")
                            st.write("✅ 推荐话术与对比表格已生成")
                        elif node_name == "critic":
                            critic_out = node_update.get("critic_output")
                            if critic_out and not critic_out.passed:
                                status.update(label="🔄 审查未通过，正在反思重试...")
                                st.write("❌ 审查未通过，打回重做")
                            else:
                                status.update(label="📋 正在排版最终结果...")
                                st.write("✅ 审查通过")
                        elif node_name == "presenter":
                            status.update(label="🎉 思考完成！", state="complete", expanded=False)

            # 提取结果
            result = {
                "final_output": final_state.get("final_output", ""),
                "reflection_log": final_state.get("reflection_log", []),
                "iteration": final_state.get("iteration", 0),
                "constraints": final_state.get("constraints"),
                "candidates": final_state.get("candidates", []),
            }

            # 保存结果
            st.session_state.graph_results.append(result)

            # 流式输出最终文本 (模拟打字机效果)
            import time
            def stream_text(text):
                for char in text:
                    yield char
                    time.sleep(0.01)

            st.write_stream(stream_text(result["final_output"]))

            # 添加到消息历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["final_output"],
            })

            # 显示步骤详情
            with st.expander("🔍 查看 Agent 思考过程", expanded=False):
                # Planner 步骤
                constraints = result.get("constraints")
                if constraints:
                    st.markdown("**📝 Planner 解析的约束:**")
                    cols = st.columns(2)
                    with cols[0]:
                        if constraints.budget_max:
                            st.metric("预算上限", f"{constraints.budget_max:.0f} 元")
                        if constraints.scenario:
                            st.metric("使用场景", constraints.scenario)
                    with cols[1]:
                        if constraints.core_needs:
                            st.metric("核心需求", ", ".join(constraints.core_needs))
                        if constraints.brands:
                            st.metric("品牌偏好", ", ".join(constraints.brands))

                # Retriever 步骤
                candidates = result.get("candidates", [])
                if candidates:
                    st.markdown(f"**🔍 Retriever 检索的候选:** {len(candidates)} 个商品")
                    for j, cand in enumerate(candidates[:5]):  # 只显示前5个
                        st.caption(f"{j+1}. {cand.name} - ¥{cand.price}")

        except Exception as e:
            error_msg = f"❌ 运行出错: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })
            st.session_state.graph_results.append({})

    # 刷新页面以更新侧边栏
    st.rerun()
