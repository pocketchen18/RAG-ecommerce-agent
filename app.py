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
    
    # 读取环境变量，如果不存在则使用 config.py 里的默认值
    from config import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, EMBEDDING_API_BASE, EMBEDDING_API_KEY, EMBEDDING_MODEL
    
    # 初始化 session state
    if "llm_key" not in st.session_state:
        st.session_state.llm_key = os.getenv("LLM_API_KEY", LLM_API_KEY)
    if "llm_base" not in st.session_state:
        st.session_state.llm_base = os.getenv("LLM_API_BASE", LLM_API_BASE)
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = os.getenv("LLM_MODEL", LLM_MODEL)

    if "emb_key" not in st.session_state:
        st.session_state.emb_key = os.getenv("EMBEDDING_API_KEY", EMBEDDING_API_KEY)
    if "emb_base" not in st.session_state:
        st.session_state.emb_base = os.getenv("EMBEDDING_API_BASE", EMBEDDING_API_BASE)
    if "emb_model" not in st.session_state:
        st.session_state.emb_model = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)

    with st.expander("🛠️ LLM 配置", expanded=True):
        llm_key = st.text_input("LLM API Key", type="password", value=st.session_state.llm_key)
        llm_base = st.text_input("LLM Base URL", value=st.session_state.llm_base)
        llm_model = st.text_input("LLM Model", value=st.session_state.llm_model)

    with st.expander("🛠️ Embedding 配置", expanded=False):
        emb_key = st.text_input("Embedding API Key", type="password", value=st.session_state.emb_key)
        emb_base = st.text_input("Embedding Base URL", value=st.session_state.emb_base)
        emb_model = st.text_input("Embedding Model", value=st.session_state.emb_model)
    
    # 实时同步到环境变量，保证运行期代码读取最新配置
    os.environ["LLM_API_KEY"] = llm_key
    os.environ["LLM_API_BASE"] = llm_base
    os.environ["LLM_MODEL"] = llm_model
    os.environ["EMBEDDING_API_KEY"] = emb_key
    os.environ["EMBEDDING_API_BASE"] = emb_base
    os.environ["EMBEDDING_MODEL"] = emb_model
    
    st.session_state.llm_key = llm_key
    st.session_state.llm_base = llm_base
    st.session_state.llm_model = llm_model
    st.session_state.emb_key = emb_key
    st.session_state.emb_base = emb_base
    st.session_state.emb_model = emb_model

    if st.button("💾 保存配置"):
        env_content = f"""# ── 嵌入模型 API（用于 RAG 向量检索） ────────────────────
EMBEDDING_API_BASE={emb_base}
EMBEDDING_API_KEY={emb_key}
EMBEDDING_MODEL={emb_model}

# ── 语言模型 API（用于 Agent 推理、生成、审查） ──────────
LLM_API_BASE={llm_base}
LLM_API_KEY={llm_key}
LLM_MODEL={llm_model}
"""
        env_path = Path(__file__).resolve().parent / ".env"
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        st.success(f"✅ 成功保存至 .env 文件，下次启动将自动加载！")
        
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
from config import CHROMA_DIR
import chromadb

# 检查向量数据库是否已初始化
client = chromadb.PersistentClient(path=str(CHROMA_DIR))
try:
    client.get_collection("phone_products")
    db_ready = True
except ValueError:
    db_ready = False

if not db_ready:
    st.warning("⚠️ 检测到本地向量数据库尚未初始化（缺少商品数据）。")
    if st.button("🚀 立即初始化数据库（约需 1 分钟）"):
        with st.spinner("正在调用大模型生成向量数据并入库，请耐心等待..."):
            from rag.indexer import main as index_main
            try:
                success = index_main()
                if success is not False:
                    st.success("✅ 数据库初始化成功！")
                    import time
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ 初始化失败，请检查数据文件是否存在。")
            except Exception as e:
                st.error(f"❌ 初始化异常: {str(e)}")
    st.stop()

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
                        budget_max = constraints.get("budget_max") if isinstance(constraints, dict) else getattr(constraints, "budget_max", None)
                        scenario = constraints.get("scenario") if isinstance(constraints, dict) else getattr(constraints, "scenario", None)
                        if budget_max:
                            st.metric("预算上限", f"{budget_max:.0f} 元")
                        if scenario:
                            st.metric("使用场景", scenario)
                    with cols[1]:
                        core_needs = constraints.get("core_needs") if isinstance(constraints, dict) else getattr(constraints, "core_needs", None)
                        brands = constraints.get("brands") if isinstance(constraints, dict) else getattr(constraints, "brands", None)
                        if core_needs:
                            st.metric("核心需求", ", ".join(core_needs))
                        if brands:
                            st.metric("品牌偏好", ", ".join(brands))

                # Retriever 步骤
                candidates = result.get("candidates", [])
                if candidates:
                    st.markdown(f"**🔍 Retriever 检索的候选:** {len(candidates)} 个商品")
                    for j, cand in enumerate(candidates[:5]):  # 只显示前5个
                        name = cand.get("name") if isinstance(cand, dict) else getattr(cand, "name", "N/A")
                        price = cand.get("price") if isinstance(cand, dict) else getattr(cand, "price", "N/A")
                        st.caption(f"{j+1}. {name} - ¥{price}")

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

            final_state = initial_state.model_dump() if hasattr(initial_state, "model_dump") else (initial_state.dict() if hasattr(initial_state, "dict") else dict(initial_state))

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
                        budget_max = constraints.get("budget_max") if isinstance(constraints, dict) else getattr(constraints, "budget_max", None)
                        scenario = constraints.get("scenario") if isinstance(constraints, dict) else getattr(constraints, "scenario", None)
                        if budget_max:
                            st.metric("预算上限", f"{budget_max:.0f} 元")
                        if scenario:
                            st.metric("使用场景", scenario)
                    with cols[1]:
                        core_needs = constraints.get("core_needs") if isinstance(constraints, dict) else getattr(constraints, "core_needs", None)
                        brands = constraints.get("brands") if isinstance(constraints, dict) else getattr(constraints, "brands", None)
                        if core_needs:
                            st.metric("核心需求", ", ".join(core_needs))
                        if brands:
                            st.metric("品牌偏好", ", ".join(brands))

                # Retriever 步骤
                candidates = result.get("candidates", [])
                if candidates:
                    st.markdown(f"**🔍 Retriever 检索的候选:** {len(candidates)} 个商品")
                    for j, cand in enumerate(candidates[:5]):  # 只显示前5个
                        name = cand.get("name") if isinstance(cand, dict) else getattr(cand, "name", "N/A")
                        price = cand.get("price") if isinstance(cand, dict) else getattr(cand, "price", "N/A")
                        st.caption(f"{j+1}. {name} - ¥{price}")

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
