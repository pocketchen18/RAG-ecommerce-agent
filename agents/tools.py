from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
import logging

logger = logging.getLogger(__name__)

@tool
def search_realtime_price(sku_name: str) -> str:
    """当需要获取某款手机当前的最新实时价格时，调用此工具。
    
    Args:
        sku_name: 手机的型号名称 (例如: '小米14 Pro', 'iPhone 15')
        
    Returns:
        一段关于该手机最新参考售价的搜索结果文本摘要。
    """
    logger.info(f"正在使用 DuckDuckGo 搜索 {sku_name} 的实时价格...")
    import streamlit as st
    msg = f"🔍 正在联网查询 **{sku_name}** 的实时参考价格..."
    try:
        if "current_status_logs" not in st.session_state:
            st.session_state.current_status_logs = []
        st.session_state.current_status_logs.append(msg)
        st.write(msg)
    except Exception:
        pass
        
    try:
        search = DuckDuckGoSearchRun()
        query = f"{sku_name} 最新 售价 价格 京东 淘宝"
        res = search.invoke(query)
        logger.info(f"搜索完成，结果长度: {len(res) if res else 0}")
        
        # 提取搜索到的简短价格线索
        import re
        prices = re.findall(r'(\d+)\s*元', res)
        price_msg = f"💡 联网搜索到相关价格线索: {', '.join(prices[:3])} 元" if prices else "⚠️ 未在网页摘要中找到明确的价格数值"
        try:
            st.write(price_msg)
            st.session_state.current_status_logs.append(price_msg)
        except Exception:
            pass
            
        return res
    except Exception as e:
        logger.error(f"联网搜索价格失败: {str(e)}")
        err_msg = f"❌ 联网搜索价格失败: {str(e)}"
        try:
            st.write(err_msg)
            st.session_state.current_status_logs.append(err_msg)
        except Exception:
            pass
        return f"查询出错: {str(e)}"
