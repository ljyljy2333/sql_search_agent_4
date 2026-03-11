import os
import sqlite3
import re
from datetime import datetime
from typing import Annotated, Literal, TypedDict, List, Optional, Union,Any

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from pydantic import BaseModel, Field, validator, ConfigDict
from langchain_openai import AzureChatOpenAI, OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import tool
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv,find_dotenv
from simpleeval import simple_eval
_ = load_dotenv(find_dotenv(), verbose=True, override=True)


# --- 1. Pydantic Schemas (Data validation) ---

class UserIntent(BaseModel):
    """Schema for identifying user intent"""
    intent_type: Literal['qa', 'calculation', 'summarization', 'general'] = Field(description="Type of the user query")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of the intent classification")
    reasoning: str = Field(description="Reason for the intent classification")


class AnswerResponse(BaseModel):
    """Schema for the final output"""
    content: str = Field(description="Response content")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence of the answer")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


# --- 2. State definition ---

class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True,from_attributes=True)
    query: str
    intent: Optional[Union[UserIntent, dict]] = None
    sql_results: Optional[str] = None
    tool_output: Optional[str] = None
    sql_answer: Optional[List[dict]] = None
    final_answer: Optional[Union[AnswerResponse, dict]] = None
    is_cached: bool = False  # 默认为 False
    #history: List[dict] = Field(default_factory=list)

# --- 3. Tools (Tool implementation) ---

@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Only numbers and basic operators (+-*/().) are allowed."""
    # Security check: only allow mathematical characters
    replacements = {
        "×": "*",  # 乘号
        "✕": "*",  # 另一种乘号
        "÷": "/",  # 除号
        "−": "-",  # Unicode 减号
        "–": "-",  # 长破折号，也有人输入
        "﹢": "+",  # Unicode 加号
        "．": ".",  # 全角小数点
        "。": ".",  # 中文句号当小数点
        "（": "(",  # 中文括号
        "）": ")",  # 中文括号
        " ": " ",  # 可以统一空格（可选）
    }

    # 批量替换
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    expression = expression.strip()
    if not re.match(r"^[0-9+\-*/().\s]+$", expression):
        return "Error: Expression contains illegal characters. For safety, only basic operations are supported."

    try:
        # Additional filtering before using eval (in production simpleeval is recommended)
        result = simple_eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"




# --- 4. Core class implementation ---

class ReportBuildingAgent:
    def __init__(self):

        self.db = SQLDatabase.from_uri("sqlite:///offer_db.sqlite")
        # # ① 先创建 memory
        # self.cache_path = cache_path
        # # self.thread_id = thread_id
        # # MemorySaver
        # os.makedirs(cache_path, exist_ok=True)
        # sqlite_path = os.path.join(cache_path, "csv_search_4eng.db")
        # conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.memory = MemorySaver()

        # ② 再 build graph
        self.graph = self._build_graph()
        # self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.llm = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=0
        )

        graph_png = self.graph.get_graph().draw_mermaid_png()
        with open("csv_searcher.png", "wb") as f:
            f.write(graph_png)
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.environ["text-embedding_3_large_deployment"],
            api_version=os.environ["text-embedding_3_large_api_version"]
        )

    def parse_output(self, retrieved_offers: str, query: str):
        from langchain_community.vectorstores import FAISS
        top_offers = retrieved_offers.split("#")
        vector_db = FAISS.from_texts(texts=top_offers, embedding=self.embeddings)
        docs_and_scores = vector_db.similarity_search_with_score(query, k=len(top_offers))

        df = pd.DataFrame([
            {"distanceScore %": score, "offer": doc.page_content}
            for doc, score in docs_and_scores
        ])
        df.index += 1
        return df

    def _get_chat_prompt_template(self, intent: str) -> ChatPromptTemplate:
        templates = {
            "qa": "You are a data analyst. Answer the user's question about the offer based on the SQL query results. Result: {context}",
            "summarization": "You are a refined assistant. Please summarize the following offer information and highlight the key offers：{context}",
            "calculation": "You are a math expert. Please explain the calculation process and give the result：{context}",
            "general": "You are a friendly assistant. Please answer the user's question。"
        }
        sys_msg = templates.get(intent, templates["general"])
        return ChatPromptTemplate.from_messages([
            ("system", sys_msg),
            # ("placeholder", "{history}"),
            ("human", "{query}"),
        ])

    def load_chat_history_for_front(self, thread_id: str = "default"):
        from datetime import datetime
        config = {"configurable": {"thread_id": thread_id}}

        # 1. 获取该 thread_id 下的所有历史快照
        # 注意：snapshots 是按时间倒序排列的（最新的在前）
        snapshots = list(self.graph.get_state_history(config))

        chat_history = []
        seen_queries = set()  # 用于确保每个 Query 只记录一次最完整的回复

        for snapshot in snapshots:
            state_values = snapshot.values

            # --- 核心过滤逻辑 ---
            # A. 必须有 final_answer (说明这步产生过答案)
            # B. snapshot.next 为空 (说明这是流程的终点，即到达了 __end__)
            # C. 该 query 还没处理过 (倒序遍历中，第一个被处理的总是该 query 最新的终态)

            query = state_values.get("query", "").strip()

            if (state_values.get("final_answer") and
                    not snapshot.next and
                    query not in seen_queries):

                seen_queries.add(query)  # 标记该问题已处理

                # 1. 处理 AnswerResponse 对象 (final_answer)
                ans_obj = state_values["final_answer"]
                # 兼容处理：将 Pydantic 对象转为字典
                ans_dict = ans_obj.model_dump() if hasattr(ans_obj, 'model_dump') else ans_obj.__dict__

                # 时间对象转字符串
                ts = ans_dict.get("timestamp")
                if isinstance(ts, datetime):
                    formatted_ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    formatted_ts = str(ts) if ts else ""

                # 2. 处理 UserIntent 对象 (intent)
                intent_obj = state_values.get("intent")
                intent_dict = None
                if intent_obj:
                    intent_dict = intent_obj.model_dump() if hasattr(intent_obj, 'model_dump') else intent_obj.__dict__

                # --- 组装 Assistant 消息 ---
                chat_history.append({
                    "role": "assistant",
                    "content": ans_dict.get("content", ""),
                    "intent": intent_dict,
                    "sql_answer": state_values.get("sql_answer"),
                    "timestamp": formatted_ts,
                    "confidence": ans_dict.get("confidence")
                })

                # --- 组装 User 消息 ---
                chat_history.append({
                    "role": "user",
                    "content": query
                })

        # 因为 snapshots 是倒序取的（从新到旧），
        # 经过上面的 append 后，chat_history 里是 [最新助手, 最新用户, 旧助手, 旧用户]
        # 所以返回前需要整体反转一次，让前端显示为：[旧用户, 旧助手, ..., 新用户, 新助手]
        return chat_history[::-1]

    # --- Node functions ---

    def intent_classifier(self, state: AgentState):
        structured_llm = self.llm.with_structured_output(UserIntent)
        system_prompt = "Analyze and categorize user queries: 'qa' (query database), 'calculation' (mathematical calculation), 'summarization' (summarization information), 'general' (other)."
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # ("placeholder", "{history}"),
            ("human", "{query}")
        ])
        #llm_history = self.format_history_for_llm(state.history)
        intent = structured_llm.invoke(prompt.format(query=state.query))
        return {"intent": intent}

    def sql_search_node(self, state: AgentState):
        PROMPT_TEMPLATE = """
            You receive a query and your task is to retrieve the relevant offer from the 'OFFER' field in the 'offer_retailer' table.
            Queries can be mixed case, so search for the uppercase version of the query as well.
            Importantly, you may need to use information from other tables in the database, i.e.: 'brand_category', 'categories', 'offer_retailer', to retrieve the correct offer.
            Don't make up offers. If you can't find an offer in the 'offer_retailer' table, return the string: 'NONE'.
            If you can retrieve offers from the 'offer_retailer' table, separate each offer with the separator '#'. For example, the output should look something like this: 'offer1#offer2#offer3'.
            If SQLResult is empty, return 'None'. Do not generate any offers.
            Don't return any Markdown formatting, don't start or end with ''.
            Only plain text SQL statements are returned.

            This is the query: '{}'
        """
        sqlichain_prompt = PROMPT_TEMPLATE.format(state.query)
        from langchain_experimental.sql import SQLDatabaseChain
        db_chain = SQLDatabaseChain.from_llm(self.llm, self.db)
        try:
            res = db_chain.run(sqlichain_prompt)
            df = self.parse_output(res, state.query)
            return {"sql_results": res, "sql_answer": df.to_dict(orient="records")}
        except Exception as e:
            return {"sql_results": f"Query failed: {str(e)}"}

    def calculator_node(self, state: AgentState):
        expr = state.query
        result = calculator.invoke(expr)
        return {"tool_output": result}

    def final_generator(self, state: AgentState):
        intent_type = state.intent.intent_type
        context = state.sql_results or state.tool_output or ""
        prompt_tmpl = self._get_chat_prompt_template(intent_type)
        chain = prompt_tmpl | self.llm.with_structured_output(AnswerResponse)
        # llm_history = self.format_history_for_llm(state.history)
        response = chain.invoke({"query": state.query, "context": context})
        return {"final_answer": response}

    def reset_state(self,state):
        # 只保留 query，清空其他所有字段
        return {
            "intent": None,
            "sql_results": None,
            "tool_output": None,
            "sql_answer": None,
            "final_answer": None,
            "is_cached": False

        }

    def check_cache(self, state: AgentState, config: dict):
        # 提取 thread_id，构建一个纯净的 config
        tid = config.get("configurable", {}).get("thread_id")
        clean_config = {"configurable": {"thread_id": tid}}

        # 使用纯净的 config 查询
        history = list(self.graph.get_state_history(clean_config))

        current_query = state.query.strip()
        # 调试打印，确认这次 history 是否有内容
        print(f"DEBUG: History length for thread {tid} is {len(history)}")

        # 重点：我们需要找的是“过去”的记录
        # 历史记录是按时间倒序排的，snapshots[0] 通常是当前这次运行的初始状态
        for snapshot in history[1:]:# 倒序遍历
            h_values = snapshot.values

            # 核心逻辑：
            # 1. 必须有 final_answer (说明是之前成功运行完的)
            # 2. query 匹配
            # 3. 确保这不是当前正在进行的这一步 (可以通过 snapshot.next 的状态来辅助判断)
            if (h_values.get("query") == current_query and
                    h_values.get("final_answer") is not None):
                # 命中缓存
                return {
                    "is_cached": True,  # 👈 标记命中
                    "intent": h_values.get("intent"),
                    "sql_results": h_values.get("sql_results"),
                    "sql_answer": h_values.get("sql_answer"),
                    "tool_output": h_values.get("tool_output"),
                    "final_answer": h_values.get("final_answer")
                }
        return {"is_cached": False}
    # --- Graph construction ---

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("classify", self.intent_classifier)
        workflow.add_node("retrieve", self.sql_search_node)
        workflow.add_node("calculate", self.calculator_node)
        workflow.add_node("generate", self.final_generator)
        workflow.add_node("reset", self.reset_state)
        workflow.add_node("check_cache", lambda state, config: self.check_cache(state, config))



        # 定义路由

        def route_by_intent(state: AgentState):
            it = state.intent.intent_type
            if it == "calculation": return "calculate"
            if it == "qa" or it == "summarization": return "retrieve"
            return "generate"

        def cache_router(state: AgentState):
            """
            根据 check_cache 节点是否设置了 is_cached 标签来决定去向
            """
            if state.is_cached:
                return "end"
            return "reset"

        workflow.set_entry_point("check_cache")
        workflow.add_conditional_edges(
            "check_cache",
            cache_router,
            {
                "end": END,
                "reset": "reset"
            }
        )

        workflow.add_edge("reset", "classify")
        workflow.add_conditional_edges(
            "classify",
            route_by_intent,
            {
                "calculate": "calculate",
                "retrieve": "retrieve",
                "generate": "generate"
            }
        )

        workflow.add_edge("calculate", "generate")

        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)


        return workflow.compile(checkpointer=self.memory)

    def run(self, query: str,thread_id: str = "default"):

        state = self.graph.invoke(
            {"query": query},
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        # state 已经更新 history 并保存到 MemorySaver
        return state


