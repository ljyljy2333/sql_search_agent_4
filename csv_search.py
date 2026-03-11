import streamlit as st
import pandas as pd
from llm import ReportBuildingAgent
from dotenv import load_dotenv,find_dotenv
_ = load_dotenv(find_dotenv(), verbose=True, override=True)
import uuid
import os

st.set_page_config(page_title="SMART OFFER REPORT ASSISTANT", layout="wide")
st.title("SMART_OFFER_REPORT_ASSISTANT🤖")
st.markdown("Support natural language database search, mathematical calculations, and information summarization。")

# ① 初始化 thread_id
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# ② 初始化 agent
if "agent" not in st.session_state:
    st.session_state.agent = ReportBuildingAgent()

# Display session status in sidebar
with st.sidebar:
    st.header("SYSTEM_STATUS")
    if st.button("New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())
        # cache_path = f"./cache/{st.session_state.thread_id}"
        # os.makedirs(cache_path, exist_ok=True)
        st.session_state.agent = ReportBuildingAgent()
        st.session_state.history = []
    if st.button("ClearHistory"):
        st.session_state.history = []
        st.session_state.agent_history = []
        st.rerun()
# Display chat history


st.session_state.history = st.session_state.agent.load_chat_history_for_front(
    st.session_state.thread_id)

def render_chat_history(history):
    for msg in history:
        with st.chat_message(msg["role"]):

            # 用户消息
            if msg["role"] == "user":
                st.write(msg["content"])

            # assistant 消息
            else:
                intent = msg.get("intent")

                if intent:
                    st.info(
                        f"IDENTIFY_INTENT: **{intent['intent_type']}** "
                        f"(CONFIDENCE: {intent['confidence']:.2f})"
                    )

                st.subheader("ANALYZE_THE_RESULTS")
                st.write(msg["content"])

                sql_answer = msg.get("sql_answer")

                if isinstance(sql_answer, list) and len(sql_answer) > 0:
                    df = pd.DataFrame(sql_answer)
                    st.table(df)

                if msg.get("timestamp") and msg.get("answer_confidence"):
                    st.caption(
                        f"ResponseTime: {msg['timestamp']} | "
                        f"AnswerReliability: {msg['answer_confidence']:.2f}"
                    )

render_chat_history(st.session_state.history)

# Search input form
query = st.chat_input("Please enter your question (e.g. 'calculate 129*0.85', 'Retrieve all offers containing 'KFC''):")
if query:
    with st.chat_message("user"):
        st.write(query)

if query:
    with st.spinner("THINKING..."):
        try:
            # Run LangGraph
            thread_id = st.session_state.thread_id

            # 1. 调用 run()，内部已经检查 cache
            result = st.session_state.agent.run(query, thread_id=thread_id)

            intent = result['intent']
            answer = result['final_answer']


            # Display intent recognition result
            st.info(f"IDENTIFY_INTENT: **{intent.intent_type}** (CONFIDENCE: {intent.confidence:.2f})")

            # Display final answer
            with st.chat_message("assistant"):

                st.subheader("ANALYZE_THE_RESULTS")
                st.write(answer.content)


                sql_answer = result.get("sql_answer")
                if isinstance(sql_answer, list) and len(sql_answer) > 0:
                    df = pd.DataFrame(sql_answer)
                    st.table(df)

                # Display metadata
                st.caption(f"ResponseTime: {answer.timestamp} | AnswerReliability: {answer.confidence:.2f}")

        except Exception as e:
            st.error(f"OperationError: {str(e)}")

# Example display
with st.expander("Example Queries"):
    st.write("- **Search**: Retrieve all offers containing 'KFC'")
    st.write("- **Math**: (500 - 120) × 0.9")
    st.write("- **Summary**: Provide a summary of the current offer data")