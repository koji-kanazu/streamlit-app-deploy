import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ===== 環境変数読み込み =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ===== 設定（最小）=====
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 300
MAX_INPUT_CHARS = 2000

def ask_llm(user_text: str, expert_type: str) -> str:
    if not api_key:
        return "OPENAI_API_KEYが読み込めていません。.env を確認してください。"

    if expert_type == "ITコンサルタント":
        system_prompt = "あなたは優秀なITコンサルタントです。結論→理由→具体策の順で簡潔に答えてください。"
    else:
        system_prompt = (
            "あなたは医療アドバイザーです。診断はせず、一般的な情報として、"
            "受診の目安や注意点を安全寄りに簡潔に答えてください。"
        )

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        api_key=api_key,
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_text),
    ]
    return llm.invoke(messages).content

# ===== UI（最小）=====
st.title("LLMアプリ課題テスト")
st.write("入力したテキストに対して、選択した専門家の立場で回答を生成します。")

expert_type = st.radio(
    "専門家の種類を選択してください",
    ["ITコンサルタント", "医療アドバイザー"],
    key="expert_type_radio",  # ← duplicate対策
)

user_text = st.text_area("質問・相談内容を入力してください", height=200)

# 入力の簡易制限
if len(user_text) > MAX_INPUT_CHARS:
    st.warning(f"入力が長すぎます（{len(user_text)}文字）。{MAX_INPUT_CHARS}文字以内にしてください。")

if st.button("実行"):
    if not user_text.strip():
        st.error("入力してください。")
    elif len(user_text) > MAX_INPUT_CHARS:
        st.error("入力が長すぎます。短くして再実行してください。")
    else:
        with st.spinner("生成中..."):
            answer = ask_llm(user_text, expert_type)
        st.subheader("回答")
        st.write(answer)

# 参考情報（任意）
st.caption(f"model={MODEL_NAME} / max_output_tokens={MAX_OUTPUT_TOKENS}")
