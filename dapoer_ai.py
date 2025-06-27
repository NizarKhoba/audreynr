import streamlit as st
from dapoer_module import create_agent

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

GOOGLE_API_KEY = st.text_input("🔐 Masukkan API Key Gemini kamu:", type="password")
if not GOOGLE_API_KEY:
    st.warning("Masukkan dulu API key dari Google Gemini.")
    st.stop()

try:
    agent = create_agent(GOOGLE_API_KEY)
except Exception as e:
    st.error(f"Gagal inisialisasi Agent Gemini:\n\n{e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hai! Ketik nama masakan, bahan, atau metode memasak yang kamu ingin tahu."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan resep masakan, bahan, dll..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang memasak jawaban... 🍳"):
            try:
                response = agent.run(prompt)
            except Exception as e:
                response = f"⚠️ Error:\n\n```\n{e}\n```"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
