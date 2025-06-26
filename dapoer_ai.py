# dapoer_ai.py
import streamlit as st
from dapoer_module import create_agent

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Inisialisasi agent (API key sudah ditanam di dapoer_module.py)
agent = create_agent()

# Inisialisasi chat memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "👋 Hai! Mau masak apa hari ini?"})

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"⚠️ Maaf, terjadi kesalahan: {str(e)}"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
