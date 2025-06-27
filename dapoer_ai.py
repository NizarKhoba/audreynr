import streamlit as st
from dapoer_module import create_agent

st.set_page_config(page_title="Dapoer-AI", page_icon="🍲")
st.title("🍛 Dapoer-AI - Asisten Resep Masakan Indonesia")

# Input API Key
GOOGLE_API_KEY = st.text_input("🔐 Masukkan API Key Gemini Anda:", type="password")
if not GOOGLE_API_KEY:
    st.warning("⚠️ Silakan masukkan API key terlebih dahulu untuk mulai menggunakan Dapoer-AI.")
    st.stop()

# Inisialisasi agent
try:
    agent = create_agent(GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ Gagal inisialisasi agent:\n\n{e}")
    st.stop()

# Inisialisasi sesi chat
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "👋 Hai! Mau masak apa hari ini? Ketikkan nama masakan, bahan, atau metode yang kamu ingin tahu."
    })

# Tampilkan riwayat obrolan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input pengguna
if prompt := st.chat_input("Tanyakan resep, bahan, atau metode memasak..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari jawaban... 🍳"):
            try:
                response = agent.run(prompt)
            except Exception as e:
                response = f"⚠️ Terjadi kesalahan saat memproses permintaan:\n\n```\n{e}\n```"
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
