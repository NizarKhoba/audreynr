# dapoer_module.py
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
import random

CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return text

df_cleaned['Title_Normalized'] = df_cleaned['Title'].apply(normalize_text)
df_cleaned['Ingredients_Normalized'] = df_cleaned['Ingredients'].apply(normalize_text)
df_cleaned['Steps_Normalized'] = df_cleaned['Steps'].apply(normalize_text)

def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}\n\nBahan-bahan:\n{bahan_md}\n\nLangkah Memasak:\n{langkah_md}"""

def search_by_title(query):
    query_normalized = normalize_text(query)
    match_title = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_normalized)]
    if not match_title.empty:
        return format_recipe(match_title.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    prompt_lower = normalize_text(query)
    bahan_keywords = [w for w in prompt_lower.split() if w not in stopwords and len(w) > 1]

    if bahan_keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(lambda x: all(k in x for k in bahan_keywords))
        match_bahan = df_cleaned[mask]
        if not match_bahan.empty:
            hasil = match_bahan.head(3).apply(format_recipe, axis=1).tolist()
            return "Berikut beberapa resep yang menggunakan bahan tersebut:\n\n" + "\n\n---\n\n".join(hasil)
        else:
            return rag_search(query=query, api_key=None, fallback_only=True)
    return "Silakan sebutkan bahan utama yang ingin kamu cari resepnya."

def search_by_method(query):
    prompt_lower = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt_lower:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan dengan metode {metode}:\n- " + "\n- ".join(hasil)
    return "Metode memasak tidak ditemukan."

def recommend_easy_recipes(query):
    prompt_lower = normalize_text(query)
    if "mudah" in prompt_lower or "pemula" in prompt_lower:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi resep mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ada rekomendasi khusus untuk masakan mudah."

def build_vectorstore(api_key=None):
    docs = [
        Document(page_content=f"Judul: {row['Title']}\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}")
        for _, row in df_cleaned.iterrows()
    ]
    texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key) if api_key else None
    return FAISS.from_documents(texts, embeddings) if embeddings else texts

def rag_search(api_key=None, query="", fallback_only=False):
    if not fallback_only:
        vectorstore = build_vectorstore(api_key)
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents(query)
        if docs:
            return "\n\n".join([doc.page_content for doc in docs[:3]])
    fallback = df_cleaned.sample(3)
    return "Berikut beberapa resep pilihan untukmu:\n\n" + "\n\n---\n\n".join([
        format_recipe(row) for _, row in fallback.iterrows()
    ])

def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        system_instruction="Jawablah semua pertanyaan dalam Bahasa Indonesia secara informatif, ramah, dan gunakan data yang tersedia bila memungkinkan."
    )

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul masakan."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari resep berdasarkan bahan utama."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari resep berdasarkan metode memasak."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi masakan mudah."),
        Tool(name="RAGSearch", func=lambda q: rag_search(api_key, q), description="Cari informasi menggunakan FAISS + fallback.")
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=False
    )
