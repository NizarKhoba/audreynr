import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# Load data resep
CSV_FILE_PATH = 'https://raw.githubusercontent.com/audreeynr/dapoer-ai/refs/heads/main/data/Indonesian_Food_Recipes.csv'
df = pd.read_csv(CSV_FILE_PATH)
df_cleaned = df.dropna(subset=['Title', 'Ingredients', 'Steps']).drop_duplicates()

# Normalisasi teks
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

# Format output
def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}\n\nBahan-bahan:\n{bahan_md}\n\nLangkah Memasak:\n{langkah_md}"""

# Tool 1: Berdasarkan judul
def search_by_title(query):
    query_normalized = normalize_text(query)
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_normalized)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

# Tool 2: Berdasarkan bahan
def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    words = [w for w in normalize_text(query).split() if w not in stopwords and len(w) > 2]
    if words:
        mask = df_cleaned['Ingredients_Normalized'].apply(lambda x: all(w in x for w in words))
        match = df_cleaned[mask]
        if not match.empty:
            hasil = match.head(5).apply(format_recipe, axis=1).tolist()
            return "Berikut beberapa resep yang menggunakan bahan tersebut:\n\n" + "\n\n---\n\n".join(hasil)
        return f"Tidak ditemukan resep dengan bahan: {', '.join(words)}"
    return "Silakan sebutkan bahan utama masakan."

# Tool 3: Berdasarkan metode masak
def search_by_method(query):
    prompt = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt:
            match = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not match.empty:
                hasil = match.head(5)['Title'].tolist()
                return f"Masakan dengan metode {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan metode memasak yang cocok."

# Tool 4: Rekomendasi mudah
def recommend_easy_recipes(query):
    prompt = normalize_text(query)
    if "mudah" in prompt or "pemula" in prompt:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan mudah yang cocok."

# Tool 5: RAG FAISS
def build_vectorstore(api_key):
    docs = [
        Document(page_content=f"Title: {row['Title']}\nIngredients: {row['Ingredients']}\nSteps: {row['Steps']}")
        for _, row in df_cleaned.iterrows()
    ]
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key)
    return FAISS.from_documents(texts, embeddings)

def rag_search(api_key, query):
    vs = build_vectorstore(api_key)
    docs = vs.as_retriever().get_relevant_documents(query)
    if not docs:
        fallback = df_cleaned.sample(5)
        return "Tidak ditemukan info relevan. Coba beberapa resep ini:\n\n" + "\n\n".join(
            [f"{r['Title']}:\nBahan: {r['Ingredients']}\nLangkah: {r['Steps']}" for _, r in fallback.iterrows()])
    return "\n\n".join([doc.page_content for doc in docs[:5]])

# ✅ Agent Setup
def create_agent(api_key):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.7,
        system_instruction="Jawablah dalam Bahasa Indonesia dengan ramah dan jelas."
    )

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari resep dari bahan."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari resep dari metode memasak."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi masakan mudah."),
        Tool(name="RAGSearch", func=lambda q: rag_search(api_key, q), description="Cari dari semua info dengan RAG."),
    ]

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=ConversationBufferMemory(memory_key="chat_history"),
        verbose=False
    )
