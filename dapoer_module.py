# dapoer_module.py
import pandas as pd
import re
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory

# Sistem prompt
SYSTEM_PREFIX = (
    "Tolong jawab dalam Bahasa Indonesia. "
    "Kamu adalah asisten dapur yang ramah dan membantu pengguna mencari resep masakan khas Indonesia.\n"
)

# Load dataset
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

# Format resep
def format_recipe(row):
    bahan_raw = re.split(r'\n|--|,', row['Ingredients'])
    bahan_list = [b.strip().capitalize() for b in bahan_raw if b.strip()]
    bahan_md = "\n".join([f"- {b}" for b in bahan_list])
    langkah_md = row['Steps'].strip()
    return f"""🍽 {row['Title']}\n\nBahan-bahan:\n{bahan_md}\n\nLangkah Memasak:\n{langkah_md}"""

# Tool 1
def search_by_title(query):
    query_normalized = normalize_text(query)
    match = df_cleaned[df_cleaned['Title_Normalized'].str.contains(query_normalized)]
    if not match.empty:
        return format_recipe(match.iloc[0])
    return "Resep tidak ditemukan berdasarkan judul."

# Tool 2
def search_by_ingredients(query):
    stopwords = {"masakan", "apa", "saja", "yang", "bisa", "dibuat", "dari", "menggunakan", "bahan", "resep"}
    prompt = normalize_text(query)
    keywords = [w for w in prompt.split() if w not in stopwords and len(w) > 2]

    if keywords:
        mask = df_cleaned['Ingredients_Normalized'].apply(lambda x: all(k in x for k in keywords))
        match = df_cleaned[mask]
        if not match.empty:
            hasil = match.head(5)['Title'].tolist()
            return "Masakan yang menggunakan bahan tersebut:\n- " + "\n- ".join(hasil)

    fallback = df_cleaned.sample(3)['Title'].tolist()
    return (
        "Tidak ditemukan masakan yang cocok. Berikut alternatif:\n- " +
        "\n- ".join(fallback)
    )

# Tool 3
def search_by_method(query):
    prompt = normalize_text(query)
    for metode in ['goreng', 'panggang', 'rebus', 'kukus']:
        if metode in prompt:
            cocok = df_cleaned[df_cleaned['Steps_Normalized'].str.contains(metode)]
            if not cocok.empty:
                hasil = cocok.head(5)['Title'].tolist()
                return f"Masakan yang dimasak dengan cara {metode}:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan metode memasak yang cocok."

# Tool 4
def recommend_easy_recipes(query):
    prompt = normalize_text(query)
    if "mudah" in prompt or "pemula" in prompt:
        hasil = df_cleaned[df_cleaned['Steps'].str.len() < 300].head(5)['Title'].tolist()
        return "Rekomendasi masakan mudah:\n- " + "\n- ".join(hasil)
    return "Tidak ditemukan masakan mudah yang relevan."

# Tool 5: RAG dengan FAISS
def build_vectorstore(api_key):
    docs = [
        Document(page_content=f"Title: {row['Title']}\nIngredients: {row['Ingredients']}\nSteps: {row['Steps']}")
        for _, row in df_cleaned.iterrows()
    ]
    texts = CharacterTextSplitter(chunk_size=300, chunk_overlap=30).split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key)
    return FAISS.from_documents(texts, embeddings)

def rag_search(api_key, query):
    vectorstore = build_vectorstore(api_key)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_rel_
