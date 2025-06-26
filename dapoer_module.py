def build_vectorstore(api_key):
    import os
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores import FAISS

    os.environ["GOOGLE_API_KEY"] = api_key

    splitter = CharacterTextSplitter(chunk_size=256, chunk_overlap=32)

    docs = []
    for _, row in df_cleaned.iterrows():
        content = f"{row['Title']}\n\nBahan:\n{row['Ingredients']}\n\nLangkah:\n{row['Steps']}"
        docs.append(Document(page_content=content))

    texts = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        google_api_key=api_key,
        model="models/embedding-001",
        task_type="retrieval_document"
    )

    return FAISS.from_documents(texts, embeddings)

def rag_search(api_key, query):
    vectorstore = build_vectorstore(api_key)
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        fallback_samples = df_cleaned.sample(5)
        fallback_response = "\n\n".join([
            f"{row['Title']}:\nBahan: {row['Ingredients']}\nLangkah: {row['Steps']}"
            for _, row in fallback_samples.iterrows()
        ])
        return f"Tidak ditemukan informasi yang relevan. Berikut beberapa rekomendasi masakan acak:\n\n{fallback_response}"
    return "\n\n".join([doc.page_content for doc in docs[:5]])

def create_agent(api_key):
    import os
    os.environ["GOOGLE_API_KEY"] = api_key

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=api_key
    )

    def rag_tool_func(query):
        return rag_search(api_key, query)

    tools = [
        Tool(name="SearchByTitle", func=search_by_title, description="Cari resep berdasarkan judul masakan."),
        Tool(name="SearchByIngredients", func=search_by_ingredients, description="Cari masakan berdasarkan bahan."),
        Tool(name="SearchByMethod", func=search_by_method, description="Cari masakan berdasarkan metode memasak."),
        Tool(name="RecommendEasyRecipes", func=recommend_easy_recipes, description="Rekomendasi masakan yang mudah dibuat."),
        Tool(name="RAGSearch", func=rag_tool_func, description="Cari informasi masakan menggunakan FAISS dan RAG.")
    ]

    memory = ConversationBufferMemory(memory_key="chat_history")

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        memory=memory,
        verbose=False
    )
