import os
import glob
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# routes
DATA_PATH = "data"
DB_PATH = "chroma_db"

#embedding
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "mps"},  #for mac cpus : "mps" intel cpus : "cpu"
    encode_kwargs={"normalize_embeddings": True}
)

#doc loader
def load_documents():
    docs = []
    # find all files 
    file_paths = glob.glob(os.path.join(DATA_PATH, "**/*"), recursive=True)
    
    for file_path in file_paths:
        if os.path.isfile(file_path):
            try:
                if file_path.lower().endswith('.pdf'):
                    loader = PyPDFLoader(file_path)
                else:
                    # txt, md, docx, etc → TextLoader
                    loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load())
                print(f"{os.path.basename(file_path)} loaded successfully")
            except Exception as e:
                print(f"{os.path.basename(file_path)}: {e} error in loading")
                continue
    
    return docs

#create or load db
if not os.path.exists(DB_PATH):
    print("loading datas from data/")
    docs = load_documents()

    if not docs:
        print("cant find anything in data/ like: testtext.txt")
        exit()

    print(f"documents loaded counts: {len(docs)}")

    print("chunking ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"chunks count : {len(chunks)}")

    print("creating vector db")
    vectorstore = Chroma.from_documents(chunks, embedding, persist_directory=DB_PATH)
    print("db created successfully")
else:
    print("loading founded db")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embedding)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

#llm 
llm = ChatOllama(model="llama3.2:latest", temperature=0.1)

#prompt
template = """شما یک دستیار فارسی دقیق هستید. فقط از منابع زیر استفاده کنید.
اگر اطلاعات کافی نبود، بگویید "اطلاعات کافی در منابع موجود نیست."

منابع:
{context}

سوال: {question}

پاسخ کامل و مفید به فارسی:"""

prompt = PromptTemplate.from_template(template)

#RAG chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# chating interface
print("\n" + "═" * 70)
print("rag chatbot is ready for U ...")
print("═" * 70)
print("ask your question :\n")

while True:
    q = input("your question ").strip()
    if q.lower() in ["exit", "quit"]:
        print("bye bye!")
        break
    print("thinking ...")
    answer = chain.invoke(q)
    print(f"\n answer : {answer}\n")
    print("─" * 70)