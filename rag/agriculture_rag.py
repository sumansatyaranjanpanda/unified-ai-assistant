import os
from langchain_community.document_loaders import PyPDFDirectoryLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

def build_agriculture_vectorestore(folder_path:str,save_path:str="vectorstore/agriculture"):

    # Load PDF files from the healthcare directory
    #loader = PyPDFDirectoryLoader(pdf_folder_path)
    #documents = loader.load()

    # 1. Load all .txt files
    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, fname))
            docs.extend(loader.load())
    


    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    # Create a vector store using FAISS and OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(save_path)

    print(f"[✅] Vectorstore saved at {save_path}")

if __name__ == "__main__":
    build_agriculture_vectorestore("data/agriculture")