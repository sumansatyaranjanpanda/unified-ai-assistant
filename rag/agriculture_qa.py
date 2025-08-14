import os
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(
    model_name="gpt-3.5-turbo")

# Compute project root (one level up from this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VECTORSTORE_PATH = os.path.join(ROOT_DIR, "vectorstore", "agriculture")
EMBEDDING_MODEL = OpenAIEmbeddings()

# Load or recreate FAISS vectorstore
def load_vectorstore(path: str = VECTORSTORE_PATH):
    return FAISS.load_local(path, EMBEDDING_MODEL,allow_dangerous_deserialization=True)


def get_agriculture_qa_chain():

    vectorstore = load_vectorstore()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

def ask_agriculture(question: str)-> str:
    qa_chain = get_agriculture_qa_chain()
    response = qa_chain.invoke(question)
    return response