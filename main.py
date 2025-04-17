from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Prompt template
template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:
"""

# Embedding and model setup
embeddings = OllamaEmbeddings(model="llama3.2")
vector_store = InMemoryVectorStore(embeddings)
model = OllamaLLM(model="llama3.2")


# Load page content
def load_page(url):
    print(f"Loading page: {url}")
    loader = WebBaseLoader(url)
    return loader.load()


# Chunking text
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


# Index documents into the vector store
def index_docs(documents):
    vector_store.add_documents(documents)


# Retrieve relevant docs
def retrieve_docs(query):
    return vector_store.similarity_search(query)


# Generate an answer from context
def answer_question(question, context):
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    return chain.invoke({"question": question, "context": context})


# Main CLI app
if __name__ == "__main__":
    url = input("Enter URL to scrape: ").strip()
    documents = load_page(url)
    chunks = split_text(documents)
    index_docs(chunks)

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()
        if question.lower() == "exit":
            break
        results = retrieve_docs(question)
        context = "\n\n".join([doc.page_content for doc in results])
        answer = answer_question(question, context)
        print(f"\n🧠 Answer: {answer}")
