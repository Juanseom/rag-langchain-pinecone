import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from pinecone import Pinecone

load_dotenv()


print("Connecting to Pinecone")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("rag-langchain")

vector_store = PineconeVectorStore(embedding=embeddings, index=index)
print("Connected to Pinecone")


print("Setting up Google Gemini model")
model = init_chat_model(
    "gemini-2.5-flash",            
    model_provider="google_genai",  
)
print("Model ready")


@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information related to a query from the blog post about AI agents."""
    
    retrieved_docs = vector_store.similarity_search(query, k=2) # Search for the 2 most relevant chunks in Pinecone

   
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"  # Format the results as a readable string
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
print("Retrieval tool created")


tools = [retrieve_context]

system_prompt = (
    "You are a helpful assistant that answers questions about a blog post "
    "about LLM-powered autonomous agents by Lilian Weng. "
    "You have access to a tool that retrieves relevant information from the blog post. "
    "Always use the tool to search for information before answering."
)

agent = create_agent(model, tools, system_prompt=system_prompt)
print("Agent created\n")


print()
print("RAG AGENT")
print()


query1 = "What is task decomposition?"

print(f"\n QUESTION 1 ")
print(f"User: {query1}\n")

for step in agent.stream(
    {"messages": [{"role": "user", "content": query1}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print() # print each step so we can see what it does


query2 = (
    "What is the standard method for Task Decomposition? "
    "Once you get the answer, look up common extensions of that method." # requires the agent to search TWICE
)

print(f"\n\n QUESTION 2")
print(f"User: {query2}\n")

for step in agent.stream(
    {"messages": [{"role": "user", "content": query2}]},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
