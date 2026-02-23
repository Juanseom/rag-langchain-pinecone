import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model
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
print("Model ready\n")


def ask_question(question: str) -> str: # Search → Build prompt → Get answer

    print(f"  Searching for relevant information")
    retrieved_docs = vector_store.similarity_search(question, k=4)

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs) # Combine the found documents into one text block

    print(f"  Found {len(retrieved_docs)} relevant chunks")

    messages = [ # Create the messages with context included
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions about a blog post "
                "about LLM-powered autonomous agents by Lilian Weng. "
                "Use ONLY the following context to answer the question. "
                "If you cannot find the answer in the context, say so.\n\n"
                f"Context:\n{docs_content}"
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    print(f"  Generating answer")
    response = model.invoke(messages)     # send to the AI model and get the answer

    return response.content


print()
print("RAG CHAIN")
print()


query1 = "What is task decomposition?"

print(f"\n--- Question 1 ---")
print(f"User: {query1}")
answer1 = ask_question(query1)
print(f"\nAnswer: {answer1}")


query2 = "What are the different types of memory in the agent system?"

print(f"\n\n--- Question 2 ---")
print(f"User: {query2}")
answer2 = ask_question(query2)
print(f"\nAnswer: {answer2}")


query3 = "What is Chain of Thought (CoT)?"

print(f"\n\n--- Question 3 ---")
print(f"User: {query3}")
answer3 = ask_question(query3)
print(f"\nAnswer: {answer3}")