
import os
import time
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


print()
print("STEP 1: Loading the blog post from the internet")
print()


bs4_strainer = bs4.SoupStrainer(
    class_=("post-title", "post-header", "post-content") # only want the main content of the blog post
)

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)

docs = loader.load()

print(f"Loaded {len(docs)} document(s)")
print(f"Total characters: {len(docs[0].page_content)}")
print(f"\nFirst 500 characters of the blog post:")
print()
print(docs[0].page_content[:500])



print()
print("STEP 2: Splitting the document into small chunks...")
print()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      
    chunk_overlap=200,   
    add_start_index=True,  # Track where each chunk starts in the original text
)

all_splits = text_splitter.split_documents(docs)

print(f"Split the blog post into {len(all_splits)} chunks")
print(f"\nExample chunk (chunk #1):")
print()
print(all_splits[0].page_content[:300])



print()
print("STEP 3: Creating embeddings and storing in Pinecone...")
print()

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004") # converts text into numbers/vectors

# Connect to Pinecone using our API key
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "rag-langchain"

print("Calculating embedding dimension...")
sample_embedding = embeddings.embed_query("Hello world")
dimension = len(sample_embedding)
print(f"Embedding dimension: {dimension}")

existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"\nCreating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=dimension,        
        metric="cosine",            #how to measure similarity between vectors
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",    
        ),
    )
    
    while not pc.describe_index(index_name).status["ready"]:
        print("  Waiting for index to be ready...")
        time.sleep(1)
    print(f"Index '{index_name}' created!")
else:
    print(f"Index '{index_name}' already exists")

index = pc.Index(index_name)


vector_store = PineconeVectorStore(embedding=embeddings, index=index) # connects embeddings model + Pinecone index


print(f"\nStoring {len(all_splits)} chunks in Pinecone...")
document_ids = vector_store.add_documents(documents=all_splits) #converts each chunk into an embedding and stores it

print(f"Stored {len(document_ids)} chunks in Pinecone!")
print(f"\nFirst 3 document IDs: {document_ids[:3]}")



print()
print("INDEXING COMPLETE")
print()

