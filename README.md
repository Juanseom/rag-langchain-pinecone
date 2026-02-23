# RAG with LangChain and Pinecone

This project shows how to build a **RAG (Retrieval-Augmented Generation)** system using **LangChain**, **Pinecone**, and **Google Gemini**.

This project follows the official [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/).

---

## Table of Contents
- [What is this project?](#what-is-this-project)
- [Project Architecture](#project-architecture)
- [What is RAG?](#what-is-rag)
- [How RAG Works](#how-rag-works)
- [Components Explained](#components-explained)
- [Installation Instructions](#installation-instructions)
- [How to Run the Code](#how-to-run-the-code)
- [Example Output](#example-output)
- [Differences Between Scripts](#differences-between-rag_agentpy-and-rag_chainpy)

---

## What is this project?

This project has three Python scripts that show how to build a RAG system step by step:

| File | What it does |
|------|-------------|
| `index_documents.py` | Loads a blog post, splits it into chunks, and stores them in Pinecone |
| `rag_agent.py` | An AI agent that decides when to search Pinecone to answer questions |
| `rag_chain.py` | A simple chain that always searches before answering (faster) |

The blog post used is: [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.

---

## Project Architecture

### File Structure

```
rag-langchain-pinecone/
│
├── index_documents.py     # Load, split, and store documents in Pinecone
├── rag_agent.py           # RAG agent approach (AI decides when to search)
├── rag_chain.py           # RAG chain approach (always searches first)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in Git)
├── .env.example           # Template for .env file
├── .gitignore             # Files to ignore in Git
└── README.md              # This file
```

### Architecture Flow

#### Step 1: Indexing (run once with `index_documents.py`)

```
Blog Post (URL)
     ↓
[ WebBaseLoader ] ← Downloads the web page
     ↓
[ BeautifulSoup ] ← Filters only the content we need
     ↓
[ Text Splitter ] ← Splits text into small chunks (~1000 characters)
     ↓
[ Embeddings Model ] ← Converts text chunks into numbers (vectors)
     ↓
[ Pinecone ] ← Stores the vectors for searching later
```

#### Step 2a: Querying with Agent (`rag_agent.py`)

```
User Question
     ↓
[ Agent ] ← AI decides what to do
     ↓
Decision: Need to search?
     ↓                    ↓
   YES                   NO
     ↓                    ↓
[ Search Pinecone ]   Direct Answer
     ↓
Get Relevant Chunks
     ↓
Generate Answer with Context
     ↓
Return to User
```

#### Step 2b: Querying with Chain (`rag_chain.py`)

```
User Question
     ↓
[ Search Pinecone ] ← Always searches (no decision needed)
     ↓
Get Relevant Chunks
     ↓
[ AI Model ] ← Question + Context → Answer
     ↓
Return to User
```

---

## What is RAG?

**RAG** stands for **Retrieval-Augmented Generation**. It is a technique that makes AI answers **better** by giving the AI **extra information** from an external source.

### The Problem

AI models like ChatGPT or Gemini have a **knowledge limit**:
- They only know what they learned during training
- They don't know about YOUR documents, files, or specific data
- They can give wrong answers when they don't have the right information

### The Solution: RAG

RAG fixes this problem in 3 steps:

1. **Retrieve**: Search for relevant information from your data
2. **Augment**: Add that information to the AI's prompt
3. **Generate**: The AI generates an answer using both the question AND the found information

### Simple Example

```
WITHOUT RAG:
  User: "What does the blog say about task decomposition?"
  AI: "I don't have access to that specific blog post." 

WITH RAG:
  User: "What does the blog say about task decomposition?"
  System: [Searches Pinecone → Finds relevant text from the blog]
  AI: "According to the blog, task decomposition is the process of
       breaking down a complex task into smaller steps..." 
```

---

## How RAG Works

### Phase 1: Indexing (Preparing the Data)

Before we can search, we need to prepare the data. This happens only **once**:

1. **Load**: Download the blog post from the internet using `WebBaseLoader`
2. **Split**: Break the long text into small chunks (~1000 characters each) using `RecursiveCharacterTextSplitter`
3. **Embed**: Convert each chunk into a vector (a list of numbers) using Google's embedding model
4. **Store**: Save all vectors in Pinecone so we can search them later

### Phase 2: Retrieval and Generation (Answering Questions)

Every time a user asks a question:

1. **Embed the question**: Convert the user's question into a vector
2. **Search**: Find the most similar vectors in Pinecone (these are the most relevant chunks)
3. **Build prompt**: Add the found chunks as context to the AI's prompt
4. **Generate**: The AI reads the context + question and generates an answer

---

## Components Explained

### 1. Document Loader (WebBaseLoader)

Loads text from a web page. We use **BeautifulSoup** to filter only the blog content (no sidebars, footers, etc.).

```python
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4.SoupStrainer(class_=("post-content", "post-title"))},
)
docs = loader.load()
```

### 2. Text Splitter

Splits a long document into small chunks. This is necessary because:
- AI models have a **limited context window** (they can't read too much text at once)
- Smaller chunks are **easier to search** through

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Max 1000 characters per chunk
    chunk_overlap=200,   # 200 characters overlap between chunks
)
all_splits = text_splitter.split_documents(docs)
```

### 3. Embeddings Model (Google Gemini)

Converts text into numbers (vectors). Similar texts get similar numbers, which makes searching possible.

```python
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
```

### 4. Vector Store (Pinecone)

A cloud database that stores vectors and can search for similar ones. When you ask a question, Pinecone finds the chunks most related to your question.

```python
vector_store = PineconeVectorStore(embedding=embeddings, index=index)
vector_store.add_documents(documents=all_splits)
```

### 5. AI Model (Google Gemini)

The language model that generates answers. We use Gemini because it is **free**.

```python
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
```

### 6. Retrieval Tool (for the Agent approach)

A function that the AI agent can call to search for information. The agent reads the function description and decides when to use it.

```python
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
```

---

## Installation Instructions

### Step 1: Prerequisites

- Python 3.10 or higher

### Step 2: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated.

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `langchain` - Main framework
- `langchain-google-genai` - Google Gemini integration
- `langchain-pinecone` - Pinecone integration
- `langchain-community` - Document loaders (WebBaseLoader)
- `langchain-text-splitters` - Text splitting tools
- `langgraph` - For creating agents
- `beautifulsoup4` - For parsing HTML from web pages
- `python-dotenv` - For loading API keys from .env file

### Step 4: Get Your API Keys

**Google API Key (free):**
1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **"Create API Key"**
3. Copy the key

**Pinecone API Key (free tier):**
1. Go to [Pinecone](https://app.pinecone.io/)
2. Create a free account
3. Go to **API Keys** in the dashboard
4. Copy the key

### Step 5: Configure the .env File

1. Open the `.env` file in the project folder

2. Paste your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

3. Save the file

---

## How to Run the Code

### IMPORTANT: Run `index_documents.py` FIRST

You must index the documents **before** you can ask questions. This only needs to be done **once**.

```bash
python index_documents.py
```

**What happens:**
1. Downloads the blog post from the internet
2. Splits it into ~66 small chunks
3. Converts each chunk into an embedding (vector)
4. Stores all vectors in Pinecone

### Run the RAG Agent

```bash
python rag_agent.py
```

**What happens:**
1. Connects to Pinecone
2. Creates an AI agent with a search tool
3. The agent receives a question
4. The agent **decides** to use the search tool
5. It searches Pinecone for relevant chunks
6. It generates an answer using the found context

### Run the RAG Chain

```bash
python rag_chain.py
```

**What happens:**
1. Connects to Pinecone
2. Receives a question
3. **Always** searches Pinecone first (no decision needed)
4. Creates a prompt with the question + found context
5. Sends everything to the AI model
6. Returns the answer

---

## Example Output

### index_documents.py

```
============================================================
STEP 1: Loading the blog post from the internet...
============================================================
Loaded 1 document(s)
Total characters: 43131

============================================================
STEP 2: Splitting the document into small chunks...
============================================================
Split the blog post into 66 chunks

============================================================
STEP 3: Creating embeddings and storing in Pinecone...
============================================================
Embedding dimension: 768
Creating Pinecone index 'rag-langchain'...
Index 'rag-langchain' created!

Storing 66 chunks in Pinecone...
Stored 66 chunks in Pinecone!

============================================================
INDEXING COMPLETE!
============================================================
```

### rag_agent.py

```
--- Question 1 ---
User: What is task decomposition?

================================== Ai Message ==================================
Tool Calls:
  retrieve_context (call_abc123)
   Call ID: call_abc123
    Args:
      query: task decomposition
================================= Tool Message =================================
Name: retrieve_context

Source: {'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}
Content: Task decomposition can be done by LLM with simple prompting...

================================== Ai Message ==================================

Task decomposition is the process of breaking down a large, complex task
into smaller, more manageable steps. According to the blog post, this can
be achieved through several methods including Chain of Thought (CoT) prompting...
```

### rag_chain.py

```
--- Question 1 ---
User: What is task decomposition?
  Searching for relevant information...
  Found 4 relevant chunks
  Generating answer...

Answer: Task decomposition is the process of breaking down a large,
complex task into smaller and simpler steps. This makes it easier for
an AI agent to handle tasks step by step. Common methods include Chain
of Thought (CoT) and Tree of Thoughts...
```

---

## Differences Between rag_agent.py and rag_chain.py

| Feature | rag_agent.py | rag_chain.py |
|---------|-------------|-------------|
| Search decision | AI decides when to search | Always searches first |
| Number of AI calls | 2+ per question | 1 per question |
| Speed | Slower (multiple calls) | Faster (single call) |
| Flexibility | More flexible | Less flexible |
| Multi-step questions | Yes (can search multiple times) | No (searches once) |
| Best for | Complex questions | Simple questions |

---