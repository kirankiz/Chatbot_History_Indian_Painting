# Chatbot on History of Indian Painting

## Problem statement

**1. Domain identification:** A chatbot that answers user queries based on the History of Indian Painting. It works based on the knowledge from the work of Percy Brown (Indian Painting, 1917).
Book is made available in the public domain by Central Archaeological Library and can be found in https://ignca.gov.in/Asi_data/14801.pdf

**2. Problem description:** The archives of the Archaeological Survey of India (ASI) houses lot of historical documents and books, which are open to the public. The archive is refrred by huge number of researchers around the globe. Since most of the available refernces are scanned copies or photos of old books, they cannot be analysed through key word saerch. A person should download the pdf file and have to go through each pages manually to saerch topic of interest.
This affects the speed of reasearch, as the pdf files are heavy in size with small fonts with poor clarity. Manually searhing them can be tiresome.

**3. Why this problem is unique?:** The documents available in the archives of ASI are either scanned copies or photographs of old books. They lacks clarity with small fonts, old style prints, random types of foot notes, multiple images etc. There are no text content available in the pdf and all are in the image format. Accessing and analysing these kind of documents is a key challenge faced by the research academia around the world.

**4. Why RAG is the right approach?:** Traditional saerch methodologies are not useful in the mentioned type of documents. Currently, only manual search is the only available option, which is very time consuming and tiresome. Here, RAG can be an effective solution in handling these kind of documents effectively.

**5. Expected outcomes:** The RAG chatbot would be able to answer queries related to the pdf documents that the user feeds it. It mainly aims the type of multimodal pdfs similar to the ones present in ASI archives.

**6. Problem statement:** Prepare a RAG chatbot that ingests old archived documents (scanned or photographed pdf files) and answers user query based on that.

## Architecture overview

**7. Architecture overview:**

This system implements a multimodal Retrieval‑Augmented Generation (RAG) architecture that ingests a scanned art history PDF containing texts, illustrations and tables (at index section). Text is extracted using OCR and chunked with rich metadata, while images are detected using OpenCV and converted into descriptive text using a Vision‑Language Model (LLaVA). The contents in index are considered as in table format and trated using numpy arrays. All text chunks and image summaries are embedded into a unified ChromaDB vector store. A FastAPI backend enables semantic querying, retrieving the most relevant multimodal context and optionally generating grounded responses using a large language model.

**7.1. High‑Level Components**

A. Data Sources
----------------
Scanned PDF:

  History_of_Indian_painting.pdf
  Contains OCR‑required text and embedded illustrations.

B. Ingestion & Indexing Layer
------------------------------
Handles offline preprocessing and vector indexing.

a). Text Extraction:

    PyMuPDF renders each page as an image
    
    Tesseract OCR converts images → text
    
    Output: extracted_text.txt

b). Text Chunking & Metadata:

    LangChain RecursiveCharacterTextSplitter
    
    Chunk size: ~500 chars, overlap: 100
    
    Metadata attached:
    
      chunk_id
      
      page
      
      section
      
      chunk_type = "text"

c). Image Extraction:

    PyMuPDF exports each page as a high‑res image
    
    OpenCV detects illustration contours
    
    Cropped illustrations saved in refined_illustrations/

d). Image Indexing:

    Filenames parsed for page numbers
    
    Image metadata stored in image_index.json

e). Image → Text (VLM):

    LLaVA‑1.5‑7B (4‑bit quantized)
    
    Generates art‑historical descriptions
    
    Output: image_summaries.json
    
    Metadata:
    
      chunk_id
      
      page
      
      filename
      
      chunk_type = "image_summary"

f). Corpus Merge:

    Text chunks + image summaries merged into:
    
      rag_corpus.json

g). Embedding & Vector Store:

    SentenceTransformers (all-MiniLM-L6-v2)
    
    ChromaDB persistent collection
    
    Both text & image summaries embedded uniformly

**C. Serving Layer (FastAPI)**
--------------------------------
a). Vector Retrieval:

    Semantic search over ChromaDB
    
    Top‑K retrieval with distances & metadata

b). LLM Generation:

    OpenRouter (qwen/qwen3.6-plus:free/ OpenAI‑compatible)
    
    Retrieval context injected into prompt
    
    Graceful fallback if LLM unavailable

c). API Endpoints:

    /health → system status
    
    /ingest → add new documents
    
    /query → retrieve + generate answer


**7.2. Ingestion Pipeline (Offline)**
--------------------------------------
<img width="1017" height="2268" alt="Ingsetion pipeline" src="https://github.com/user-attachments/assets/537ab89a-530c-4d93-924f-784000233039" />

**7.3. Query Pipeline (Online / Runtime)**
--------------------------------------------

<img width="1050" height="2408" alt="Query pipeline" src="https://github.com/user-attachments/assets/4db7428f-a2bd-4fa9-aa04-1638e1d8458b" />

## Technology choices

**8. Technology Choices:**
---------------------------
The RAG chatbot is developed in Google Colab platform. So the main criteria for choosing the components are better compatiblity with the colab platform. Following are the components choosen for different tasks.

a) Parser (PyMuPDF + Tesseract OCR):

PyMuPDF efficiently renders PDF pages into high‑resolution images, while Tesseract OCR reliably extracts text from scanned documents, making the pipeline robust to non‑digitally born PDFs (scanned copies or photographs).

b) Embedding Model (SentenceTransformers – all‑MiniLM‑L6‑v2):

This model provides a strong balance of semantic accuracy, speed, and low memory usage, making it well suited for embedding both text chunks and image‑generated descriptions in a unified vector space.

c) Vector Store (ChromaDB):

ChromaDB offers persistent storage, fast similarity search, and simple metadata handling, which fits well for structured RAG pipelines and local experimentation.

d) LLM (OpenRouter‑hosted Qwen- qwen/qwen3.6-plus:free):

The LLM is used only at query time to synthesize answers from retrieved context, keeping generation grounded while allowing flexible model swapping and graceful fallback if unavailable. Qwen3.6 was a freely available LLm model from Openrouter, with less parameters and better efficiency.

e) VLM (LLaVA‑1.5):

LLaVA converts visual illustrations into rich textual descriptions, enabling images to participate in standard text based retrieval without directly embedding pixels. The model is open sourced and available in Huggingface. It was selected due to the same reason. Since the appliccation is niche, the use of LLaVA model may not be well suited and may need fine tuning for better results.


## Setup Instructions

**9. Setup Instructions**
--------------------------

a) Clone the Repository

    git clone https://github.com/<your-username>/<repo-name>.git

    cd <repo-name>

b) Create and Activate a Virtual Environment (Optional but Recommended)

    python -m venv venv

    source venv/bin/activate        # Linux / macOS

    venv\Scripts\activate           # Windows

c) Install Dependencies

    pip install -r requirements.txt

d) Configure Environment Variables

Create a .env file using the provided template:

    cp .env.example .env

Update the file with required API keys (e.g., OpenRouter, Hugging Face, ngrok if needed).

e) Run Data Ingestion & Indexing

Execute the preprocessing notebooks in order:

PDF text extraction (OCR)

Text chunking and metadata creation

Image extraction and VLM summarization

Corpus merging and embedding generation

This builds the ChromaDB vector store locally.

f) Start the FastAPI Server

    uvicorn app.main:app --host 0.0.0.0 --port 8000

g) Test the Application

Open Swagger UI:

    http://127.0.0.1:8000/docs

Or query via Python

    requests.post(
    
      "http://127.0.0.1:8000/query",  
      
      json={"query": "Explain Mughal miniature painting"}  
      
    )

The system is now ready for multimodal RAG queries.

## API Documentation

**10. API Documentation**
-------------------------

The system exposes a RESTful API built using FastAPI to support health monitoring, document ingestion, and semantic querying over the multimodal RAG corpus.

Base URL (local):

    http://127.0.0.1:8000

a) Health Check API

GET /health

Description:
Checks whether the FastAPI server, vector database, and LLM integration are running correctly. Also returns basic runtime statistics.

Sample Request:

  GET /health

Sample Response:

  {
    "status": "ok",
    "message": "RAG API is running",
    "llm_ready": true,
    "indexed_documents": 594,
    "uptime": "00:12:37"
  }

b) Document Ingestion API

POST /ingest

Description:
Dynamically ingests a new document chunk into the ChromaDB vector store. This endpoint supports incremental updates without rebuilding the entire index.

Request Body:

    {
      "id": "custom_doc_001",
      "content": "Indian miniature painting flourished under Mughal patronage...",
      "metadata": {
        "source": "Manual Ingest",
        "page": 120,
        "section": "Mughal Painting",
        "chunk_type": "text"
      }
    }

Sample Response:

    {
      "status": "success",
      "message": "Document with ID custom_doc_001 ingested.",
      "id": "custom_doc_001"
    }

Error Responses:

  503 – Vector database not initialized
  500 – Error during ingestion

C) Query API (Core RAG Endpoint)

POST /query

Description:
Accepts a natural‑language query, retrieves the most relevant text and image‑summary chunks from ChromaDB, and optionally generates a grounded answer using an LLM.

Request Body:

    {
      "query": "Explain Mughal miniature painting",
      "n_results": 5
    }

Successful Response (LLM enabled):

    {
      "query": "Explain Mughal miniature painting",
      "answer": "Mughal miniature painting developed under imperial patronage, blending Persian techniques with Indian themes...",
      "sources": [
        {
          "content": "Mughal miniature painting reached its height during the reign of Jahangir...",
          "metadata": {
            "page": 66,
            "section": "Mughal School",
            "chunk_type": "text"
          },
          "distance": 0.21
        },
        {
          "content": "The image depicts a courtly Mughal scene with fine brushwork and rich pigments...",
          "metadata": {
            "page": 71,
            "section": "Image Description",
            "chunk_type": "image_summary"
          },
          "distance": 0.29
        }
      ]
    }

Fallback Response (LLM not available):

    {
      "query": "Explain Mughal miniature painting",
      "answer": "LLM not initialized. Showing raw retrieval results.",
      "sources": [
        {
          "content": "Mughal miniature painting reached its height...",
          "metadata": {
            "page": 66,
            "chunk_type": "text"
          },
          "distance": 0.21
        }
      ]
    }

Notes

All responses are returned in JSON format.

Image content is represented indirectly through VLM‑generated textual summaries, not raw images.

API can be explored interactively via Swagger UI at:

  http://127.0.0.1:8000/docs


**11.	Screenshots**

Refer the screenshot folder.

**12.	Limitations & Future Work**

For the specific topic selected, VLM is not properly giving description for the images. For example, it cannot identify the mughal painting or a cave painting properly and describe the school, painting style etc. We need Fine-tuned VLM model for this purposes.

Due to access restrictions in office systems, I used Google colab for the entire chatbot development. This demanded use of LLM model through API call. This could've been avoided if an open-sourced LLM is installed in the system. This would've supported offline working also.

Major issues were faced due to version difference of different libraries. This demanded frequest re-installation of libraries and runtime restart. This severly affected code robustness and development time.

Due to frequent runtime disconnect and GPU unavailablility, it was very difficult to complete the development and printing output.

Swagger UI is not used and the FastAPI endpoints are ran directly on the colab interface. This was done due to the firewall and security systems in the office laptop. Tried to get it done using nGrok API, but failed to do so due to the above and version change issues. 


