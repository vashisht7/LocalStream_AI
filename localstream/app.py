import os
import logging
import torch
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM

from localstream.config import settings
from localstream.transcribe import run_transcription
from localstream.database import VectorDBManager

# Set up logging for the web application
logger = logging.getLogger("localstream.app")

# Initialize Vector DB Manager
db_manager = VectorDBManager()

# Initialize Local LLM Model and Tokenizer
model_name = settings.local_llm_model_name
device = settings.compute_device
# On CPU, load using float32. On CUDA, we can use float16.
dtype = torch.float16 if device == "cuda" else torch.float32

logger.info(f"Loading local LLM '{model_name}' on device '{device}'...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)
    logger.info("Local LLM model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load local LLM model '{model_name}': {e}")
    raise RuntimeError(f"Local LLM initialization failed: {e}") from e

# Create FastAPI app
app = FastAPI(
    title="LocalStream RAG Engine",
    description="A 100% self-hosted, local multimedia semantic search and RAG service.",
    version="2.0.0"
)

# Request / Response Schemas
class IngestRequest(BaseModel):
    file_path: str = Field(
        ...,
        description="Absolute path to the video or audio file on disk to ingest."
    )

class QueryRequest(BaseModel):
    prompt: str = Field(
        ...,
        description="The semantic search query or question about the multimedia content."
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="The maximum number of relevant transcript chunks to retrieve from the vector database."
    )

class QueryResponse(BaseModel):
    query: str
    answer: str
    references: list[dict]

# Background Worker
def background_ingest_worker(file_path: str):
    """
    Executes transcription and indexing operations sequentially.
    Runs asynchronously in a background thread to prevent blocking client requests.
    """
    logger.info(f"Background worker started for file: {file_path}")
    try:
        # Step 1: Extract audio (if video) and run Whisper transcription
        chunks = run_transcription(file_path)
        
        # Step 2: Ensure the collection/schema exists in Qdrant
        db_manager.initialize_schema()
        
        # Step 3: Embed chunks and index them in the database
        db_manager.index_media_content(file_path, chunks)
        
        logger.info(f"Background worker finished successfully for file: {file_path}")
    except Exception as e:
        logger.error(f"Background worker failed for file: {file_path}. Error: {e}")

# API Endpoints

@app.get("/", response_class=HTMLResponse)
def serve_homepage():
    """Serves the glassmorphic interactive RAG web UI at the root path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    html_file_path = os.path.join(current_dir, "index.html")
    
    if not os.path.exists(html_file_path):
        logger.error(f"HTML frontend file missing at: {html_file_path}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="index.html file not found in the package."
        )
        
    try:
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error reading frontend file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load user interface: {e}"
        )

@app.get("/health")
def health():
    """Simple service health and compute configuration check."""
    return {
        "status": "healthy",
        "compute_device": settings.compute_device,
        "collection_name": settings.collection_name,
        "embedding_model": settings.embedding_model_name,
        "whisper_model": settings.whisper_model_size,
        "local_llm_model": settings.local_llm_model_name
    }

@app.post("/api/ingest", status_code=status.HTTP_202_ACCEPTED)
def ingest_media(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Triggers transcription and vector indexing of a media file.
    Validates file existence and delegates work to a background task.
    """
    file_path = request.file_path
    
    # Validate file existence on disk
    if not os.path.exists(file_path):
        logger.error(f"Ingestion failed. File does not exist: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File not found at specified path: {file_path}"
        )
        
    if not os.path.isfile(file_path):
        logger.error(f"Ingestion failed. Path is not a file: {file_path}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Specified path is a directory, not a file: {file_path}"
        )

    # Add ingestion worker to FastAPI background tasks
    background_tasks.add_task(background_ingest_worker, file_path)
    
    logger.info(f"Ingestion request accepted for: {file_path}")
    return {
        "status": "accepted",
        "message": "Ingestion task queued in the background.",
        "file_path": file_path
    }

@app.post("/api/query", response_model=QueryResponse)
def query_media(request: QueryRequest):
    """
    RAG Query endpoint. Retrieves relevant video chunks using dense embeddings,
    compiles a structured context prompt, and synthesizes an answer using the local LLM.
    """
    # 1. Encode incoming text string query
    try:
        query_vector = db_manager.model.encode(request.prompt).tolist()
    except Exception as e:
        logger.error(f"Error encoding query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate query embeddings: {e}"
        )

    # 2. Execute similarity search in Qdrant
    try:
        search_results = db_manager.client.search(
            collection_name=settings.collection_name,
            query_vector=query_vector,
            limit=request.limit
        )
    except Exception as e:
        logger.error(f"Qdrant search query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector search failed. Ensure the collection has been initialized and ingested: {e}"
        )

    # 3. Process retrieved hits and format context blocks
    context_blocks = []
    references = []
    
    for idx, hit in enumerate(search_results):
        payload = hit.payload
        block = (
            f"Source File: {payload.get('filename')}\n"
            f"Timeline: {payload.get('start_timestamp')} to {payload.get('end_timestamp')}\n"
            f"Transcript: {payload.get('text')}\n"
        )
        context_blocks.append(block)
        
        # Save reference metadata to return to client
        references.append({
            "filename": payload.get("filename"),
            "file_path": payload.get("file_path"),
            "start_timestamp": payload.get("start_timestamp"),
            "end_timestamp": payload.get("end_timestamp"),
            "start_seconds": payload.get("start_seconds"),
            "text": payload.get("text"),
            "score": hit.score
        })

    # If no results found, return an early response
    if not context_blocks:
        return QueryResponse(
            query=request.prompt,
            answer="No relevant transcription chunks found in the database. Please ingest some files first.",
            references=[]
        )

    # 4. Construct prompts and execute local LLM inference
    context_str = "\n".join(context_blocks)
    
    system_instruction = (
        "You are a helpful local multimedia assistant. "
        "Your task is to synthesize an answer to the User Query based ONLY on the retrieved Context Blocks. "
        "Follow these rules strictly:\n"
        "1. Do not use any external knowledge. If the answer cannot be found in the context blocks, state clearly that you do not know.\n"
        "2. For every fact or statement you present in the response, you MUST cite the source filename and exact start timestamp in brackets at the end of the sentence or clause.\n"
        "3. Citation Format: [Filename.ext (HH:MM:SS)] (e.g. [demo_video.mp4 (00:01:23)]).\n"
        "4. Keep your answer factual, precise, and concise."
    )
    
    prompt_content = (
        f"User Query: {request.prompt}\n\n"
        f"Retrieved Context Blocks:\n"
        f"{context_str}\n\n"
        f"Please write a response summarizing the retrieved information to answer the user query."
    )

    # Format the prompt using Qwen's chat template
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": prompt_content}
    ]
    
    try:
        # Convert messages format to chat token structure
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize and run local generation
        inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract new tokens (remove prompt prefix)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
    except Exception as e:
        logger.error(f"Local LLM text generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Local text generation failed: {e}"
        )

    return QueryResponse(
        query=request.prompt,
        answer=answer,
        references=references
    )
