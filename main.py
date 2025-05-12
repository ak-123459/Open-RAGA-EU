from fastapi import FastAPI
from app.assistant import ChatManagerRunnable,ChatManager
from llms.model_manager import ModelManager
from retrieval.vector_store import get_vector_store
from app.models.schemas import ChatInput,ChatOutput
from fastapi import HTTPException
from typing import Optional
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
import traceback
import uvicorn
import time
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("uvicorn")


# load .env files
load_dotenv()

# Initialise fast api app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (use specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

#Check Nvidia Key is available


if( os.getenv('NVIDIA_CLOUD_MODEL_API_KEY')
):
  Nvidia_key = os.getenv('NVIDIA_CLOUD_MODEL_API_KEY')
    
else:
     print("NVIDIA Key not avaliable..")





# Create Instance of Model manger
model_manager = ModelManager()

start_time = time.perf_counter()  # Start timer

# Get LLM instance
llm = model_manager.get_model(name='google/gemma-2-9b-it', provider='nvidia',
                              key=os.getenv('NVIDIA_CLOUD_MODEL_API_KEY'))


elapsed = (time.perf_counter() - start_time) * 1000

logger.info(f"⚡ Latency (llm initialization): {elapsed:.2f} ms")

start_time = time.perf_counter()  # Start timer

vector_db = get_vector_store()

elapsed = (time.perf_counter() - start_time) * 1000
logger.info(f"⚡ Latency (vector initialization): {elapsed:.2f} ms")

# Create instance of chat manger
manager = ChatManager(llm, vector_db.as_retriever())



# this is for api health check up
@app.get("/")
def health_check():

    return {'health':'ok'}



# this  startup will take care of model loading and vector database availability on app start up
@app.on_event("startup")
async def on_startup():
    # check model
    if llm is None:

        logger.error("❌ Error model not initialized: %s", traceback.format_exc())

        raise RuntimeError("Model not initialized.")


    # check vector db
    try:

        _ = vector_db.as_retriever().invoke("startup test")

    except Exception as e:

        logger.error("❌ Error vector database not available: %s", traceback.format_exc())
        raise RuntimeError(f"Vector DB error: {e}")





# Endpoint to get the response from the chat model
@app.post("/chat",response_model=ChatOutput)
async def get_response(request: ChatInput):


    print("request:-----------",request)
    query = request.query
    last_3_turn = request.last_3_turn

    try:

        logger.info(f"📥 Received query: {request.query}")

        logger.info(f"📚 Context: {request.last_3_turn}")

        start_time = time.perf_counter()  # Start timer

        response = await manager.run(query, last_3_turn)

        elapsed = (time.perf_counter() - start_time) * 1000

        logger.info(f"⚡ Latency (model response): {elapsed:.2f} ms")

        logger.info(f"📤 Returning response: {response}")

        return {'response':response}


    except Exception as e:

        logger.error("❌ Error processing request: %s", traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))





if __name__=="__main__":


  uvicorn.run(app,host="0.0.0.0", port=8000, reload=True)
