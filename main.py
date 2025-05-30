from fastapi import FastAPI
from app.assistant import ChatManager
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
import argparse
import json
from src.llm.llm_factory import CHATLLMFactory
from src.embedder.embedder_factory import EMBFactory
from src.vector_database.vector_db_factory import VECTORDBFactory
import os



# load .env files
load_dotenv()


# Add logging for uvicorn
logger = logging.getLogger("uvicorn")


# Nvidia api key
NVIDIA_NVC_API_KEY = os.getenv('NVIDIA_CLOUD_MODEL_API_KEY')


# function to load model config files...
def load_config(path="./config/prod/model_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)




try:

    model_config =   load_config()
    
    chat_llm_args = model_config['chat_llm_args']
    db_args = model_config['db_args']
    embedder_args = model_config['embedder_args']




except Exception as e:
    
        print(f"Error in : {e}")




llm_pipe = CHATLLMFactory.create_chat_model_pipeline(chat_llm_args['type'],**chat_llm_args)

embedder_pipe =  EMBFactory.create_embedder_model_pipeline(embedder_args['type'],**embedder_args)

# Pass the embeddings model ...

embedder_args['embedding_model']  = embedder_pipe.load_model()

vector_db_pipe = VECTORDBFactory.create_vector_db_pipeline(db_args['type'],**db_args)




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






start_time = time.perf_counter()  # Start timer



try :
        
    # Get LLM instance
    llm =  llm_pipe.load_model()
 
    elapsed = (time.perf_counter() - start_time) * 1000
    
    logger.info(f"⚡ Latency (llm initialization): {elapsed:.2f} ms")

except Exception as e:
    
    print(f" something went wrong  {e}")
    

start_time = time.perf_counter()  # Start timer

# Get vector database
vector_db = vector_db_pipe.load_faiss_db()

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
