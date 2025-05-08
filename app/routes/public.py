from fastapi import APIRouter
from app.models.schemas import ChatInput,ChatOutput
from  app.assistant import ChatManager
from fastapi import HTTPException
from fastapi import FastAPI
from langserve import add_routes
from app.assistant import ChatManagerRunnable,ChatManager
from llms.model_manager import ModelManager
from retrieval.vector_store import get_vector_store
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
from pathlib import Path
import logging
import traceback

logger = logging.getLogger("uvicorn")




# app router
router  = APIRouter()




@router.post("/chat",response_model=ChatOutput)
async def get_response(request: ChatInput):

    query = request.query
    last_3_turn = request.last_3_turn


    try:

        logger.info(f"📥 Received query: {request.query}")
        logger.info(f"📚 Context: {request.last_3_turn}")

        response = await manager.run(query, last_3_turn)

        logger.info(f"📤 Returning response: {response}")

        return response


    except Exception as e:

        logger.error("❌ Error processing request: %s", traceback.format_exc())

        raise HTTPException(status_code=500, detail=str(e))
