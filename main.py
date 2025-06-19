import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from langchain.memory import ConversationSummaryBufferMemory
import uuid
from typing import Optional
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from agent import get_rag_agent, pre_router_chain
from agent import get_rag_agent, get_text_generation_llm


# Load environment variables from .env file
load_dotenv() 



# --- SERVER-SIDE MEMORY & PRE-ROUTER SETUP ---
conversation_memories = {}
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None # Now we will take only an ID from the client, not the full history


app = FastAPI(
    title="Bignalytics RAG Agent API",
    description="Endpoint for interacting with the Bignalytics RAG agent.",
    version="1.0.0"
)

# Add CORS middleware to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the real agent
agent = get_rag_agent()

@app.post("/chat")
def chat_with_agent(request: ChatRequest):
    """
    Handles greetings, manages server-side memory, and invokes the RAG agent.
    """
    print(f"\n\n--- New Chat Request ---")
    print(f"Received request: {request}")

     # 1. Handle the conversation ID
    convo_id = request.conversation_id
    if not convo_id:
        # If the client did not send an ID, create a new unique ID
        convo_id = str(uuid.uuid4())
        print(f"--- New Conversation Started with ID: {convo_id} ---")
    
    # Assign convo_id back to request if it was newly generated
    if not request.conversation_id:
        request.conversation_id = convo_id


    # 1. First, classify the intent
    intent = pre_router_chain.invoke({"question": request.question})
    print(f"--- Intent Classified as: {intent} ---")

    # 2. Handle greeting and respond quickly
    if "greeting" in intent.lower():
        greeting_response = """Hello! How can I assist you with information about Bignalytics?

Here are our contact details:

Address: Pearl Business Park, 3, Bhawarkua Main Rd, Above Ramesh Dosa, Near Vishnupuri i bus stop, Vishnu Puri Colony, Indore, Madhya Pradesh - 452001 
Contact Number: 093992-00960"""
        print(f"--- Responding with greeting for ID: {request.conversation_id} ---")
        return {"answer": greeting_response, "conversation_id": request.conversation_id}

    # 3. Get or create memory for each conversation
    if request.conversation_id not in conversation_memories:
        print(f"--- Creating new ConversationSummaryBufferMemory for ID: {request.conversation_id} ---")
        conversation_memories[request.conversation_id] = ConversationSummaryBufferMemory(
            llm=get_text_generation_llm(), max_token_limit=1000, return_messages=True
        )           


    memory = conversation_memories[request.conversation_id]
    print(f"--- Using memory for ID: {request.conversation_id}. Current memory content: {memory.load_memory_variables({})} ---")

    # 4. Call the main agent
    history_langchain_messages = memory.load_memory_variables({})['history']
    input_data = {
        "user_question": request.question,
        "conversation_history": history_langchain_messages
    }
    print(f"--- Invoking RAG agent with input_data for ID {request.conversation_id}: {input_data} ---")
    response = agent.invoke(input_data)
    print(f"--- Raw response from RAG agent for ID {request.conversation_id}: {response} ---")
    final_answer = response.get("final_response")

    # 5. Save the new Q&A to memory
    memory.save_context({"input": request.question}, {"output": final_answer})
    print(f"--- Saved context to memory for ID: {request.conversation_id}. New memory content: {memory.load_memory_variables({})} ---")
    print(f"--- Final answer for ID {request.conversation_id}: {final_answer} ---")

    return {"answer": final_answer, "conversation_id": request.conversation_id}

@app.get("/")
def read_root():
    """Serve the chatbot frontend"""
    return FileResponse("index.html")

@app.get("/api/status")
def api_status():
    return {"status": "Bignalytics RAG Agent is running."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)