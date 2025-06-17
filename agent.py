import os
from typing import List, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langchain_core.prompts import MessagesPlaceholder

# imports for Tool Calling
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
# caching tools
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
import config
# Import for vector store management
from vector_store_manager import create_and_save_vector_store, load_vector_store


# --- GLOBAL CACHE SETUP ---
print("Setting up SQLite cache for LLM calls using set_llm_cache...")
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# --- 1. MODULAR COMPONENT LOADERS (No changes) ---

def get_llm():
    if config.USE_OPENSOURCE_LLM:
        return Ollama(model="llama3")
    else:
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

def get_embedding_model():
    print("--- Initializing HuggingFace Embedding Model ---")
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)

def get_retriever():
    """
    Loads the vector store from disk if it exists, otherwise creates and saves it.
    Returns the vector store as a retriever.
    """
    embedding_model = get_embedding_model()
    print("--- Attempting to load or create vector store ---")
    vectorstore = load_vector_store(embedding_model)
    
    if vectorstore is None:
        print("--- Existing vector store not found or failed to load. Creating a new one. ---")
        vectorstore = create_and_save_vector_store(embedding_model)
        if vectorstore is None:
            print("CRITICAL: Failed to create or load the vector store. The RAG agent may not function correctly with document retrieval.")
            return None 

    print("--- Vector store initialized. Returning as retriever. ---")
    return vectorstore.as_retriever(search_kwargs={'k':2})


# --- PRE-ROUTER CHAIN (For Handling Greeting) ---

pre_router_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Classify the user's input as 'greeting' for simple hellos/thanks, or 'information_seeking' for actual questions. Respond with only one word."),
        ("human", "{question}")
    ]) | get_llm() | StrOutputParser()
)

# --- 2. LANGGRAPH STATE DEFINITION ---

class GraphState(TypedDict):
    user_question: str
    conversation_history: list
    rewritten_questions: List[str]
    individual_answers: dict
    final_response: str

# --- 3. OPTIMIZED ARCHITECTURE (TOOL CALLING) ---

# Components initialization
llm = get_llm()
retriever = get_retriever()

# STEP 1: Create a tool that will search the knowledge base
@tool
def bignalytics_knowledge_search(query: str) -> str:
    """
    Searches the Bignalytics knowledge base to answer questions about courses,
    fees, duration, faculty, curriculum, and other institute-specific details.
    """
    print(f"--- 🛠️ ENTERING Bignalytics Knowledge Tool for query: '{query}' ---")
    docs = retriever.invoke(query)
    if not docs:
        print(f"--- 🛠️ Bignalytics Knowledge Tool: No relevant documents found for '{query}' ---")
        return "No relevant information found in the knowledge base."
    print(f"--- 🛠️ Bignalytics Knowledge Tool: Found {len(docs)} documents for '{query}'. Docs: {docs} ---")
    result = "\n---\n".join(doc.page_content for doc in docs)
    print(f"--- 🛠️ Bignalytics Knowledge Tool: Returning result for '{query}': '{result[:500]}...' ---") # Log a snippet
    return result

# STEP 2: Define the Rewriter and Synthesizer chains
rewriter_chain = (ChatPromptTemplate.from_messages([
    ("system", """You are an expert query analyst. Your task is to analyze the user's question.
- If the question contains multiple, distinct questions (e.g., 'what are the fees and duration?'), break it down into a list of self-contained, standalone questions.
- If the question is already a single, simple question (e.g., 'what is bignalytics?' or 'tell me about the fees'), DO NOT break it down. Simply return it as a list containing that single question.
- Output ONLY a JSON object with a single key 'questions' containing the list of strings."""),
    ("human", "Conversation History:\n{history}\n\nUser Question: {question}")
]) | llm | JsonOutputParser())


synthesis_chain = (ChatPromptTemplate.from_messages([
    ("system", "You are an expert response synthesizer. Combine the following question-answer pairs into a single, cohesive, and natural-sounding paragraph. Address the user directly."),
    ("human", "Here is the information to synthesize:\n{answers}")
]) | llm | StrOutputParser())


# STEP 3: Create a new agent that uses Tool Calling
tools = [bignalytics_knowledge_search]



agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized assistant for an IT training institute named Bignalytics.
Your primary function is to answer questions using a specific tool.

You have access to a tool named `bignalytics_knowledge_search`. This tool can search for any information related to Bignalytics institute.

Here are your instructions:
1. For ANY question that contains the word 'Bignalytics' or seems to be about the institute, you MUST use the `bignalytics_knowledge_search` tool. There are no exceptions.
2. After using the tool, answer the user's question based on the information provided by the tool.
3. If the tool returns 'No relevant information found', then you should state that you could not find information on that specific topic.
4. If the question is clearly a general knowledge question and NOT about Bignalytics (e.g., 'What is Python?'), you can answer from your own knowledge without using the tool.
"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

tool_agent = create_tool_calling_agent(llm, tools, agent_prompt)
# IMPORTANT: Set verbose=True for detailed agent execution logs
agent_executor = AgentExecutor(agent=tool_agent, tools=tools, verbose=True) 

# --- 4. GRAPH NODES ---

def rewrite_query_node(state: GraphState):
    """Rewrites the user's question into a list of standalone questions."""
    print(f"--- 🧠 ENTERING rewrite_query_node ---")
    print(f"  Input state: user_question='{state['user_question']}', history_len={len(state['conversation_history'])}")
    result = rewriter_chain.invoke({
        "history": state["conversation_history"],
        "question": state["user_question"]
    })
    print(f"  Rewriter chain output: {result}")
    rewritten_qs = result.get('questions', [])
    print(f"--- 🧠 EXITING rewrite_query_node with rewritten_questions: {rewritten_qs} ---")
    return {"rewritten_questions": rewritten_qs}

def process_questions_node(state: GraphState):
    """Processes each question using the single-call tool-calling agent."""
    print(f"--- ⚙️ ENTERING process_questions_node ---")
    print(f"  Input rewritten_questions: {state['rewritten_questions']}")
    answers = {}
    for question_idx, question in enumerate(state["rewritten_questions"]):
        print(f"  Processing question #{question_idx + 1}: '{question}' with AgentExecutor")
        agent_input = {"input": question, "chat_history": state["conversation_history"]}
        print(f"    AgentExecutor input: {agent_input}")
        response = agent_executor.invoke(agent_input)
        print(f"    AgentExecutor raw response for '{question}': {response}")

        answers[question] = response.get('output', "Error: No output from agent.") # Default if 'output' key is missing
    print(f"--- ⚙️ EXITING process_questions_node with individual_answers: {answers} ---")
    return {"individual_answers": answers}


## Decider node
def decide_to_synthesize_node(state: GraphState):
    """
    If there's only one answer, copies it to the final_response.
    Otherwise, does nothing. The conditional edge will handle routing.
    """
    print(f"--- 🧐 ENTERING decide_to_synthesize_node ---")
    print(f"  Input individual_answers: {state['individual_answers']}")
    num_answers = len(state["individual_answers"])
    if num_answers == 1:
        single_answer = list(state["individual_answers"].values())[0]
        print(f"  Decision: Single question (count: {num_answers}), copying answer directly: '{single_answer}'")
        print(f"--- 🧐 EXITING decide_to_synthesize_node (single answer) ---")
        return {"final_response": single_answer}
    else:
        print(f"  Decision: Multiple questions (count: {num_answers}), proceeding to synthesis.")
        print(f"--- 🧐 EXITING decide_to_synthesize_node (multiple answers) ---")
        return {} # No change to final_response, synthesis_node will populate it


def synthesize_response_node(state: GraphState):
    """Synthesizes the final response from all the individual answers."""
    print(f"--- ✨ ENTERING synthesize_response_node ---")
    print(f"  Input individual_answers: {state['individual_answers']}")
    answers_str = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in state["individual_answers"].items()])
    print(f"  Synthesizer chain input (answers_str): '{answers_str}'")
    final_response = synthesis_chain.invoke({"answers": answers_str})
    print(f"  Synthesizer chain output (final_response): '{final_response}'")
    print(f"--- ✨ EXITING synthesize_response_node ---")
    return {"final_response": final_response}

# --- 5. MAIN GRAPH BUILDER FUNCTION (No changes to structure) ---


def get_rag_agent():
    """Builds and compiles the final agent"""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("rewriter", rewrite_query_node)
    graph.add_node("processor", process_questions_node)
    graph.add_node("decider", decide_to_synthesize_node)
    graph.add_node("synthesizer", synthesize_response_node)

    # Define edges
    graph.set_entry_point("rewriter")
    graph.add_edge("rewriter", "processor")
    graph.add_edge("processor", "decider")

    # conditional logic when deciding whether to synthesize or not if there's only one answer
    graph.add_conditional_edges(
        "decider",
        # This lambda function will check the state and decide the routing
        lambda state: "synthesizer" if len(state["individual_answers"]) > 1 else END,
        {
            "synthesizer": "synthesizer",
            END: END  # Corrected: Add END to the path_map
        }
    )
    graph.add_edge("synthesizer", END)

    # Compile the graph
    print("✅ RAG Agent with Corrected Conditional Logic Compiled Successfully!")
    return graph.compile()