"""
Digital Twin Agent - Runner
Handles queries using pre-built vector store
"""

import os
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    english_query: str
    vector_store: Chroma
    raw_answer: str  # Store the answer before speech optimization


def load_vector_store():
    """Load existing vector store."""
    persist_dir = Path("chroma_db")
    
    if not persist_dir.exists() or not any(persist_dir.iterdir()):
        raise FileNotFoundError(
            f"Vector store not found at {persist_dir}\n"
            "Please run: python -m src.create_database"
        )
    
    print("Loading vector store...")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    vector_store = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="digital_twin"
    )
    
    print(f"✓ Loaded vector store from {persist_dir}\n")
    return vector_store


def router_node(state: AgentState) -> AgentState:
    """Use LLM to translate and route the query."""
    llm = ChatOpenAI(
        model="gpt-4o",  
        temperature=0
    )
    
    user_query = state["messages"][-1].content
    
    routing_prompt = f"""You are a query router. Analyze the question and determine which data source to use.

DATA SOURCES:
1. linkedin - Use for Yauheniya's work experience, jobs, education, degrees, universities, career history, professional background, skills
2. github - Use for Yauheniya's programming projects, code repositories, technical implementations, software development, coding work
3. medium - Use for Yauheniya's articles, blog posts, writing, published content, opinions, topics written about, what someone wrote, content they created
4. general - Use for broad overview questions that need multiple sources

EXAMPLES:
- "What did Yauheniya write about?" → Medium articles (asking about published content)
- "Tell me about Yauheniya's published articles" → Medium articles (asking about publications)
- "What topics interest Yauheniya?" → Medium articles 
- "Where did Yauheniya study?" → linkedin (education)
- "What projects has Yauheniya built?" → github (technical work)
- "Tell me about Yauheniya" → general (overview)

USER QUESTION: {user_query}

Respond in this exact format:
TRANSLATED_QUERY: [question in English]
ROUTE: [linkedin|github|medium|general]"""

    response = llm.invoke([HumanMessage(content=routing_prompt)])
    
    # Parse response
    translated_query = user_query
    route = "general"
    
    for line in response.content.strip().split('\n'):
        if line.startswith('TRANSLATED_QUERY:'):
            translated_query = line.replace('TRANSLATED_QUERY:', '').strip()
        elif line.startswith('ROUTE:'):
            route = line.replace('ROUTE:', '').strip().lower()
    
    print(f"[DEBUG] Routing to: {route}")  # Debug output
    
    state["english_query"] = translated_query
    state["messages"].append(
        SystemMessage(content=f"[Route: {route} | Query: {translated_query}]")
    )
    
    return state


def get_route(state: AgentState) -> Literal["linkedin", "github", "medium", "general"]:
    """Extract route from system message."""
    last_msg = state["messages"][-1].content
    if "[Route:" in last_msg:
        route = last_msg.split("[Route:")[1].split("|")[0].strip()
        return route
    return "general"


def retrieve_linkedin(state: AgentState) -> AgentState:
    """Retrieve LinkedIn context."""
    query = state["english_query"]
    results = state["vector_store"].similarity_search(
        query,
        k=3,
        filter={"source": "linkedin"}
    )
    
    context = "\n\n".join([doc.page_content for doc in results])
    state["messages"].append(
        SystemMessage(content=f"[LinkedIn Context]\n{context}")
    )
    return state


def retrieve_github(state: AgentState) -> AgentState:
    """Retrieve GitHub context."""
    query = state["english_query"]
    results = state["vector_store"].similarity_search(
        query,
        k=8,
        filter={"source": "github"}
    )
    
    context = "\n\n".join([doc.page_content for doc in results])
    state["messages"].append(
        SystemMessage(content=f"[GitHub Projects Context]\n{context}")
    )
    return state


def retrieve_medium(state: AgentState) -> AgentState:
    """Retrieve Medium context."""
    query = state["english_query"]
    
    print(f"[DEBUG] Searching Medium with query: {query}")  # Debug
    
    # For general "what did you write" questions, get diverse results
    try:
        results = state["vector_store"].max_marginal_relevance_search(
            query,
            k=8,
            fetch_k=20,  # Fetch more candidates for diversity
            filter={"source": "medium"}
        )
    except:
        # Fallback to regular search if MMR fails
        results = state["vector_store"].similarity_search(
            query,
            k=8,
            filter={"source": "medium"}
        )
    
    print(f"[DEBUG] Found {len(results)} Medium chunks")  # Debug
    
    if results:
        # Debug: print first chunk to see what we're getting
        print(f"[DEBUG] First chunk preview: {results[0].page_content[:200]}...")
        print(f"[DEBUG] First chunk metadata: {results[0].metadata}")
        
        context = "\n\n".join([doc.page_content for doc in results])
        state["messages"].append(
            SystemMessage(content=f"[Medium Articles Context]\n{context}")
        )
    else:
        state["messages"].append(
            SystemMessage(content=f"[Medium Articles Context]\nNo articles found.")
        )
    
    return state


def retrieve_general(state: AgentState) -> AgentState:
    """Retrieve from all sources."""
    query = state["english_query"]
    results = state["vector_store"].similarity_search(query, k=8)
    
    # Group by source
    by_source = {}
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(doc.page_content)
    
    # Format context
    contexts = []
    for source, chunks in by_source.items():
        contexts.append(f"[{source.title()} Context]\n" + "\n\n".join(chunks))
    
    state["messages"].append(
        SystemMessage(content="\n\n".join(contexts))
    )
    return state


def answer_node(state: AgentState) -> AgentState:
    """Generate final answer."""

    llm = ChatOpenAI(
        model="gpt-4o",  
        temperature=0
    )
    
    english_query = state["english_query"]
    
    # Collect all context from system messages
    contexts = []
    for msg in state["messages"]:
        if isinstance(msg, SystemMessage) and "Context]" in msg.content:
            contexts.append(msg.content)
    
    combined_context = "\n\n".join(contexts)
    
    system_prompt = f"""You are a digital twin assistant representing Yauheniya. Answer questions based ONLY on the provided context.

WHAT YOU HAVE ACCESS TO:
- LinkedIn: Yauheniya's professional background, education, work experience
- GitHub: Yauheniya'a programming projects and technical work
- Medium: Articles and blog posts Yauheniya has written

INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. Be conversational and natural in your responses. Talk about Yauheniya in 3rd person. Yauheniya studied ...
3. When asked broad questions like "what did Yauheniya write about?", extract and summarize the main topics/themes from the article chunks provided
4. Mention specific article titles or topics when available in the metadata or content
5. If the context contains article content but not clear titles, describe the topics discussed
6. ALWAYS respond in English

CONTEXT:
{combined_context}"""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=english_query)
    ]
    
    response = llm.invoke(messages)
    
    # Store raw answer for speech optimization
    state["raw_answer"] = response.content
    state["messages"].append(AIMessage(content=response.content))
    
    return state


def speech_optimization_node(state: AgentState) -> AgentState:
    """Optimize answer for voice output."""
    
    llm = ChatOpenAI(
        model="gpt-4o",  
        temperature=0
    )
    
    raw_answer = state["raw_answer"]
    
    speech_prompt = f"""You are a speech optimization assistant. Transform the following answer to be natural and clear when spoken aloud by a text-to-speech system.

CRITICAL RULES:
1. Remove ALL markdown formatting (**bold**, *italic*, `code`, etc.) - just plain text
2. Remove ALL special characters like asterisks, backticks, underscores used for formatting
3. Do NOT add any preamble like "Sure! Here's..." or "Here is..." - start DIRECTLY with the answer content
4. Use simple, conversational language
5. Break long sentences into shorter ones for better pacing
6. Replace technical jargon with clear explanations when possible
7. Add natural transitions between ideas
8. Keep the same factual content, just make it speech-friendly
9. Keep it concise but natural
10. Article titles should be mentioned naturally without quotes or formatting (e.g., "the article Denoising Images with Autoencoders" not "the article *Denoising Images with Autoencoders*")

ORIGINAL ANSWER:
{raw_answer}

Provide ONLY the speech-optimized answer with NO preamble:"""
    
    response = llm.invoke([HumanMessage(content=speech_prompt)])
    
    # Additional cleanup to ensure no markdown artifacts remain
    cleaned_content = response.content
    
    # Remove common markdown patterns that might slip through
    import re
    cleaned_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned_content)  # **bold**
    cleaned_content = re.sub(r'\*([^*]+)\*', r'\1', cleaned_content)      # *italic*
    cleaned_content = re.sub(r'`([^`]+)`', r'\1', cleaned_content)        # `code`
    cleaned_content = re.sub(r'_([^_]+)_', r'\1', cleaned_content)        # _underline_
    cleaned_content = re.sub(r'#{1,6}\s+', '', cleaned_content)           # # headers
    
    # Remove any preamble phrases if they slipped through
    preamble_patterns = [
        r'^Sure[!,.]?\s+Here[\'s\s]+.*?:\s*',
        r'^Here[\'s\s]+.*?:\s*',
        r'^Okay[!,.]?\s+',
        r'^Alright[!,.]?\s+'
    ]
    for pattern in preamble_patterns:
        cleaned_content = re.sub(pattern, '', cleaned_content, flags=re.IGNORECASE)
    
    cleaned_content = cleaned_content.strip()
    
    # Replace the last message with speech-optimized version
    state["messages"][-1] = AIMessage(content=cleaned_content)
    
    return state


def create_graph(vector_store: Chroma):
    """Create the LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("linkedin", retrieve_linkedin)
    workflow.add_node("github", retrieve_github)
    workflow.add_node("medium", retrieve_medium)
    workflow.add_node("general", retrieve_general)
    workflow.add_node("answer", answer_node)
    workflow.add_node("speech_optimize", speech_optimization_node)
    
    # Edges
    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        get_route,
        {
            "linkedin": "linkedin",
            "github": "github",
            "medium": "medium",
            "general": "general"
        }
    )
    workflow.add_edge("linkedin", "answer")
    workflow.add_edge("github", "answer")
    workflow.add_edge("medium", "answer")
    workflow.add_edge("general", "answer")
    workflow.add_edge("answer", "speech_optimize")
    workflow.add_edge("speech_optimize", END)
    
    return workflow.compile()


class DigitalTwin:
    """Digital Twin Agent."""
    
    def __init__(self):
        print("Initializing Digital Twin...")
        self.vector_store = load_vector_store()
        self.graph = create_graph(self.vector_store)
        print("✓ Ready!\n")
        
        self._save_graph_visualization()
    
    def _save_graph_visualization(self):
        """Save LangGraph visualization as PNG."""
        try:
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            output_path = Path("langgraph_visualization.png")
            with open(output_path, "wb") as f:
                f.write(png_bytes)
            print(f"✓ Graph visualization saved to {output_path}\n")
        except Exception as e:
            print(f"Note: Could not save graph visualization: {e}\n")
    
    def ask(self, question: str) -> str:
        """Ask a question."""
        state = {
            "messages": [HumanMessage(content=question)],
            "english_query": "",
            "vector_store": self.vector_store,
            "raw_answer": ""
        }
        
        result = self.graph.invoke(state)
        return result["messages"][-1].content


if __name__ == "__main__":
    # Initialize
    twin = DigitalTwin()
    
    # Test questions
    print("Testing with sample questions:\n")
    questions = [
        "What is Yauheniya's educational background?",
        "Welche Projekte hat Yauheniya mit Java entwickelt?",
        "What has Yauheniya written about?",
    ]
    
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {twin.ask(q)}\n")
        print("-" * 80 + "\n")
    
    # Interactive
    print("Interactive mode (type 'quit' to exit):\n")
    while True:
        q = input("Your question: ").strip()
        if q.lower() in ['quit', 'exit', 'q']:
            break
        if q:
            print(f"\n{twin.ask(q)}\n")