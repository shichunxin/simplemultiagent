import json
import time

from markdown import Markdown
from rich.jupyter import display
from tqdm.auto import tqdm
import tiktoken
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re
import textwrap
import copy
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI

from multiagent.core.agent import run_once_messages
from multiagent.core.context import context_engine, AgentRegistry
from multiagent.core.mcp import create_mcp_message

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536 # Dimension for text-embedding-3-small
GENERATION_MODEL = ""

INDEX_NAME = 'genai-mas-mcp-ch3'
NAMESPACE_KNOWLEDGE = "KnowledgeStore"
NAMESPACE_CONTEXT = "ContextLibrary"

client = OpenAI()
pc = Pinecone(api_key="")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(text):
    """Generates embeddings for a single text query with retries."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding

# === Pinecone Interaction ===
def query_pinecone(query_text, namespace, top_k=1):
    """Embeds the query text and searches the specified Pinecone namespace."""
    try:
        query_embedding = get_embedding(query_text)
        response = pc.Index(INDEX_NAME).query(
            vector=query_embedding,
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        return response['matches']
    except Exception as e:
        print(f"Error querying Pinecone (Namespace: {namespace}): {e}")
        raise e

def agent_context_librarian(mcp_message):
    """
    Retrieves the appropriate Semantic Blueprint from the Context Library.
    """
    print("\n[Librarian] Activated. Analyzing intent...")
    # Extract the specific input required by this agent
    requested_intent = mcp_message['content'].get('intent_query')

    if not requested_intent:
        raise ValueError("Librarian requires 'intent_query' in the input content.")

    # Query Pinecone Context Namespace
    results = query_pinecone(requested_intent, NAMESPACE_CONTEXT, top_k=1)

    if results:
        match = results[0]
        print(f"[Librarian] Found blueprint '{match['id']}' (Score: {match['score']:.2f})")
        # Retrieve the blueprint JSON string stored in metadata
        blueprint_json = match['metadata']['blueprint_json']
        # The output content IS the blueprint itself (as a string)
        content = blueprint_json
    else:
        print("[Librarian] No specific blueprint found. Returning default.")
        # Fallback default
        content = json.dumps({"instruction": "Generate the content neutrally."})

    return create_mcp_message("Librarian", content)

# === 4.2. Researcher Agent (Factual RAG) ===
def agent_researcher(mcp_message):
    """
    Retrieves and synthesizes factual information from the Knowledge Base.
    """
    print("\n[Researcher] Activated. Investigating topic...")
    # Extract the specific input required by this agent
    topic = mcp_message['content'].get('topic_query')

    if not topic:
        raise ValueError("Researcher requires 'topic_query' in the input content.")

    # Query Pinecone Knowledge Namespace
    results = query_pinecone(topic, NAMESPACE_KNOWLEDGE, top_k=3)

    if not results:
        print("[Researcher] No relevant information found.")
        # Return a string indicating no data found
        return create_mcp_message("Researcher", "No data found on the topic.")

    # Synthesize the findings (Retrieve-and-Synthesize)
    print(f"[Researcher] Found {len(results)} relevant chunks. Synthesizing...")
    source_texts = [match['metadata']['text'] for match in results]

    system_prompt = """You are an expert research synthesis AI.
    Synthesize the provided source texts into a concise, bullet-pointed summary relevant to the user's topic. Focus strictly on the facts provided in the sources. Do not add outside information."""

    user_prompt = f"Topic: {topic}\n\nSources:\n" + "\n\n---\n\n".join(source_texts)

    # Use a low temperature for factual synthesis
    messages = [
        {
            "role": "system",
            "context": system_prompt
        },
        {
            "role": "user",
            "context": user_prompt
        }
    ]
    findings = run_once_messages(messages).choices[0].message.context

    # The output content IS the findings (as a string)
    return create_mcp_message("Researcher", findings)

# === 4.3. Writer Agent (Generation) ===
def agent_writer(mcp_message):
    """
    Combines the factual research with the semantic blueprint to generate the final output.
    Crucially enhanced to handle either raw facts OR previous content for rewriting tasks.
    """
    print("\n[Writer] Activated. Applying blueprint to source material...")

    # Extract inputs.
    blueprint_json_string = mcp_message['content'].get('blueprint')
    # Check for 'facts' first, then 'previous_content'
    facts = mcp_message['content'].get('facts')
    previous_content = mcp_message['content'].get('previous_content')

    if not blueprint_json_string:
         raise ValueError("Writer requires 'blueprint' in the input content.")

    # Determine the source material and label for the prompt
    if facts:
        source_material = facts
        source_label = "RESEARCH FINDINGS"
    elif previous_content:
        source_material = previous_content
        source_label = "PREVIOUS CONTENT (For Rewriting)"
    else:
        raise ValueError("Writer requires either 'facts' or 'previous_content'.")


    # The Writer's System Prompt incorporates the dynamically retrieved blueprint
    system_prompt = f"""You are an expert content generation AI.
    Your task is to generate content based on the provided SOURCE MATERIAL.
    Crucially, you MUST structure, style, and constrain your output according to the rules defined in the SEMANTIC BLUEPRINT provided below.

    --- SEMANTIC BLUEPRINT (JSON) ---
    {blueprint_json_string}
    --- END SEMANTIC BLUEPRINT ---

    Adhere strictly to the blueprint's instructions, style guides, and goals. The blueprint defines HOW you write; the source material defines WHAT you write about.
    """

    user_prompt = f"""
    --- SOURCE MATERIAL ({source_label}) ---
    {source_material}
    --- END SOURCE MATERIAL ---

    Generate the content now, following the blueprint precisely.
    """

    # Generate the final content (slightly higher temperature for potential creativity)
    messages = [
        {
            "role":"system",
            "context":system_prompt
        },
        {
            "role": "user",
            "context": user_prompt
        }
    ]
    final_output = run_once_messages(messages).choices[0].message.context
    return create_mcp_message("Writer", final_output)

print("Specialist Agents defined.")

def main():
    print("******** Example 1: STANDARD WORKFLOW (Suspenseful Narrative) **********\n")
    goal_1 = "Write a short, suspenseful scene for a children's story about the Apollo 11 moon landing, highlighting the danger."

    AGENT_TOOLKIT = AgentRegistry()
    AGENT_TOOLKIT.register("Librarian",agent_context_librarian)
    AGENT_TOOLKIT.register("Researcher",agent_researcher)
    AGENT_TOOLKIT.register("Writer", agent_writer)

    result_1, trace_1 = context_engine(goal_1)

    if result_1:
        print("\n******** FINAL OUTPUT 1 **********\n")
        display(Markdown(result_1))
        print("\n\n" + "=" * 50 + "\n\n")

if __name__ == "__main__":
    main()