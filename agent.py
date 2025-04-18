import google.generativeai as genai
import feedparser
import urllib.parse
from typing import List
from pathlib import Path
import json

# --- Configuration ---
from kaggle_secrets import UserSecretsClient
genai.configure(api_key=UserSecretsClient().get_secret("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash-lite-001")

# --- Conversation Memory ---
MEMORY_FILE = Path("agent_memory.json")

def load_memory():
    if MEMORY_FILE.exists():
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return []

def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)

conversation_history = load_memory()

def update_memory(user_prompt, agent_response):
    conversation_history.append({"role": "user", "content": user_prompt})
    conversation_history.append({"role": "agent", "content": agent_response})
    save_memory(conversation_history)

# --- Core Agent Functions ---
def rewrite_to_query(user_prompt: str) -> str:
    prompt = f"Extract short search keywords from: \"{user_prompt}\""
    return model.generate_content(prompt).text.strip().replace(" ", "+")

def fetch_articles(user_prompt: str, max_results: int = 5):
    clean_prompt = " ".join(user_prompt.strip().split())
    encoded_query = urllib.parse.quote_plus(clean_prompt)
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    return feed.entries[:max_results]

def summarize_with_context(articles: List[dict], prompt: str, memory: str = "") -> List[dict]:
    summarized = []
    for article in articles:
        text = f"""
User prompt: {prompt}
Context: {memory}

News:
Title: {article['title']}
Summary: {article['summary']}

Give 3 bullet-point insights relevant to user query. Cite link: {article['link']}
"""
        summary = model.generate_content(text).text
        summarized.append({
            "title": article['title'],
            "summary": summary,
            "link": article['link']
        })
    return summarized

def generate_plan(summaries: List[dict], prompt: str, memory: str) -> str:
    text = f"""
User Query: {prompt}
Context: {memory}
Summarized News:
{''.join([s['summary'] for s in summaries])}

Give output as function: investment_plan(decision, reasons, confidence_score)
"""
    return model.generate_content(text).text.strip()

def evaluate_plan(plan: str) -> str:
    text = f"""
Evaluate this investment_plan:
{plan}

Rate 1-10 with reasoning.
"""
    return model.generate_content(text).text.strip()

def financial_agent(user_prompt: str, rag_context: str = ""):
    conversation_history.append({"role": "user", "content": user_prompt})
    context_memory = " ".join([item["content"] for item in conversation_history if item["role"] == "user"])

    query = rewrite_to_query(user_prompt)
    articles = fetch_articles(query)
    summaries = summarize_with_context(articles, user_prompt, context_memory)
    action_plan = generate_plan(summaries, user_prompt, rag_context)
    evaluation = evaluate_plan(action_plan)

    conversation_history.append({"role": "agent", "content": action_plan})

    return {
        "query": query,
        "summaries": summaries,
        "action_plan": action_plan,
        "evaluation": evaluation
    }
