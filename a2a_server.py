#!/usr/bin/env python3
"""A2A (Agent-to-Agent) server wrapper for WorldMonitor.

Exposes WorldMonitor's news intelligence APIs as an A2A-compliant agent
so OpenFang can dispatch news/intelligence queries to it.

Port: 5173 is WorldMonitor's Vite dev server. This A2A wrapper runs on 5174
and proxies to the Vite dev server's API endpoints.

Uses Ollama (local) for AI summarization of raw data into intelligence briefs.
"""

import json
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="WorldMonitor A2A Wrapper")

WORLDMONITOR_URL = "http://localhost:5173"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:9b"

AGENT_CARD = {
    "name": "worldmonitor",
    "description": "Real-time global intelligence dashboard. AI-powered news aggregation, geopolitical monitoring, market data, and infrastructure tracking across 435+ feeds and 92 exchanges.",
    "url": "http://localhost:5174",
    "version": "2.6.5",
    "skills": [
        {
            "id": "news-intelligence",
            "name": "News Intelligence",
            "description": "Aggregate and analyze global news across 435+ feeds, 15 categories with AI synthesis",
        },
        {
            "id": "geopolitical-monitoring",
            "name": "Geopolitical Monitoring",
            "description": "Monitor geopolitical events, conflict signals, military movements, and escalation indicators",
        },
        {
            "id": "market-data",
            "name": "Market Data",
            "description": "Real-time data from 92 stock exchanges, commodities, crypto, and market composite signals",
        },
        {
            "id": "country-risk",
            "name": "Country Risk Assessment",
            "description": "Country Intelligence Index — composite risk scoring across 12 signal categories",
        },
        {
            "id": "intelligence-brief",
            "name": "Intelligence Brief",
            "description": "AI-synthesized intelligence brief combining news, market data, geopolitical signals, and risk assessments",
        },
    ],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
}

# Query classification keywords for routing to relevant data sources
CATEGORY_KEYWORDS = {
    "market": ["market", "stock", "stocks", "trading", "trade", "index", "indices",
               "s&p", "nasdaq", "dow", "bull", "bear", "rally", "crash", "earnings",
               "commodit", "oil", "gold", "crypto", "bitcoin", "etf", "sector",
               "fear", "greed", "price", "ticker"],
    "geopolitical": ["geopolitical", "geopolitic", "war", "conflict", "military",
                     "nato", "sanctions", "diplomacy", "treaty", "escalat",
                     "iran", "russia", "china", "ukraine", "israel", "gaza",
                     "missile", "nuclear", "posture", "theater"],
    "news": ["news", "headline", "breaking", "latest", "today", "happening",
             "update", "briefing", "brief", "overnight", "morning"],
    "risk": ["risk", "threat", "alert", "warning", "crisis", "emergency",
             "cyber", "infrastructure", "outage", "earthquake", "wildfire",
             "radiation", "unrest", "displacement"],
    "economic": ["economic", "economy", "gdp", "inflation", "fed", "interest rate",
                 "unemployment", "consumer", "spending", "debt", "macro",
                 "supply chain", "shipping", "trade"],
    "climate": ["climate", "weather", "storm", "hurricane", "flood",
                "temperature", "drought", "wildfire", "natural disaster"],
}

# Bootstrap data keys grouped by category
BOOTSTRAP_KEYS_BY_CATEGORY = {
    "market": ["marketQuotes", "commodityQuotes", "sectors", "etfFlows",
               "cryptoQuotes", "fearGreedIndex", "predictions", "forecasts"],
    "geopolitical": ["theaterPosture", "riskScores", "iranEvents", "ucdpEvents",
                     "crossSourceSignals", "gdeltIntel"],
    "news": ["insights", "correlationCards", "positiveGeoEvents"],
    "risk": ["earthquakes", "outages", "cyberThreats", "wildfires",
             "radiationWatch", "unrestEvents", "weatherAlerts",
             "securityAdvisories"],
    "economic": ["macroSignals", "shippingRates", "chokepoints", "spending",
                 "nationalDebt", "marketImplications", "consumerPricesOverview"],
    "climate": ["climateAnomalies", "naturalEvents", "thermalEscalation"],
}

tasks: dict[str, dict] = {}


def classify_query(text: str) -> list[str]:
    """Classify query text into relevant data categories."""
    text_lower = text.lower()
    matched = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(category)
    # Default to news + market if no specific category matched
    if not matched:
        matched = ["news", "market"]
    return matched


def select_bootstrap_keys(categories: list[str]) -> list[str]:
    """Select bootstrap data keys based on query categories."""
    keys = []
    for cat in categories:
        keys.extend(BOOTSTRAP_KEYS_BY_CATEGORY.get(cat, []))
    # Always include insights for context
    if "insights" not in keys:
        keys.append("insights")
    return list(dict.fromkeys(keys))  # dedupe preserving order


async def fetch_bootstrap_subset(client: httpx.AsyncClient, keys: list[str]) -> dict:
    """Fetch bootstrap data, filtering to relevant keys."""
    try:
        resp = await client.get(f"{WORLDMONITOR_URL}/api/bootstrap", timeout=20)
        if resp.status_code == 200:
            data = resp.json()
            # Filter to only the keys we need
            return {k: v for k, v in data.items() if k in keys and v}
        return {}
    except Exception:
        return {}


def truncate_data(data: dict, max_items: int = 10) -> dict:
    """Truncate lists in data to avoid overwhelming the LLM."""
    truncated = {}
    for key, value in data.items():
        if isinstance(value, list):
            truncated[key] = value[:max_items]
        elif isinstance(value, dict):
            # For nested dicts, keep as-is but truncate nested lists
            truncated[key] = {
                k: v[:max_items] if isinstance(v, list) else v
                for k, v in value.items()
            }
        else:
            truncated[key] = value
    return truncated


async def summarize_with_ollama(query: str, raw_data: dict) -> str:
    """Use Ollama to synthesize raw data into an intelligence brief."""
    data_str = json.dumps(truncate_data(raw_data), indent=1, ensure_ascii=False, default=str)
    # Cap data to avoid token limits
    if len(data_str) > 12000:
        data_str = data_str[:12000] + "\n... [truncated]"

    prompt = f"""You are a concise intelligence analyst. Synthesize the following WorldMonitor data into a focused intelligence brief answering the user's query.

USER QUERY: {query}

RAW DATA:
{data_str}

Instructions:
- Be concise and actionable — no filler
- Highlight the most important signals relevant to the query
- Include specific numbers, tickers, and percentages when available
- Flag any urgent alerts or anomalies
- Structure with clear sections if multiple topics
- If data is missing or unavailable, say so briefly
- Keep the brief under 500 words"""

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 800},
                },
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get("message", {}).get("content", "AI summarization failed — no content returned.")
            return f"Ollama returned status {resp.status_code}. Raw data available but AI summary unavailable."
    except httpx.ConnectError:
        return f"Ollama not running at {OLLAMA_URL}. Returning raw data summary instead.\n\nKeys available: {', '.join(raw_data.keys())}"
    except Exception as e:
        return f"AI summarization error: {e}. Raw data keys: {', '.join(raw_data.keys())}"


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(content=AGENT_CARD)


@app.post("/a2a")
async def handle_a2a_task(request: Request):
    body = await request.json()
    method = body.get("method", "")
    params = body.get("params", {})

    if method == "tasks/send":
        return await send_task(params, body.get("id"))
    elif method == "tasks/get":
        return get_task(params.get("id"), body.get("id"))
    elif method == "tasks/cancel":
        return cancel_task(params.get("id"), body.get("id"))
    else:
        return JSONResponse(content={"jsonrpc": "2.0", "id": body.get("id"), "error": {"code": -32601, "message": f"Unknown method: {method}"}})


async def send_task(params: dict, rpc_id=None):
    message_parts = params.get("message", {}).get("parts", [])
    text = next((p["text"] for p in message_parts if p.get("type") == "text"), "Get latest news and market summary")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"id": task_id, "status": {"state": "working"}, "created": datetime.now(timezone.utc).isoformat()}

    # Step 1: Classify the query to determine which data sources to fetch
    categories = classify_query(text)

    # Step 2: Select relevant bootstrap keys
    keys = select_bootstrap_keys(categories)

    # Step 3: Fetch data from WorldMonitor
    raw_data = {}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            raw_data = await fetch_bootstrap_subset(client, keys)
    except httpx.ConnectError:
        response_text = f"WorldMonitor not running at {WORLDMONITOR_URL}. Start with: cd repos/worldmonitor && npm run dev"
        tasks[task_id]["status"] = {"state": "completed"}
        tasks[task_id]["artifacts"] = [{"parts": [{"type": "text", "text": response_text}]}]
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": tasks[task_id]})
    except Exception as e:
        response_text = f"WorldMonitor query error: {e}"
        tasks[task_id]["status"] = {"state": "completed"}
        tasks[task_id]["artifacts"] = [{"parts": [{"type": "text", "text": response_text}]}]
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": tasks[task_id]})

    if not raw_data:
        response_text = f"No data returned from WorldMonitor for categories: {', '.join(categories)}. Query: {text}"
        tasks[task_id]["status"] = {"state": "completed"}
        tasks[task_id]["artifacts"] = [{"parts": [{"type": "text", "text": response_text}]}]
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": tasks[task_id]})

    # Step 4: Synthesize with Ollama AI
    response_text = await summarize_with_ollama(text, raw_data)

    tasks[task_id]["status"] = {"state": "completed"}
    tasks[task_id]["artifacts"] = [{"parts": [{"type": "text", "text": response_text}]}]
    return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": tasks[task_id]})


def get_task(task_id: str, rpc_id=None):
    task = tasks.get(task_id)
    if not task:
        return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "error": {"code": -32602, "message": "Task not found"}})
    return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": task})


def cancel_task(task_id: str, rpc_id=None):
    task = tasks.get(task_id)
    if task:
        task["status"] = {"state": "canceled"}
    return JSONResponse(content={"jsonrpc": "2.0", "id": rpc_id, "result": {"id": task_id, "status": {"state": "canceled"}}})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5174)

