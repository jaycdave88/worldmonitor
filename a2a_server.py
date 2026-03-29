#!/usr/bin/env python3
"""A2A (Agent-to-Agent) server wrapper for WorldMonitor.

Exposes WorldMonitor's news intelligence APIs as an A2A-compliant agent
so OpenFang can dispatch news/intelligence queries to it.

Port: 5173 is WorldMonitor's Vite dev server. This A2A wrapper runs on 5174
and proxies to the Vite dev server's API endpoints.
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

AGENT_CARD = {
    "name": "worldmonitor",
    "description": "Real-time global intelligence dashboard. AI-powered news aggregation, geopolitical monitoring, market data, and infrastructure tracking.",
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
    ],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": False,
    },
    "defaultInputModes": ["text/plain"],
    "defaultOutputModes": ["text/plain"],
}

tasks: dict[str, dict] = {}


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
    text = next((p["text"] for p in message_parts if p.get("type") == "text"), "Get latest news")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"id": task_id, "status": {"state": "working"}, "created": datetime.now(timezone.utc).isoformat()}

    results = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch bootstrap data (aggregated news + market data)
            resp = await client.get(f"{WORLDMONITOR_URL}/api/bootstrap")
            if resp.status_code == 200:
                data = resp.json()
                results.append(f"**WorldMonitor Intelligence Brief**\n\nBootstrap data retrieved with {len(data)} sections.")
            # Try news endpoint
            news_resp = await client.get(f"{WORLDMONITOR_URL}/api/news/top")
            if news_resp.status_code == 200:
                news = news_resp.json()
                results.append(f"\n**Top News:** {json.dumps(news[:5], indent=2, ensure_ascii=False)}")
    except httpx.ConnectError:
        results.append(f"WorldMonitor not running at {WORLDMONITOR_URL}. Start with: npm run dev")
    except Exception as e:
        results.append(f"WorldMonitor query error: {str(e)}")

    response_text = "\n".join(results) if results else f"Query: {text}\n\nWorldMonitor data unavailable."
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

