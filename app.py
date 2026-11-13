"""INFERENCE - Phase 1 backend (Flask + Groq API).

Run:
    pip install flask requests
    export GROQ_API_KEY="your-key"      # Codespaces: use a Codespaces secret instead
    python app.py
"""
import json
import os
import threading
import time
import uuid

import requests
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "inference_data.json")  # created at runtime
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_HISTORY = 30  # messages sent to Groq per request

SYSTEM_PROMPT = (
    "You are INFERENCE, an AI software engineer inside a developer platform. "
    "Be precise and practical. Prefer minimal changes to existing code. "
    "Give complete, runnable code in fenced blocks. If you are unsure, say so."
)

app = Flask(__name__)
lock = threading.Lock()


# ---------- storage (single JSON file, safe for one-user Codespaces use) ----------
def load():
    if not os.path.exists(DATA_FILE):
        return {"projects": {}}
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"projects": {}}


def save(data):
    tmp = DATA_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, DATA_FILE)


def summary(p):
    return {k: p[k] for k in ("id", "name", "description", "stack", "created", "updated")} | {
        "messages": len(p.get("messages", []))
    }


# ---------- routes ----------
@app.get("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/api/status")
def status():
    # Reports only whether a key exists. The key itself is never sent to the browser.
    return jsonify(
        api_key_configured=bool(os.environ.get("GROQ_API_KEY")),
        model=MODEL,
        project_count=len(load()["projects"]),
    )


@app.get("/api/projects")
def list_projects():
    projects = sorted(load()["projects"].values(), key=lambda p: p["updated"], reverse=True)
    return jsonify([summary(p) for p in projects])


@app.post("/api/projects")
def create_project():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify(error="Project name is required."), 400
    if len(name) > 80:
        return jsonify(error="Project name must be 80 characters or fewer."), 400
    now = time.time()
    project = {
        "id": uuid.uuid4().hex[:10],
        "name": name,
        "description": (body.get("description") or "").strip()[:500],
        "stack": (body.get("stack") or "").strip()[:100],
        "created": now,
        "updated": now,
        "messages": [],
    }
    with lock:
        data = load()
        data["projects"][project["id"]] = project
        save(data)
    return jsonify(summary(project)), 201


@app.get("/api/projects/<pid>")
def get_project(pid):
    p = load()["projects"].get(pid)
    if not p:
        return jsonify(error="Project not found."), 404
    return jsonify(summary(p) | {"history": p["messages"]})


@app.delete("/api/projects/<pid>")
def delete_project(pid):
    with lock:
        data = load()
        if pid not in data["projects"]:
            return jsonify(error="Project not found."), 404
        del data["projects"][pid]
        save(data)
    return jsonify(ok=True)


@app.delete("/api/projects/<pid>/messages")
def clear_chat(pid):
    with lock:
        data = load()
        p = data["projects"].get(pid)
        if not p:
            return jsonify(error="Project not found."), 404
        p["messages"] = []
        save(data)
    return jsonify(ok=True)


@app.post("/api/chat")
def chat():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return jsonify(error="GROQ_API_KEY is not set on the server. Set it and restart app.py."), 503

    body = request.get_json(silent=True) or {}
    pid = body.get("project_id")
    text = (body.get("message") or "").strip()
    if not text:
        return jsonify(error="Message is empty."), 400

    data = load()
    project = data["projects"].get(pid)
    if not project:
        return jsonify(error="Project not found."), 404

    context = SYSTEM_PROMPT + f"\n\nProject: {project['name']}"
    if project["description"]:
        context += f"\nDescription: {project['description']}"
    if project["stack"]:
        context += f"\nStack: {project['stack']}"

    history = project["messages"][-MAX_HISTORY:]
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": context}, *history, {"role": "user", "content": text}],
    }

    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        return jsonify(error=f"Could not reach the Groq API: {e.__class__.__name__}."), 502

    if r.status_code != 200:
        try:
            detail = r.json().get("error", r.text)
        except ValueError:
            detail = r.text
        if isinstance(detail, dict):
            detail = detail.get("message", str(detail))
        return jsonify(error=f"Groq API returned {r.status_code}: {str(detail)[:300]}"), 502

    try:
        reply = r.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError):
        return jsonify(error="Unexpected response format from the Groq API."), 502

    with lock:
        data = load()
        p = data["projects"].get(pid)
        if p:  # project may have been deleted while waiting
            p["messages"] += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
            p["updated"] = time.time()
            save(data)
    return jsonify(reply=reply)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
