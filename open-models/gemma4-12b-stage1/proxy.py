# Forward chat requests to vLLM, add the fields of one test condition, and log every full response.
import json, os, sys, time
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

CONDITIONS = json.load(open(sys.argv[1]))
LOG_DIR = sys.argv[2]
UPSTREAM = "http://127.0.0.1:8000/v1/chat/completions"
os.makedirs(LOG_DIR, exist_ok=True)
FILES = {c: open(os.path.join(LOG_DIR, c + ".jsonl"), "a", buffering=1) for c in CONDITIONS}


async def on_start(app):
    app["session"] = ClientSession(timeout=ClientTimeout(total=None), connector=TCPConnector(limit=0))


async def on_stop(app):
    await app["session"].close()


async def chat(request):
    cond = request.match_info["cond"]
    body = await request.json()
    body.update(CONDITIONS[cond])
    t0 = time.time()
    async with request.app["session"].post(UPSTREAM, json=body) as r:
        text = await r.text()
        status = r.status
    messages = body.get("messages") or []
    # tau2 conversations resend the whole history and tool list each turn, so keep only the newest turns.
    agent_run = cond.startswith("t")
    rec = {"t0": t0, "t1": time.time(), "status": status,
           "sent": {k: v for k, v in body.items() if k not in ("messages", "logit_bias", "tools")},
           "n_messages": len(messages),
           "messages": messages[-2:] if agent_run else messages}
    try:
        reply = json.loads(text)
        rec["choices"] = reply.get("choices")
        rec["usage"] = reply.get("usage")
    except ValueError:
        rec["raw"] = text[:2000]
    FILES[cond].write(json.dumps(rec, ensure_ascii=False) + "\n")
    return web.Response(text=text, status=status, content_type="application/json")


app = web.Application(client_max_size=64 * 1024 * 1024)
app.on_startup.append(on_start)
app.on_cleanup.append(on_stop)
app.router.add_post("/{cond}/v1/chat/completions", chat)
web.run_app(app, host="127.0.0.1", port=8001, print=None)
