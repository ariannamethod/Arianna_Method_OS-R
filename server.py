# flake8: noqa
# server.py — "liquid weights" server (Responses API + JSON Schema + SSE + cache)
# CHANGES (key):
# - Hardened cache (SimHash near-dup + TTL LRU) kept, minor guards.
# - Safer prompt sanitizer (truncates giant code-blocks, base64 redaction).
# - Rate limiter & headers preserved; added few small try/excepts around JSON repair.
# - Kept OpenAI Responses API usage; model pick by size/hints; SSE streaming retained.

import os
import json
import time
import uuid
import random
import logging
import threading
import hashlib
import re
from typing import Dict, Any, Callable, Generator, Tuple, Optional
from logging.handlers import RotatingFileHandler
from functools import wraps
from collections import OrderedDict

from flask import Flask, request, jsonify, Response, make_response
from openai import OpenAI

app = Flask(__name__)

LOG_DIR = os.path.join("logs", "server")
os.makedirs(LOG_DIR, exist_ok=True)
handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "server.log"),
    maxBytes=2_000_000,
    backupCount=3,
    encoding="utf-8",
)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
app.logger.setLevel(logging.INFO)
app.logger.addHandler(handler)

if not os.getenv("OPENAI_API_KEY"):
    app.logger.critical("OPENAI_API_KEY не задан — выход")
    raise RuntimeError("OPENAI_API_KEY is required")

# ────────────────────────────────────────────────────────────────────────────────
# Конфиг
# ────────────────────────────────────────────────────────────────────────────────
SECRET               = os.getenv("ARIANNA_SERVER_TOKEN", "")
MODEL_DEFAULT        = os.getenv("ARIANNA_MODEL", "gpt-4.1")
MODEL_LIGHT          = os.getenv("ARIANNA_MODEL_LIGHT", MODEL_DEFAULT)
MODEL_HEAVY          = os.getenv("ARIANNA_MODEL_HEAVY", MODEL_DEFAULT)
HEAVY_TRIGGER_TOKENS = int(os.getenv("HEAVY_TRIGGER_TOKENS", "3500"))
HEAVY_HINTS          = tuple(x.strip() for x in os.getenv("HEAVY_HINTS", "deep analysis;докажи;пошагово;reason;рефлексия").split(";") if x.strip())

PROMPT_LIMIT_CHARS   = int(os.getenv("PROMPT_LIMIT_CHARS",   "16000"))
CACHE_TTL            = int(os.getenv("CACHE_TTL_SECONDS",    "120"))
CACHE_MAX            = int(os.getenv("CACHE_MAX_ITEMS",      "256"))
SCHEMA_VERSION       = os.getenv("SCHEMA_VERSION",           "1.3")
RATE_CAPACITY        = int(os.getenv("RATE_CAPACITY",        "20"))
RATE_REFILL_PER_SEC  = float(os.getenv("RATE_REFILL_PER_SEC","0.5"))
HEARTBEAT_EVERY      = int(os.getenv("SSE_HEARTBEAT_EVERY",  "10"))
TIME_PING_SECONDS    = float(os.getenv("SSE_TIME_HEARTBEAT_SEC", "12"))
TIME_SENSITIVE_HINTS = ("now", "today", "сейчас", "сегодня", "latest", "свеж", "текущ")
CODE_BLOCK_LIMIT     = int(os.getenv("CODE_BLOCK_LIMIT", "65536"))
SIMHASH_HAMMING_THR  = int(os.getenv("SIMHASH_HAMMING_THR", "3"))

# ────────────────────────────────────────────────────────────────────────────────
# Авторизация
# ────────────────────────────────────────────────────────────────────────────────
def _extract_auth_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(None, 1)[1].strip()
    return request.headers.get("X-Auth-Token", "").strip()

def require_auth(fn: Callable):
    @wraps(fn)
    def inner(*args, **kwargs):
        if SECRET:
            token = _extract_auth_token()
            if token != SECRET:
                return jsonify({"error": "unauthorized"}), 401
        return fn(*args, **kwargs)
    return inner

# ────────────────────────────────────────────────────────────────────────────────
# Leaky-bucket rate-limiter
# ────────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    def __init__(self, capacity: int, refill_per_sec: float):
        self.capacity, self.refill = capacity, refill_per_sec
        self.state: Dict[str, Tuple[float, float]] = {}
        self.lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self.lock:
            tokens, last = self.state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            if tokens < 1.0:
                self.state[key] = (tokens, now)
                return False
            self.state[key] = (tokens - 1.0, now)
            return True

    def remaining(self, key: str) -> int:
        now = time.time()
        with self.lock:
            tokens, last = self.state.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill)
            return int(tokens)

def _client_key() -> str:
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    tok = _extract_auth_token()
    if tok:
        return tok
    return request.remote_addr or "unknown"

limiter = RateLimiter(RATE_CAPACITY, RATE_REFILL_PER_SEC)

# ────────────────────────────────────────────────────────────────────────────────
# LRU + TTL Cache (+ SimHash near-dup search)
# ────────────────────────────────────────────────────────────────────────────────
class LRUCacheTTL:
    def __init__(self, max_items: int, ttl: int, evict_cb: Callable[[str], None] | None = None):
        self.max, self.ttl = max_items, ttl
        self._od: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self.lock = threading.Lock()
        self._evict_cb = evict_cb

    def _purge_locked(self):
        now = time.time()
        removed = []
        stale = [k for k, (ts, _) in list(self._od.items()) if now - ts > self.ttl]
        for k in stale:
            self._od.pop(k, None)
            removed.append(k)
        while len(self._od) > self.max:
            k, _ = self._od.popitem(last=False)
            removed.append(k)
        for k in removed:
            if self._evict_cb:
                try:
                    self._evict_cb(k)
                except Exception:
                    pass

    def get(self, key: str):
        with self.lock:
            self._purge_locked()
            if key in self._od:
                ts, v = self._od.pop(key)
                self._od[key] = (ts, v)
                return v
            return None

    def set(self, key: str, value: Any):
        with self.lock:
            self._purge_locked()
            self._od[key] = (time.time(), value)

    def items(self):
        with self.lock:
            self._purge_locked()
            return list(self._od.items())

    def __len__(self):
        with self.lock:
            self._purge_locked()
            return len(self._od)

def _simhash64(text: str) -> int:
    s = text.lower()
    if len(s) < 3:
        grams = [s] if s else []
    else:
        grams = [s[i:i+3] for i in range(len(s) - 2)]
    v = [0] * 64
    for g in grams:
        if not g:
            continue
        h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for b in range(64):
            v[b] += 1 if (h >> b) & 1 else -1
    out = 0
    for b in range(64):
        if v[b] > 0:
            out |= (1 << b)
    return out

def _hamdist64(a: int, b: int) -> int:
    return ((a ^ b).bit_count())

_semantic_meta: Dict[str, int] = {}
cache = LRUCacheTTL(CACHE_MAX, CACHE_TTL, evict_cb=lambda k: _semantic_meta.pop(k, None))

def _semantic_cache_get_fuzzy(target_key_prefix: str, prompt: str) -> Optional[Dict[str, Any]]:
    try:
        target_sim = _simhash64(prompt)
    except Exception:
        return None
    best = None
    best_d = 65
    for k, (ts, _) in cache.items():
        if not k.startswith(target_key_prefix):
            continue
        sim = _semantic_meta.get(k)
        if sim is None:
            continue
        d = _hamdist64(sim, target_sim)
        if d < best_d and d <= SIMHASH_HAMMING_THR:
            best = k
            best_d = d
    if best:
        v = cache.get(best)
        if isinstance(v, dict):
            return v
    return None

# ────────────────────────────────────────────────────────────────────────────────
# OpenAI client
# ────────────────────────────────────────────────────────────────────────────────
def _openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    base = os.getenv("OPENAI_BASE_URL")
    return OpenAI(api_key=key, base_url=base) if base else OpenAI(api_key=key)

# ────────────────────────────────────────────────────────────────────────────────
# JSON-schema + валидация
# ────────────────────────────────────────────────────────────────────────────────
ALLOWED_MODES = {"plan", "act", "reflect", "final"}
HALT_REASONS  = {"final", "stagnation", "budget", "error", "client_closed"}

def _response_json_schema() -> Dict[str, Any]:
    return {
        "name": "AriannaJSON", "strict": True,
        "schema": {
            "type": "object", "additionalProperties": False,
            "properties": {
                "trace_id": {"type": "string"},
                "step": {"type": "integer"},
                "mode": {"type": "string", "enum": list(ALLOWED_MODES)},
                "think": {"type": "string"},
                "answer": {"type": "string"},
                "stop": {"type": "boolean"},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "halt_reason": {"type": "string", "enum": list(HALT_REASONS)},
                "action": {
                    "type": "object", "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "args": {"type": "object"}},
                    "required": ["name"]
                },
                "observation": {"type": "string"},
                "controls": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "temperature": {"type": "number"},
                        "top_p": {"type": "number"},
                        "presence_penalty": {"type": "number"},
                        "frequency_penalty": {"type": "number"}
                    }
                },
                "tokens_used": {
                    "type": "object", "additionalProperties": False,
                    "properties": {
                        "input": {"type": "integer"},
                        "output": {"type": "integer"},
                        "total": {"type": "integer"}
                    }
                }
            },
            "required": ["answer", "stop", "mode"]
        }
    }

def _truncate_large_code_blocks(text: str) -> str:
    pat = re.compile(r"```([a-zA-Z0-9_-]+)?\n([\s\S]*?)\n```", re.MULTILINE)
    def repl(m):
        lang = m.group(1) or ""
        body = m.group(2) or ""
        if len(body.encode("utf-8", "ignore")) > CODE_BLOCK_LIMIT:
            return f"```{lang}\n[CODE_BLOCK_REDACTED:{len(body)}]\n```"
        return m.group(0)
    return pat.sub(repl, text)

def _approx_tokens(text: str) -> int:
    return max(1, len(text.encode("utf-8", "ignore")) // 4)

def _pick_model(prompt: str, fallback: str) -> str:
    t = _approx_tokens(prompt)
    if t >= HEAVY_TRIGGER_TOKENS:
        return MODEL_HEAVY
    lo = prompt.lower()
    if any(h in lo for h in HEAVY_HINTS):
        return MODEL_HEAVY
    return MODEL_LIGHT or fallback

def _sanitize_prompt(prompt: str, limit: int = PROMPT_LIMIT_CHARS) -> str:
    prompt = prompt.strip()
    if len(prompt) > limit:
        prompt = prompt[:limit] + f"\n\n[truncated at {limit} chars]"
    prompt = re.sub(r"[A-Za-z0-9+/]{200,}={0,2}", "[BASE64_REDACTED]", prompt)
    prompt = _truncate_large_code_blocks(prompt)
    return prompt

def _micro_repair_json(txt: str) -> str:
    s = txt.strip()
    if s.startswith("```"):
        s = s.strip("`")
    if "{" in s:
        s = s[s.find("{"):]
    if "}" in s:
        s = s[: s.rfind("}") + 1]
    s = "".join(ch for ch in s if ch >= " " or ch == "\n")
    return s

def _coerce_and_validate(obj: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not isinstance(obj, dict):
        return {"mode": "final", "think": "", "answer": str(obj), "stop": True,
                "step": 1, "confidence": 0.5, "halt_reason": "error"}
    mode = obj.get("mode")
    out["mode"] = mode if isinstance(mode, str) and mode in ALLOWED_MODES else "final"
    out["think"] = str(obj.get("think", ""))
    out["answer"] = str(obj.get("answer", ""))
    out["stop"] = bool(obj.get("stop", out["mode"] == "final"))
    try:
        out["step"] = max(1, int(obj.get("step", 1)))
    except Exception:
        out["step"] = 1
    if isinstance(obj.get("trace_id"), str):
        out["trace_id"] = obj["trace_id"].strip()
    try:
        c = float(obj.get("confidence", 0.7))
        out["confidence"] = max(0.0, min(1.0, c))
    except Exception:
        out["confidence"] = 0.7
    hr = obj.get("halt_reason")
    if isinstance(hr, str) and hr in HALT_REASONS:
        out["halt_reason"] = hr
    if isinstance(obj.get("action"), dict) and isinstance(obj["action"].get("name"), str):
        a = {"name": obj["action"]["name"]}
        if isinstance(obj["action"].get("args"), dict):
            a["args"] = obj["action"]["args"]
        out["action"] = a
    if isinstance(obj.get("observation"), str):
        out["observation"] = obj["observation"]
    if isinstance(obj.get("controls"), dict):
        c: Dict[str, float] = {}
        for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if k in obj["controls"]:
                try:
                    c[k] = float(obj["controls"][k])
                except Exception:
                    pass
        if c:
            out["controls"] = c
    if isinstance(obj.get("tokens_used"), dict):
        try:
            tu = obj["tokens_used"]
            out["tokens_used"] = {
                "input": int(tu.get("input", 0)),
                "output": int(tu.get("output", 0)),
                "total": int(tu.get("total", 0)),
            }
        except Exception:
            pass
    return out

def _repair_prompt(base_prompt: str) -> str:
    return base_prompt + "\n\n[REPAIR] Return STRICT JSON matching the schema. No extra text."

# ────────────────────────────────────────────────────────────────────────────────
# Helpers для OpenAI Responses API
# ────────────────────────────────────────────────────────────────────────────────
def _extract_usage(resp) -> Dict[str, int]:
    usage = {"input": 0, "output": 0, "total": 0}
    try:
        d = resp.to_dict_recursive()
        u = d.get("usage") or {}
        usage["input"]  = int(u.get("input_tokens", u.get("input", 0)) or 0)
        usage["output"] = int(u.get("output_tokens", u.get("output", 0)) or 0)
        usage["total"]  = usage["input"] + usage["output"]
    except Exception:
        pass
    return usage

def _detect_red_flags(answer: str) -> list[str]:
    flags: list[str] = []
    if re.search(r"<[^/>][^>]*></[^>]+>", answer):
        flags.append("empty_tag")
    lower = answer.lower()
    if ("yes" in lower and "no" in lower) or ("да" in lower and "нет" in lower):
        flags.append("contradiction")
    return flags

def _responses_create(prompt: str, *, model: Optional[str], temperature: float, top_p: float,
                      check_flags: bool = True) -> Tuple[Dict[str, Any], Dict[str, int], Optional[str]]:
    client = _openai_client()
    resp = client.responses.create(
        model=model or MODEL_DEFAULT,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        response_format={"type": "json_schema", "json_schema": _response_json_schema()},
    )
    usage = _extract_usage(resp)
    openai_id = None
    try:
        openai_id = resp.to_dict_recursive().get("id")
    except Exception:
        pass
    content = getattr(resp, "output_text", "") or json.dumps(resp.to_dict_recursive(), ensure_ascii=False)
    try:
        obj = json.loads(content)
    except Exception:
        try:
            obj = json.loads(_micro_repair_json(content))
        except Exception:
            obj = {"mode": "final", "think": "", "answer": content, "stop": True, "step": 1, "halt_reason": "error"}
    obj = _coerce_and_validate(obj)
    if usage and "tokens_used" not in obj:
        obj["tokens_used"] = usage
    if "confidence" not in obj:
        l = len(obj.get("answer", "")) or 1
        obj["confidence"] = max(0.3, min(0.95, min(l, 800) / 800))
    if check_flags:
        flags = _detect_red_flags(obj.get("answer", ""))
        if flags:
            app.logger.warning("red flags detected: %s", flags)
            fix_prompt = (
                "Исправь ответ, убрав проблемы: "
                f"{', '.join(flags)}.\n\n" + obj.get("answer", "")
            )
            app.logger.info("sending corrective request")
            obj2, usage2, oid2 = _responses_create(
                fix_prompt, model=model, temperature=0.0, top_p=top_p, check_flags=False
            )
            app.logger.info("corrective response received")
            if usage2:
                for k, v in usage2.items():
                    usage[k] = usage.get(k, 0) + v
            if oid2:
                openai_id = oid2
            obj = obj2
    return obj, usage, openai_id

def _extract_field_partial(buf: str, field: str) -> Optional[str]:
    pat_complete = rf'"{field}"\s*:\s*"(.*?)(?<!\\)"'
    m = re.search(pat_complete, buf, re.DOTALL)
    if m:
        return m.group(1)
    pat_partial = rf'"{field}"\s*:\s*"(.*)$'
    m = re.search(pat_partial, buf, re.DOTALL)
    if m:
        return m.group(1)
    return None

def _responses_stream(prompt: str, *, model: Optional[str], temperature: float, top_p: float
                      ) -> Generator[str, None, None]:
    client = _openai_client()
    yield "retry: 10000\n\n"
    yield "event: ping\ndata: {}\n\n"
    last_emit = time.time()
    buf = ""
    field_vals = {"plan": "", "reasoning": "", "repair": ""}
    try:
        with client.responses.stream(
            model=model or MODEL_DEFAULT,
            input=prompt,
            temperature=temperature,
            top_p=top_p,
            response_format={"type": "json_schema", "json_schema": _response_json_schema()},
        ) as stream:
            events = 0
            for ev in stream:
                now = time.time()
                if now - last_emit > TIME_PING_SECONDS:
                    yield "event: ping\ndata: {}\n\n"
                    last_emit = now
                et = getattr(ev, "type", None)
                if et == "response.output_text.delta":
                    d = getattr(ev, "delta", "")
                    buf += d or ""
                    yield f"event: response.output_text.delta\ndata: {json.dumps({'delta': d})}\n\n"
                    for fld in ("plan", "reasoning", "repair"):
                        cur = _extract_field_partial(buf, fld)
                        if cur is None:
                            continue
                        prev = field_vals[fld]
                        if len(cur) > len(prev):
                            delta_txt = cur[len(prev):]
                            yield f"event: {fld}.delta\ndata: {json.dumps({'delta': delta_txt})}\n\n"
                            field_vals[fld] = cur
                    last_emit = time.time()
                elif et == "response.created":
                    rid = None
                    try:
                        rid = ev.response.id  # type: ignore
                    except Exception:
                        pass
                    yield f"event: response.created\ndata: {json.dumps({'id': rid})}\n\n"
                    last_emit = time.time()
                elif et == "response.completed":
                    txt = buf
                    try:
                        obj = json.loads(txt)
                    except Exception:
                        try:
                            obj = json.loads(_micro_repair_json(txt))
                        except Exception:
                            obj = {"mode": "final", "think": "", "answer": txt, "stop": True, "step": 1, "halt_reason": "error"}
                    obj = _coerce_and_validate(obj)
                    yield f"event: response.completed\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"
                    last_emit = time.time()
                elif et == "response.error":
                    err = str(getattr(ev, "error", None))
                    yield f"event: response.error\ndata: {json.dumps({'error': err})}\n\n"
                    last_emit = time.time()
                events += 1
                if events % HEARTBEAT_EVERY == 0:
                    yield "event: ping\ndata: {}\n\n"
                    last_emit = time.time()
    except GeneratorExit:
        return
    except Exception as e:
        obj, _, _ = _responses_create(prompt, model=model, temperature=temperature, top_p=top_p)
        yield f"event: response.completed\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"
        app.logger.exception("stream fallback to non-stream due to: %s", e)

# ────────────────────────────────────────────────────────────────────────────────
# Retry helper
# ────────────────────────────────────────────────────────────────────────────────
def with_retries(call: Callable[[], Any], *, tries: int = 3, base: float = 0.5, max_jitter: float = 0.2):
    last_exc = None
    for i in range(tries):
        try:
            return call()
        except Exception as e:
            last_exc = e
            time.sleep(base * (2 ** i) + random.uniform(0, max_jitter))
    raise last_exc  # type: ignore[misc]

# ────────────────────────────────────────────────────────────────────────────────
# Вспомогательные
# ────────────────────────────────────────────────────────────────────────────────
def _time_sensitive(prompt: str) -> bool:
    lo = prompt.lower()
    return any(k in lo for k in TIME_SENSITIVE_HINTS)

def _prompt_key(prompt: str, model: str, t: float, tp: float, ver: str) -> str:
    h = hashlib.sha256(prompt.encode("utf-8", "ignore")).hexdigest()
    return f"{ver}:{model}:{t}:{tp}:{h}"

def _log(level: int, msg: Dict[str, Any]):
    app.logger.log(level, json.dumps(msg, ensure_ascii=False))

# ────────────────────────────────────────────────────────────────────────────────
# /health
# ────────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({
        "ok": True,
        "model": MODEL_DEFAULT,
        "model_light": MODEL_LIGHT,
        "model_heavy": MODEL_HEAVY,
        "cache_items": len(cache),
        "schema_version": SCHEMA_VERSION
    }), 200

# ────────────────────────────────────────────────────────────────────────────────
# /generate (sync)
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/generate")
@require_auth
def generate():
    req_id = uuid.uuid4().hex
    key_id = _client_key()
    t0 = time.time()
    if not limiter.allow(key_id):
        r = make_response(jsonify({"error": "rate_limited", "req_id": req_id}), 429)
        r.headers["Retry-After"] = "2"
        r.headers["X-Req-Id"] = req_id
        r.headers["X-Schema-Version"] = SCHEMA_VERSION
        r.headers["X-Cache"] = "MISS"
        r.headers["X-RateLimit-Remaining"] = str(limiter.remaining(key_id))
        r.headers["X-Model"] = MODEL_DEFAULT
        return r
    try:
        payload = request.get_json(force=True) or {}
        prompt_raw = str(payload.get("prompt", ""))
        prompt = _sanitize_prompt(prompt_raw)
        if not prompt:
            r = make_response(jsonify({"error": "empty prompt", "req_id": req_id}), 400)
            r.headers["X-Req-Id"] = req_id
            r.headers["X-Schema-Version"] = SCHEMA_VERSION
            r.headers["X-Cache"] = "MISS"
            r.headers["X-RateLimit-Remaining"] = str(limiter.remaining(key_id))
            r.headers["X-Model"] = MODEL_DEFAULT
            return r

        client_model = payload.get("model")
        model = client_model or _pick_model(prompt, MODEL_DEFAULT)

        temperature = float(payload.get("temperature", 0.3))
        top_p = float(payload.get("top_p", 0.95))
        trace_id = str(payload.get("trace_id", "")) or None
        time_sensitive = _time_sensitive(prompt)

        key = _prompt_key(prompt, model, temperature, top_p, SCHEMA_VERSION)
        cached = None if (temperature > 0.4 or time_sensitive) else cache.get(key)
        if not cached and (temperature <= 0.4 and not time_sensitive):
            prefix = ":".join(key.split(":")[:-1]) + ":"
            cached = _semantic_cache_get_fuzzy(prefix, prompt)

        if cached:
            obj = cached if isinstance(cached, dict) else {}
            if (not isinstance(obj.get("answer"), str)) or (not obj.get("answer", "").strip()) or (obj.get("confidence", 1.0) < 0.5):
                cached = None

        if cached:
            latency = int((time.time() - t0) * 1000)
            r = make_response(jsonify({"response": cached, "req_id": req_id, "latency_ms": latency}), 200)
            r.headers["X-Req-Id"] = req_id
            r.headers["X-Schema-Version"] = SCHEMA_VERSION
            r.headers["X-Cache"] = "HIT"
            r.headers["X-RateLimit-Remaining"] = str(limiter.remaining(key_id))
            r.headers["X-Time-Sensitive"] = "1" if time_sensitive else "0"
            r.headers["X-Model"] = model
            return r

        def _call_once():
            obj, usage, oid = _responses_create(prompt, model=model, temperature=temperature, top_p=top_p)
            if obj.get("mode") not in ALLOWED_MODES or "answer" not in obj or not isinstance(obj.get("answer"), str):
                rep_prompt = _repair_prompt(prompt)
                obj2, usage2, oid2 = _responses_create(rep_prompt, model=model, temperature=0.0, top_p=top_p)
                if "halt_reason" not in obj2:
                    obj2["halt_reason"] = "error"
                return obj2, (usage2 or usage), (oid2 or oid)
            return obj, usage, oid

        obj, usage, openai_id = with_retries(_call_once, tries=3, base=0.4)

        if trace_id and "trace_id" not in obj:
            obj["trace_id"] = trace_id

        if (temperature <= 0.4) and (not time_sensitive) and obj.get("halt_reason") != "error":
            cache.set(key, obj)
            try:
                _semantic_meta[key] = _simhash64(prompt)
            except Exception:
                pass

        latency = int((time.time() - t0) * 1000)
        _log(logging.INFO, {
            "req_id": req_id,
            "prompt_sha": (key.split(":")[-1]),
            "meta": {k: obj[k] for k in ("mode", "step", "stop", "confidence", "halt_reason") if k in obj},
            "usage": usage, "latency_ms": latency, "openai_id": openai_id,
            "model": model
        })
        r = make_response(jsonify({"response": obj, "req_id": req_id, "latency_ms": latency}), 200)
        r.headers["X-Req-Id"] = req_id
        r.headers["X-Schema-Version"] = SCHEMA_VERSION
        r.headers["X-Cache"] = "MISS"
        r.headers["X-RateLimit-Remaining"] = str(limiter.remaining(key_id))
        r.headers["X-Time-Sensitive"] = "1" if time_sensitive else "0"
        r.headers["X-Model"] = model
        return r

    except Exception as e:
        app.logger.exception("generate failed")
        r = make_response(jsonify({"error": str(e), "req_id": req_id}), 500)
        r.headers["X-Req-Id"] = req_id
        r.headers["X-Schema-Version"] = SCHEMA_VERSION
        r.headers["X-Cache"] = "MISS"
        r.headers["X-RateLimit-Remaining"] = str(limiter.remaining(key_id))
        r.headers["X-Model"] = MODEL_DEFAULT
        return r

# ────────────────────────────────────────────────────────────────────────────────
# /generate_sse
# ────────────────────────────────────────────────────────────────────────────────
@app.post("/generate_sse")
@require_auth
def generate_sse():
    req_id = uuid.uuid4().hex
    key_id = _client_key()
    if not limiter.allow(key_id):
        body = (
            "retry: 10000\n\n"
            f"event: response.error\ndata: {json.dumps({'error':'rate_limited','req_id':req_id})}\n\n"
        )
        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Retry-After": "2",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Req-Id": req_id,
            "X-Schema-Version": SCHEMA_VERSION,
            "X-RateLimit-Remaining": str(limiter.remaining(key_id)),
            "X-Model": MODEL_DEFAULT,
        }
        return Response(body, headers=headers)
    try:
        payload = request.get_json(force=True) or {}
        prompt = _sanitize_prompt(str(payload.get("prompt", "")))
        if not prompt:
            return Response(
                f"event: response.error\ndata: {json.dumps({'error':'empty prompt','req_id':req_id})}\n\n",
                mimetype="text/event-stream"
            )
        client_model = payload.get("model")
        model       = client_model or _pick_model(prompt, MODEL_DEFAULT)
        temperature = float(payload.get("temperature", 0.3))
        top_p       = float(payload.get("top_p", 0.95))
        time_sensitive = _time_sensitive(prompt)

        def _gen():
            yield ": ready\n\n"
            try:
                for chunk in _responses_stream(prompt, model=model, temperature=temperature, top_p=top_p):
                    yield chunk
            except Exception as e:
                app.logger.exception("SSE stream failed: %s", e)
                obj, _, _ = _responses_create(prompt, model=model, temperature=temperature, top_p=top_p)
                yield f"event: response.completed\ndata: {json.dumps(obj, ensure_ascii=False)}\n\n"

        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "X-Req-Id": req_id,
            "X-Schema-Version": SCHEMA_VERSION,
            "X-Time-Sensitive": "1" if time_sensitive else "0",
            "X-RateLimit-Remaining": str(limiter.remaining(key_id)),
            "X-Model": model,
        }
        return Response(_gen(), headers=headers)
    except Exception as e:
        app.logger.exception("generate_sse failed")
        return Response(
            f"event: response.error\ndata: {json.dumps({'error':str(e),'req_id':req_id})}\n\n",
            mimetype="text/event-stream"
        )

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
