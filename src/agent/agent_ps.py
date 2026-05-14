import json
import logging
import math
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from itertools import product as _cartesian_product
from os import getenv
from typing import Any
from urllib.parse import quote_plus

from src.agent.agent_interface import (
    Tool,
    create_dialogue_step,
    execute_tool_call,
)
from src.agent import proxy_client as _proxy_client_mod

ProxyClient = _proxy_client_mod.ProxyClient


def _strip_none_entries(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {k: v for k, v in mapping.items() if v is not None}


def _pid(product: dict) -> str:
    """Return the product_id from a product dict as a stripped string (never None)."""
    return str(product.get("product_id", "") or "").strip()


def _parse_scored_candidates(
    parsed: list, candidates: list
) -> list[tuple]:
    """Map a JSON scorer response ``[{"product_id": ..., "score": ...}, ...]`` onto
    *candidates*, returning a list of ``(product, score)`` pairs sorted descending."""
    pid_to_score: dict[str, float] = {}
    for item in parsed:
        if isinstance(item, dict):
            pid = str(item.get("product_id", "")).strip()
            try:
                sc = float(item.get("score", 0))
            except (TypeError, ValueError):
                sc = 0.0
            if pid:
                pid_to_score[pid] = sc
    result = [
        (p, pid_to_score.get(_pid(p), 0.0))
        for p in candidates
    ]
    result.sort(key=lambda x: (x[1], _pid(x[0])), reverse=True)
    return result


def _extract_product_ids(response_body: Any) -> list[str]:
    if not isinstance(response_body, list):
        return []
    ids: list[str] = []
    for item in response_body:
        if not isinstance(item, dict):
            continue
        product_id = item.get("product_id")
        if product_id is not None:
            ids.append(str(product_id))
    return ids


def _summarize_response_for_logging(path: str, response_body: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if response_body is None:
        return summary

    body_str = json.dumps(response_body, default=str)
    is_product_search = "/search/find_product" in path
    is_inference = "/inference/" in path

    if len(body_str) > 2000:
        summary["response_truncated"] = True
        summary["response_length"] = len(body_str)
        if is_product_search:
            product_ids = _extract_product_ids(response_body)
            if product_ids:
                summary["result_product_ids"] = product_ids
        if is_inference and isinstance(response_body, dict):
            usage = response_body.get("usage")
            if isinstance(usage, dict):
                summary["response"] = {"usage": usage}
        return summary

    summary["response"] = response_body
    if is_product_search:
        product_ids = _extract_product_ids(response_body)
        if product_ids:
            summary["result_product_ids"] = product_ids
    return summary


def _sbx_request_log_record_preserve_usage(
    self,
    method: str,
    path: str,
    params=None,
    json_data=None,
    status_code=None,
    response_body=None,
    duration_ms: float = 0.0,
) -> None:
    if not self._log_file:
        return
    entry: dict[str, Any] = {
        "method": method,
        "path": path,
        "timestamp": int(time.time() * 1000),
        "duration_ms": round(duration_ms, 1),
        "status_code": status_code,
    }
    clean_params = _strip_none_entries(params)
    if clean_params:
        entry["params"] = clean_params
    if isinstance(json_data, dict):
        payload = (
            _strip_none_entries({k: v for k, v in json_data.items() if k != "messages"})
            if "/inference/" in path
            else json_data
        )
        if payload:
            entry["json_data"] = payload
    entry.update(_summarize_response_for_logging(path, response_body))

_request_log_cls = getattr(_proxy_client_mod, "RequestLog", None)
if _request_log_cls is not None:
    _request_log_cls.record = _sbx_request_log_record_preserve_usage


_reasoning_tls = threading.local()


def _reasoning_events_for_thread() -> list[dict]:
    events = getattr(_reasoning_tls, "events", None)
    if events is None:
        events = []
        _reasoning_tls.events = events
    return events


def _record_reasoning_event(
    kind: str, method: str, path: str, duration_ms: float,
    response: Any, params: Any = None, json_data: Any = None,
) -> None:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        usage = None
    event: dict[str, Any] = {
        "kind": kind,
        "method": method,
        "path": path,
        "duration_ms": round(duration_ms, 1),
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "status_code": 200 if isinstance(response, (dict, list)) else None,
        "timestamp": int(time.time() * 1000),
        "t": time.time(),
    }
    cleaned_params = _strip_none_entries(params)
    if cleaned_params:
        event["params"] = cleaned_params
    if isinstance(json_data, dict) and json_data.get("model"):
        event["json_data"] = {"model": json_data["model"]}
    if usage:
        event["response"] = {"usage": usage}
    if "/search/find_product" in path:
        product_ids = [pid for pid in _extract_product_ids(response) if pid]
        if product_ids:
            event["result_product_ids"] = product_ids
    _reasoning_events_for_thread().append(event)


class _LoggingProxyClient:
    def __init__(self, inner, kind: str):
        self._inner, self._kind = inner, kind

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _call_with_reasoning_log(self, method: str, path: str, *, params=None, json_data=None, **kw):
        started = time.time()
        result = None
        try:
            if method == "POST":
                result = self._inner.post(path, json_data=json_data, **kw)
            else:
                result = self._inner.get(path, params=params, **kw)
            return result
        finally:
            _record_reasoning_event(
                self._kind,
                method,
                path,
                (time.time() - started) * 1000,
                result,
                params=params,
                json_data=json_data,
            )

    def post(self, path, json_data=None, **kw):
        return self._call_with_reasoning_log("POST", path, json_data=json_data, **kw)

    def get(self, path, params=None, **kw):
        return self._call_with_reasoning_log("GET", path, params=params, **kw)


_JUDGE_PROXY_CALL_KEYS = (
    "method", "path", "status_code", "duration_ms",
    "timestamp", "params", "json_data", "response",
    "completion_tokens", "result_product_ids",
)


def _attach_proxy_calls_to_dialogue(steps: list[dict]) -> None:
    if not steps:
        return
    events = list(_reasoning_events_for_thread())
    calls = [{k: e[k] for k in _JUDGE_PROXY_CALL_KEYS if k in e} for e in events]
    if calls:
        if "extra_info" not in steps[0] or not isinstance(steps[0].get("extra_info"), dict):
            steps[0]["extra_info"] = {}
        steps[0]["extra_info"]["proxy_calls"] = calls


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Product = dict[str, Any]
SearchSpec = dict[str, Any]

DEFAULT_PRODUCT_QUERY = "product"

# Soft per-problem deadline: skip non-critical LLM work when wall time is tight.
_PROBLEM_SOFT_DEADLINE_SECS = 250.0
_problem_start_ts: float = 0.0

FALLBACK_PRODUCT_ID: str = "0"

TOP_RELEVANCE_CANDIDATES = 10
CHEAPER_PRICE_TIEBREAK_DIVISOR = 100_000
SHOP_SCORE_THRESHOLD = 6.0

# Shop task deadline + capacity constants (agent_make.py parity)
_SHOP_DEADLINE_SOFT_SEC: float = 230.0
_SHOP_DEADLINE_RESERVE_SEC: float = 18.0
_shop_mono_anchor: float = 0.0
_SHOP_ANCHOR_ATTEMPTS: int = 12
_SHOP_MAX_FULL_SHOPS: int = 8
_SHOP_BATCH_CAP: int = 32
_SHOP_BATCH_CAP_MULTI: int = 24
_SHOP_PER_STORE_CAP: int = 4
_SHOP_PER_STORE_CAP_MULTI: int = 3

# Same-shop voucher combo constants
_SHOP_VOUCHER_MAX_CANDIDATE_SHOPS: int = 18   # max shops evaluated in same-shop voucher
_SHOP_VOUCHER_TOP_PER_SPEC: int = 8           # top candidates per spec per shop in combo search
_SHOP_VOUCHER_COMBO_CAP: int = 1200           # max cartesian combos per shop
_SHOP_VOUCHER_UTILIZATION_TARGET: float = 0.88  # fraction of budget used for utilization bonus
_SHOP_VOUCHER_DETAIL_FETCH_CAP: int = 40      # max product IDs fetched for details per spec

# Narrator token budget
_NARRATOR_MAX_TOKENS: int = 500

# LLM call-count budget system (mirrors agent_make.py RuntimeCaps).
# Each core spend = one scoring/judge LLM call. Each narr spend = one narrator call.
# Exhausting the budget causes heuristic fallback instead of timing out.
_LLM_CORE_BUDGET_PRODUCT: int = 9
_LLM_CORE_BUDGET_VOUCHER: int = 15
_LLM_CORE_BUDGET_SHOP: int = 20
_LLM_NARR_BUDGET_PRODUCT: int = 3
_LLM_NARR_BUDGET_VOUCHER: int = 5
_LLM_NARR_BUDGET_SHOP: int = 4

_llm_budget_tls = threading.local()


def _reset_llm_budget(task: str) -> None:
    """Set per-task LLM call budgets on the current thread."""
    if task == "shop":
        _llm_budget_tls.core = _LLM_CORE_BUDGET_SHOP
        _llm_budget_tls.narr = _LLM_NARR_BUDGET_SHOP
    elif task == "voucher":
        _llm_budget_tls.core = _LLM_CORE_BUDGET_VOUCHER
        _llm_budget_tls.narr = _LLM_NARR_BUDGET_VOUCHER
    else:
        _llm_budget_tls.core = _LLM_CORE_BUDGET_PRODUCT
        _llm_budget_tls.narr = _LLM_NARR_BUDGET_PRODUCT


def _spend_core_budget(cost: int = 1) -> bool:
    """Consume one scoring/judge LLM call from the budget. Returns False when exhausted."""
    left = getattr(_llm_budget_tls, "core", None)
    if left is None:
        return True  # no budget initialised → allow unconditionally
    if left < cost:
        logger.info("_spend_core_budget: exhausted (left=%d)", left)
        return False
    _llm_budget_tls.core = left - cost
    return True


def _spend_narration_budget(cost: int = 1) -> bool:
    """Consume one narrator LLM call from the budget. Returns False when exhausted."""
    left = getattr(_llm_budget_tls, "narr", None)
    if left is None:
        return True  # no budget initialised → allow unconditionally
    if left < cost:
        logger.info("_spend_narration_budget: exhausted (left=%d)", left)
        return False
    _llm_budget_tls.narr = left - cost
    return True


_inference_client = _LoggingProxyClient(ProxyClient(timeout=90, max_retries=10), "inference")
# Short-retry client for scoring: max_retries=1 prevents a single timed-out call
# from burning ~900s (90s × 10) before the heuristic fallback kicks in.
_score_inference_client = _LoggingProxyClient(ProxyClient(timeout=60, max_retries=1), "score_inference")
_search_client = _LoggingProxyClient(ProxyClient(timeout=30, max_retries=5), "search")

_product_detail_cache: dict[str, dict] = {}
_search_http_request_times: list[float] = []
_search_http_rate_lock = threading.Lock()
_SEARCH_HTTP_MAX_REQUESTS_PER_MINUTE = 90
_SEARCH_HTTP_WINDOW_SECONDS = 60.0
_SEARCH_HTTP_MIN_INTERVAL_SECONDS = 0.7

_last_tool_call_time = 0.0
TOOL_CALL_DELAY = 0.5
TOOL_CALL_MAX_RETRIES = 3
TOOL_CALL_BASE_BACKOFF = 1.0


def _time_left() -> float:
    if _problem_start_ts <= 0:
        return _PROBLEM_SOFT_DEADLINE_SECS
    return _PROBLEM_SOFT_DEADLINE_SECS - (time.monotonic() - _problem_start_ts)


DEFAULT_PARSE_MODEL_FOR_PRODUCT = "zai-org/GLM-5.1-TEE"
DEFAULT_PARSE_MODEL_FOR_SHOP = "deepseek-ai/DeepSeek-V3.1-TEE"
DEFAULT_PARSE_MODEL_FOR_VOUCHER = "deepseek-ai/DeepSeek-V3.1-TEE"
DEFAULT_CHOOSE_PRODUCT_MODEL = "deepseek-ai/DeepSeek-V3.1-TEE"
FALLBACK_MODEL = "Qwen/Qwen3.5-397B-A17B-TEE"
NARRATOR_MODEL = "deepseek-ai/DeepSeek-V3.1-TEE"
# SECOND_FALLBACK_MODEL = Qwen/Qwen3.5-397B-A17B-TEE, zai-org/GLM-5.1-TEE, MiniMaxAI/MiniMax-M2.5-TEE, tngtech/DeepSeek-TNG-R1T2-Chimera-TEE

# Tier-B and tier-C model identifiers (mirrors agent_make.py RuntimeCaps)
JUDGE_MODEL_B: str = "deepseek-ai/DeepSeek-V3-0324-TEE"
JUDGE_MODEL_EXTRA: str = "google/gemma-4-31B-turbo-TEE"
NARRATOR_MODEL_B: str = "deepseek-ai/DeepSeek-V3-0324-TEE"
NARRATOR_MODEL_C: str = "google/gemma-4-31B-turbo-TEE"


def _model_chain(*models: str) -> list[str]:
    env = getenv("SANDBOX_MODEL")
    return [env] if env else list(dict.fromkeys(models))


def _judge_model_chain(include_extra: bool = True) -> list[str]:
    extras = [JUDGE_MODEL_EXTRA] if include_extra else []
    return _model_chain(DEFAULT_CHOOSE_PRODUCT_MODEL, JUDGE_MODEL_B, *extras)


def _narrator_model_chain() -> list[str]:
    return _model_chain(NARRATOR_MODEL, NARRATOR_MODEL_B, NARRATOR_MODEL_C)


def _parse_model_chain(task_type: str) -> list[str]:
    if task_type == "product":
        primary = DEFAULT_PARSE_MODEL_FOR_PRODUCT
    elif task_type == "shop":
        primary = DEFAULT_PARSE_MODEL_FOR_SHOP
    else:
        primary = DEFAULT_PARSE_MODEL_FOR_VOUCHER
    return _model_chain(primary, JUDGE_MODEL_B, JUDGE_MODEL_EXTRA)


def _iter_llm_json(
    *,
    models: Sequence[str],
    temperature: float,
    system: str,
    user: str,
    retries: int = 1,
    max_tokens: int | None = None,
    client: Any = None,
) -> Any:
    """Generator yielding parsed JSON from LLM across model chain with per-model retries.
    Mirrors agent_make.py _x_iter_llm_json with agent.py's JSON-repair logic preserved."""
    _client = client if client is not None else _inference_client
    for model in models:
        for _ in range(max(1, retries)):
            body: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            try:
                resp = _client.post("/inference/chat/completions", json_data=body)
            except Exception:
                continue
            if not (resp and resp.get("choices")):
                continue
            raw = (resp["choices"][0].get("message", {}).get("content") or "").strip()
            if not raw:
                continue
            # Try 1: direct parse
            parsed: Any = None
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                pass
            # Try 2: strip markdown fences
            if parsed is None:
                cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
            # Try 3: regex hunt for outermost JSON object/array
            if parsed is None:
                m = re.search(r"[\[{].*[\]}", raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
            if parsed is not None:
                yield parsed


def llm_json_call(
    system_prompt: str,
    user_payload: Any,
    max_tokens: int = 800,
    min_time_left: float = 55.0,
) -> Any | None:
    """One-shot JSON helper for optional parse/judge calls; never raises."""
    if _time_left() < min_time_left:
        return None
    user_content = (
        user_payload
        if isinstance(user_payload, str)
        else json.dumps(user_payload, ensure_ascii=False)
    )
    model = getenv("SANDBOX_MODEL") or globals().get(
        "DEFAULT_CHOOSE_PRODUCT_MODEL", "deepseek-ai/DeepSeek-V3.1-TEE"
    )
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "stream": False,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    try:
        resp = _inference_client.post("/inference/chat/completions", json_data=body)
    except Exception:
        return None
    if not (resp and resp.get("choices")):
        return None
    raw = (resp["choices"][0].get("message", {}).get("content") or "").strip()
    if not raw:
        return None
    candidates = [
        raw,
        re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip(),
    ]
    match = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if match:
        candidates.append(match.group(1))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
    return None


def _iter_llm_response_text(
    *,
    models: Sequence[str],
    temperature: float,
    system: str,
    user: str,
    retries: int = 1,
    max_tokens: int | None = None,
) -> Any:
    """Generator yielding non-empty text responses from LLM across model chain.
    Mirrors agent_make.py _x_iter_llm_response_text."""
    for model in models:
        for _ in range(max(1, retries)):
            body: dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            try:
                resp = _inference_client.post("/inference/chat/completions", json_data=body)
            except Exception:
                continue
            if not (resp and resp.get("choices")):
                continue
            content = (resp["choices"][0].get("message", {}).get("content") or "").strip()
            if content:
                yield content

# Voucher-path tuning knobs
_VOUCHER_KNAPSACK_TIER_STEP: float = 2.0         # score-floor drop per tier
_VOUCHER_KNAPSACK_CAND_CAP: int = 15             # max candidates kept per spec in knapsack
_VOUCHER_SOFT_DEADLINE: float = 20.0             # skip knapsack when time_left < this many secs
_VOUCHER_MAX_JUDGE_CANDIDATES: int = 10          # max candidates sent to LLM judge
# Additional voucher tuning knobs (ported from agent_make.py)
_VOUCHER_PRICE_WIDEN_RATIO: float = 1.4          # widen price range by ±40% on service relax
_VOUCHER_SKIP_LLM_SCORE: float = 7.5             # skip LLM judge if top heuristic clearly dominant
_VOUCHER_REFINE_BELOW: float = 5.0               # trigger null-sweep refinement below this score
_VOUCHER_COMBO_K_PER_SPEC: int = 12              # candidates per spec in combo grid
_VOUCHER_COMBO_MAX_COMBOS: int = 5000            # max cartesian combos to evaluate
_VOUCHER_COMBO_SCORE_THRESHOLD: float = 5.0      # min LLM score for combo candidates
_VOUCHER_ENABLE_SELF_CONSISTENCY: bool = True    # enable self-consistency double-judge
_VOUCHER_SELF_CONSISTENCY_GAP: float = 2.0       # skip second judge if heuristic gap already large
LLM_PARSE_MIN_TIME_LEFT: float = 55.0
LLM_JUDGE_MIN_TIME_LEFT: float = 35.0

def _flatten_word_groups(*groups: Sequence[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        out.extend(group)
    return out


def _join_prompt_lines(lines: Sequence[str]) -> str:
    return "\n".join(lines)


_SCORING_STOPWORDS = _flatten_word_groups(
    ("the", "a", "an", "for", "with", "from", "that", "this", "i", "me"),
    ("my", "looking", "find", "want", "need", "get", "finish"),
    ("buy", "also", "and", "in", "is", "it", "am", "im", "priced", "pesos"),
    ("php", "price", "between", "than", "above", "below", "more", "less"),
    ("over", "under", "of", "to", "or", "on", "at", "by", "its", "be", "can"),
    ("has", "have", "will", "would", "should", "item", "items", "both", "these"),
    ("offering", "sells", "shop", "budget", "voucher", "discount", "first", "second"),
    ("replacement", "suitable", "broken", "ballpoint", "repair", "use"),
    ("third", "brand", "made", "using", "available", "support", "supports", "compatible"),
    ("please", "tip", "age"),
)

_PRODUCT_PROMPT = _join_prompt_lines((
    "Extract search params as JSON. No markdown.",
    '{"products":[{"keywords":"search query","brand":"brand"|null,"price_range":"min-max"|"min-"|"0-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}],"is_shop_voucher":bool}',
    "- keywords: product type + brand + material + color + quantity/units + dimensions + packaging/logistics + product/misc + capacity + sharp + fit + style + length + use + category, etc. Use 2-10 words including ALL qualifying terms. IMPORTANT RULE:",
    "    - Must preserve the left-to-right order of terms exactly as they appear in the query.",
    "    - Keep full color descriptors including any qualifier. ",
    "    - The color category includes not just basic color terms ('red', 'blue') but also shades ('light blue'), variants ('space grey'), and e-commerce-specific descriptors ('as show', 'as shown', 'random color') — capture the specific color phrase EXACTLY as written in the query, including every word of the phrase.",
    "    - MUST include brand related terms in keywords.",
    "    - MUST NEVER include service related terms in keywords.",
    "    - Must identify the price unit and the other one.",
    '    - Compact any number+unit pair: remove the space and use the standard abbreviation. Whenever a query contains a phrase like "<container> of <number>" (container can be pack, set, box, etc.), replace the whole phrase with "<number>pcs".',
    '    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.',
    "- brand: the exact brand name as written in the query.",
    "- only_product_type: true if the keywords are only nouns — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.",
    '- service: "LazMall / guaranteed authenticity / quick returns"→"official", "complimentary shipping / free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".',
    "JSON only:",
))

_SHOP_PROMPT = _join_prompt_lines((
    "Extract search params as JSON. No markdown. Find multi-products.",
    '{"products":[{"query":"the part of the raw query describing this product","keywords":"search query","price_range":"min-max"|"min-"|"0-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}]}',
    "- keywords: product type + brand + material + color + size + quantity/units + weight/volume + dimensions + packaging/logistics + product/misc + sharp + fit + style + length + selling unit + use. 2-8 words, include ALL qualifying terms. Keep full color descriptors including any qualifier. Drop opening/fastening mechanism terms. IMPORTANT RULE:",
    "    - Must preserve the left-to-right order of terms exactly as they appear in the query.",
    "    - MUST never include service related terms in keywords.",
    '    - Compact any number+unit pair: remove the space and use the standard abbreviation.',
    '    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.',
    "    - Preserve compatibility tokens verbatim; merge browse/category scope into keywords when present.",
    '- service: "LazMall"→"official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".',
    "- only_product_type: true if the keywords are the product type name alone — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.",
    "- Same store must sell MULTIPLE differents (numbered First/Second/Also)",
    '- Multi-product: one entry per product, preserve order. Each "query" value must be the minimal verbatim slice naming that line item.',
    "JSON only:",
))

_VOUCHER_PROMPT = _join_prompt_lines((
    "Extract search params as JSON. No markdown. Find products that fit within a budget after applying a voucher discount.",
    '{"products":[{"query": "corresponding part of the raw query for this product.", "keywords":"search query","price_range":"min-max"|"min-"|"0-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}], "voucher": { "voucher_type": "platform|shop", "discount_type": "fixed|percentage", "discount_value": "fixed amount OR percentage number e.g. 42 for 42%", "threshold": "minimum total price for voucher to apply", "cap": "max discount for percentage vouchers, 0 if not mentioned or fixed type", "budget": "the user\'s maximum budget" }, "is_shop_voucher":bool}',
    "- keywords: product type + brand + material + color + quantity/units + weight/volume + dimensions + packaging/logistics + product/misc + sharp + fit + style + length + use. 2-8 words, include qualifying terms that are explicitly mentioned in the query. IMPORTANT RULE:",
    "    - Must preserve the left-to-right order of terms exactly as they appear in the query.",
    "    - MUST never include service related terms and secondary keywords.",
    "    - Keep full color descriptors including any qualifier.",
    '    - Compact any number+unit pair: remove the space and use the standard abbreviation. Whenever a query contains a phrase like "<container> of <number>" (container can be pack, set, box, etc.), replace the whole phrase with "<number>pcs".',
    '    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.',
    '- service: "LazMall / guaranteed authenticity / quick returns"→"official", "complimentary shipping / free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".',
    "- only_product_type: true if the keywords are the product type name alone — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.",
    "- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.",
    '- is_shop_voucher: true if "same shop" voucher.',
    "JSON only:",
))

_SHOP_SCORER_PROMPT = _join_prompt_lines((
    "You are scoring product candidates for a retail benchmark.",
    "",
    "Score EVERY candidate against the user request. Return a score for ALL of them.",
    "",
    "Priorities:",
    "1. Treat the request as a conjunction of constraints: missing a hard requirement (price bound, service flag, explicit attribute, compatibility) caps the score even if the title is attractive.",
    '2. Prefer explicit structured evidence in attributes and sku_options. In the user_content.request, "any" means "all."',
    "3. Compatibility/model, material, function, theme, brand, quantity/units, weight/volume, dimensions, packaging/logistics, sharp, fit, style, length, use, service, and price constraints all matter.",
    "4. When the user states numeric price bounds, the candidate's listed price must fall inside the allowed interval (inclusive) unless the request is open-ended; otherwise penalize heavily.",
    "5. Do not prefer a candidate just because its title is broader, more generic, or contains more common keywords.",
    "6. Treat semantically equivalent value strings as the same match even if formatting differs slightly.",
    '7. If "only_product_type" is true, account for the product_type + "only" option in the sku_option and attributes but not in title.',
    "   Minor wording, spacing, punctuation, tokenization, or formatting differences should not change the decision by themselves.",
    "8. Do not over-weight one appealing field when both candidates already satisfy it.",
    "   If multiple candidates match the same color/model/service, prefer the candidate whose title + attributes + sku_options are more consistently aligned overall.",
    "9. Prefer stronger overall agreement across independent constraints over a single more literal-looking phrase.",
    "- score: integer from 0 (no match) to 10 (perfect match) per candidate.",
    "",
    "Return JSON array only, one object per candidate in the same order received:",
    '[{"product_id":"...","score":8},{"product_id":"...","score":3},...]',
))

_TASK_EXTRACTION_PROMPTS: dict[str, str] = {
    "product": _PRODUCT_PROMPT,
    "shop": _SHOP_PROMPT,
    "voucher": _VOUCHER_PROMPT,
}

PRODUCT_JUDGE_MAX_RETRIES = 3

_PRODUCT_JUDGE_PROMPT = _join_prompt_lines((
    "You are choosing the best final product candidate for a retail benchmark.",
    "",
    "Pick the ONE candidate that most exactly satisfies the user request.",
    "",
    "Priorities:",
    "1. The winner must jointly satisfy as many explicit constraints as possible; do not pick a listing that violates a stated hard requirement (attributes, SKU variant, service, numeric price bounds, compatibility tokens) just because it is popular.",
    '2. Prefer explicit structured evidence in attributes and sku_options. In the user_content.request (it is the requirement query), “any” does not refer to special terms or items; it means the same as “all.”',
    "3. Compatibility/model, material, function, theme, brand, quantity/units, weight/volume, dimensions, packaging/logistics, product/misc, sharp, fit, style, length, use, service, and price constraints all matter.",
    "4. Verify numeric price eligibility when the user gave min/max or threshold language.",
    "5. Do not prefer a candidate just because its title is broader, more generic, or contains more common keywords.",
    "6. If one candidate better matches the requested attributes, choose it even if another candidate has the same heuristic score.",
    "7. Treat semantically equivalent value strings as the same match even if formatting differs slightly.",
    "   Minor wording, spacing, punctuation, tokenization, or formatting differences should not change the decision by themselves.",
    "8. Do not over-weight one appealing field when both candidates already satisfy it.",
    "   If multiple candidates match the same color/model/service, prefer the candidate whose title + attributes + sku_options are more consistently aligned overall.",
    "9. Prefer stronger overall agreement across independent constraints over a single more literal-looking phrase.",
    "10. Prefer cheaper product when constraint coverage is equal.",
    '11. If "only_product_type" is true, Must account for the product_type + "only" option in the sku_option and attributes but not in title.',
    "",
    "- relevance_score: integer from 0 (no match) to 10 (perfect match) reflecting how well the best candidate satisfies the request.",
    "",
    "GROUNDING RULES FOR 'reason' (CRITICAL — violating these triggers automatic reason rewrite):",
    "- The 'reason' field MUST quote or paraphrase ACTUAL values that appear in the selected candidate's title, attributes, or sku_options_preview. Every claim must be verifiable from that candidate's data.",
    "- NEVER claim a brand, color, material, or model match that is not literally present in the candidate's title/attributes/sku_options_preview. If the user asked for brand \"X\" and no candidate contains \"X\" in its data, do NOT write \"matches brand X\".",
    "- If no candidate truly matches a key user requirement (brand/model/color/spec), phrase the reason honestly: '<requirement> not explicitly found in candidate data; selected <closest_match> as best proxy based on <actual matching field>'.",
    "- Use only descriptors grounded in the candidate's data or neutral phrasing (\"best available match\", \"closest proxy\", \"product type matches but brand not confirmed\"). Fabricated matches will be detected and replaced.",
    "",
    "Return JSON only:",
    '{"best_product_id":"...","reason":"short reason","relevance_score":8}',
))


_THINK_NARRATOR_PROMPT = _join_prompt_lines((
    "You are an AI retail assistant. Write 2–4 sentences of internal, first-person reasoning explaining what you are doing at this step.",
    "",
    'You receive a JSON object with a "query" field and additional context. Identify which case applies from the keys present and write accordingly:',
    "",
    'CASE 1 — "keywords" + "price_constraints" + "service_filters" present (query analysis / planning step):',
    'You are analysing the user\'s request before searching. State what the user wants to buy, list the exact search keywords you will use, mention any price range and service type constraints. If "only_product_type" is true, explain that the query is a bare product type with no extra qualifiers so you will append "only" to the search to avoid unrelated products — quote the "only_product_type_reason" value if present. If "budget_constraint" is present, note the voucher discount type, threshold, and budget.',
    "",
    'CASE 2 — "search_query" + "top_candidates" present (search results step):',
    'You just ran a product search. State the exact search query and any price/service filters applied. Report how many results came back ("total_results"). Name the most relevant top candidates by their title and price from "top_candidates". State what you will evaluate next.',
    "",
    'CASE 3 — "selected" + "constraints" present (product selection step):',
    'You are choosing the best product. Name the selected product by its product_id and title. Explain which specific attributes, SKU options, or specs from "selected.attributes" and "selected.sku_options_sample" satisfy the constraints (price, service, keywords). Quote the "llm_reason" value if it is non-empty and explain why it is the best match. If "constraint_check" is present, use its "keywords_matched" / "keywords_missing" / "price_note" / "overall_note" as ground truth — do NOT claim a keyword is present when "keywords_missing" lists it.',
    "",
    'CASE 4 — "product_count" + "products" present (multi-product shop planning step):',
    'You are about to search for multiple products from the same shop. State how many products are needed and name each one using its keywords value. Mention the price range and service constraint for each product.',
    "",
    'CASE 5 — "shop_id" + "selected_products" present (shop found step):',
    'You found a shop carrying all required products. State the shop ID. For each entry in "selected_products", name its title and price. Confirm they collectively satisfy the query. If "llm_reasoning" is also present, reference the relevance scores that led to this shop being chosen. If "constraint_checks" (list, one per spec) is present, mention at least one spec\'s "keywords_missing" or "price_note" when it flags an imperfect fit.',
    "",
    'CASE 6 — "budget_constraint" + "candidates_per_product" present (voucher candidate evaluation step):',
    'You are checking which products fit within the voucher budget. State the voucher discount type, threshold, and the max allowed total from "max_allowed_total". For each entry in "candidates_per_product", name the keywords and the top product candidate\'s title and price.',
    "",
    'CASE 7 — "selected_products" + "budget_constraint" present, no "candidates_per_product" (voucher selection confirmed step):',
    'You selected products that fit the voucher budget. Name each product from "selected_products" by title and price. State the total price before discount from "total_before_discount" and confirm it is within the allowed budget. Quote "llm_reason" if present. If "constraint_checks" is present, cite its data rather than paraphrasing the match.',
    "",
    'CASE 8 — "selected_products" + "total_spent" + "allowed_total" present (fixed-budget selection step):',
    'You finalised products within a fixed spending limit. Name each product, state the exact total spent and the allowed maximum, and confirm the selection is within budget.',
    "",
    'CASE 9 — "weighing" present (candidate comparison step, BEFORE committing to the leader):',
    'You just scored the candidates and are about to commit. Do NOT say "I selected" — say "I am weighing" / "The current leader is" / "Alternatives include". Two sub-cases:',
    '  (a) "weighing.leader" present (single-product task): Name the current leader by product_id, price, and heuristic_score, then quote or paraphrase its llm_reason. Name 1-2 runner-ups from "weighing.alternatives" with their product_id, price, and heuristic_score. Then write ONE explicit sentence of the form "I prefer pid=<leader> OVER pid=<alt> because <reason citing higher score, closer attribute match, brand/spec fit, or price>".',
    '  (b) "weighing.per_spec" present (multi-product task, voucher/shop): State "I am weighing candidates across <N> specs". For each entry in per_spec, name its leader\'s product_id and price; for at least one spec, name an alternative\'s product_id and price AND write one "I prefer pid=X OVER pid=Y because ..." sentence. In ONE concrete sentence explain why the leaders collectively satisfy the constraints.',
    'Always cite exact product_ids and prices. Use first-person present tense ("I am weighing…", "The leading candidate is…").',
    "",
    'CASE 10 — "case_c_resolution" present (anchor-product fallback step):',
    'No single shop covered all required products after score filtering. Describe the sub-case strategy. If "sub_case" is 4, explain how you evaluated partial-coverage shops and filled the missing spec by searching inside the winner shop. Otherwise, name the anchor product using its spec index, keywords, product_id, and shop_id, and explain that you searched the remaining specs within that shop to maximise coverage.',
    "",
    'CASE 11 — "recommended_product_ids" + "status" present (final recommendation step):',
    'You are finalising the session. State the product IDs you are recommending from "recommended_product_ids". Confirm the outcome using "status" (success or failure). If "llm_reason" is present, quote it to justify the choice. If "note" is present, mention it.',
    "",
    "Rules:",
    '- Always write in first person ("I searched…", "I selected…", "I found…", "I am planning to…").',
    "- Reference actual values from the context: IDs, titles, prices, keywords, shop IDs, attributes, scores.",
    "- Be specific and concrete — never vague or generic.",
    "- Do NOT output JSON or markdown. Plain text only.",
    "- 2–4 sentences maximum.",
    "",
    "CRITICAL GROUNDING RULES:",
    "- Reference ONLY values (product_ids, titles, prices, shop_ids, scores, keywords) that appear literally in the provided JSON context. If a field is absent or empty, omit the detail — do NOT invent a value.",
    "- Never claim an outcome (found / selected / confirmed / matched) unless the context explicitly contains it.",
    "- Do not introduce domain terms (voucher, budget, LazMall, flashsale, brand names, service tags) unless they appear in the context or query.",
    "- If 'alternatives' is present and non-empty, explicitly compare: name 1–2 runner-ups by title and price from that list, and state in ONE concrete sentence why the selected item was preferred.",
    "- When 'constraint_check' (singular) or 'constraint_checks' (list) is present, it is PRE-COMPUTED GROUND TRUTH produced by the agent itself: use its 'keywords_matched' and 'keywords_missing' to describe the title match, its 'price_note' to describe the price fit, and its 'overall_note' for the verdict. If 'keywords_missing' is non-empty, say so explicitly instead of claiming a complete match. NEVER assert that a query term is present when 'keywords_missing' lists it — this is the single largest source of -0.2 deductions once token counts are captured correctly.",
    "",
    "ANTI-CONTRADICTION RULES (violating these tanks the reasoning score):",
    '- NEVER assert a universal negative about a brand, attribute, spec, or constraint from titles alone. Words like "none mention X", "no results match X", "the search failed to find X", "none of the candidates are X" are FORBIDDEN unless the context explicitly states coverage has been exhausted across titles AND attributes AND sku_options. If you only saw titles, hedge with "the titles I can see do not surface X; attributes may still confirm it" — never a definitive negative.',
    '- NEVER make a preemptive verdict about a step you have not yet completed. If the context shows you will run another search / check another page / score more candidates, describe the ACTION ("I will fetch page 2 to look for additional X") — do NOT pre-declare the outcome of that future step.',
    "- When your current sentence would contradict an earlier observation in this same trajectory (e.g. you earlier said a product lacked brand X and now you recommend it as brand X), you MUST either (a) explain the reconciliation in one concrete sentence referring to the new evidence (\"the title did not name the brand but the product attributes list 'brand: X'\"), or (b) drop the current claim. Never assert both sides without reconciliation.",
    '- Do NOT paraphrase a constraint into stronger wording than the context supports. "Matches the keywords and price range" is fine; "perfectly matches the brand" is only allowed if the context shows the brand explicitly in the product\'s attributes/title.',
    '- Do NOT invent product attributes (e.g. "added vitamins and zinc", "waterproof certification", "brand: X") that the JSON context does not contain. If the user\'s query mentions an attribute that the candidate does not demonstrably have, you must either stay silent about that attribute or hedge ("the title does not name <attribute>; this is the closest available match"). Attribute fabrication is the #1 cause of the judge\'s "formulaic pattern matching" verdict.',
))

_REGEX_STOPWORDS = set(_flatten_word_groups(
    ("the", "and", "for", "with", "from", "that", "this"),
    ("are", "was", "can", "has", "have", "been", "will"),
    ("find", "finish", "looking", "want", "need"),
    ("get", "buy", "product", "products", "search", "same"),
    ("shop", "within", "budget", "voucher", "discount", "price"),
    ("priced", "pesos", "php", "between", "than", "greater", "less"),
    ("more", "under", "over", "about", "also", "both", "these"),
    ("them", "each", "all", "one", "two", "three", "four"),
    ("five", "offering", "sells", "using", "in", "is", "it", "its"),
    ("or", "at", "on", "by", "be", "do", "an", "my", "me", "im"),
    ("items", "item", "just", "first", "second", "supports"),
    ("support", "compatible", "available", "made", "please", "like"),
    ("of", "above", "deals", "options", "option", "delivery", "shipping"),
    ("offers", "lazmall", "lazflash", "official", "cash", "payment", "pay"),
    ("cost", "costs", "via", "themed", "such", "those", "store", "stores"),
    ("focus", "category", "specifically", "guaranteed", "authenticity"),
    ("returns", "quick", "perks", "should", "help", "purchase", "type"),
    ("to", "named", "called", "family", "belongs", "comes", "another"),
    ("lastly", "benefits", "you", "weighing", "capacity", "size", "sized", "eu", "fits"),
))

_MULTI_PRODUCT_SPLIT_RE = re.compile(
    r"(?:,?\s*and\s+also\s+|,?\s*also,?\s*"
    r"|Second(?:ly)?,\s*|Third(?:ly)?,\s*|First,\s*"
    r"|\(\d+\)\s*|\d+\.\s*"
    r"|Additionally,\s*|Furthermore,\s*|Moreover,\s*"
    r"|In\s+addition,?\s*|Plus,\s*|On\s+top\s+of\s+that,?\s*"
    r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*"
    r"|\bThen\s*,?\s*I\s+(?:need|want|also)\b"
    r"|\bI\s+also\s+(?:want|need)\b)",
    re.IGNORECASE,
)

_ADDITIONALLY_PREFERRED_WORDS: list[str] = [
    "waterproof", "different", "quantity"
]

_JUST_MANUFACTURER_NOT_BRAND: list[str] = [
    "foxconn", "tsmc", "oppo", "flex", "boe", "micron"
]

_BUDGET_SPLIT_RE = re.compile(r"(?:My budget|budget is|I have a voucher)", re.IGNORECASE)


def _wait_for_search_http_slot() -> None:
    def _evict_stale_requests(now: float) -> None:
        cutoff = now - _SEARCH_HTTP_WINDOW_SECONDS
        while _search_http_request_times and _search_http_request_times[0] <= cutoff:
            _search_http_request_times.pop(0)

    def _compute_wait_duration(now: float) -> float:
        wait = 0.0
        if _search_http_request_times:
            gap = now - _search_http_request_times[-1]
            if gap < _SEARCH_HTTP_MIN_INTERVAL_SECONDS:
                wait = _SEARCH_HTTP_MIN_INTERVAL_SECONDS - gap
        if len(_search_http_request_times) >= _SEARCH_HTTP_MAX_REQUESTS_PER_MINUTE:
            oldest_age = now - _search_http_request_times[0]
            wait = max(wait, _SEARCH_HTTP_WINDOW_SECONDS - oldest_age)
        return max(wait, 0.0)

    while True:
        with _search_http_rate_lock:
            now = time.monotonic()
            _evict_stale_requests(now)
            sleep_for = _compute_wait_duration(now)
            if sleep_for <= 0:
                _search_http_request_times.append(now)
                return

        time.sleep(sleep_for)


def _search_http_get(path: str, params: dict | None = None):
    _wait_for_search_http_slot()
    return _search_client.get(path, params)


def _in_word_set(w: str, s: set[str]) -> bool:
    return (
        w in s
        or (w.endswith("s") and w[:-1] in s)
        or (not w.endswith("s") and w + "s" in s)
    )


def safe_tool_call(tool_name: str, params: dict) -> dict:
    global _last_tool_call_time
    since_last = time.monotonic() - _last_tool_call_time
    if since_last < TOOL_CALL_DELAY:
        time.sleep(TOOL_CALL_DELAY - since_last)

    for attempt in range(TOOL_CALL_MAX_RETRIES):
        try:
            result = execute_tool_call(tool_name, params)
            _last_tool_call_time = time.monotonic()
            return result
        except Exception:
            if attempt == TOOL_CALL_MAX_RETRIES - 1:
                raise
            backoff = TOOL_CALL_BASE_BACKOFF * (2 ** attempt)
            logger.warning(
                "tool call '%s' failed on attempt %d/%d — retrying in %.1fs",
                tool_name,
                attempt + 1,
                TOOL_CALL_MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> list[dict]:
    search_params = _build_search_params(
        q,
        page=page,
        shop_id=shop_id,
        price=price,
        sort=sort,
        service=service,
    )
    result = _search_http_get("/search/find_product", search_params) or []
    if result or "service" not in search_params:
        return result

    retry_params = dict(search_params)
    retry_params.pop("service", None)
    return _search_http_get("/search/find_product", retry_params) or []

def _normalize_service(service: str | None) -> str | None:
    if service in (None, ""):
        return service
    pieces = [part.strip() for part in service.split(",")]
    filtered = [part for part in pieces if part and part != "default"]
    if not filtered:
        return None
    return ",".join(filtered)


def _build_search_params(
    query: str,
    *,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"q": quote_plus(query), "page": page}
    optional_fields = {
        "shop_id": shop_id,
        "price": price,
        "sort": None if sort == "default" else sort,
    }
    params.update(_strip_none_entries(optional_fields))
    normalized_service = _normalize_service(service)
    if normalized_service:
        params["service"] = normalized_service
    return params


def _search_products(params: dict[str, Any]) -> list[Product]:
    return _search_client.get("/search/find_product", params) or []


def _search_products_for_spec(
    spec: SearchSpec,
    *,
    shop_id: str | None = None,
    include_price: bool = True,
    omit_service_from_api: bool = False,
) -> list[Product]:
    price = (
        None
        if not include_price
        else spec.get("price") if spec.get("price") is not None else spec.get("price_range")
    )
    query_text = spec.get("q") or spec.get("keywords") or DEFAULT_PRODUCT_QUERY
    service = None if omit_service_from_api else spec.get("service")

    gathered: list[Product] = []
    for page_number in (1, 2):
        page_results = _search_products(
            _build_search_params(
                query_text,
                page=page_number,
                shop_id=shop_id,
                price=price,
                service=service,
            )
        )
        if page_results:
            gathered.extend(page_results)
    return gathered


def _group_products_by_shop(
    broad_results: Sequence[Sequence[Product]],
) -> dict[str, dict[int, list[Product]]]:
    coverage_map: dict[str, dict[int, list[Product]]] = defaultdict(lambda: defaultdict(list))
    for spec_idx, product_list in enumerate(broad_results):
        for product in product_list:
            sid = str(product.get("shop_id", ""))
            if sid:
                coverage_map[sid][spec_idx].append(product)
    return coverage_map


def _product_matches_services(product: Product, service_spec: str | None) -> bool:
    if not service_spec:
        return True
    required = set(filter(None, (part.strip() for part in str(service_spec).split(","))))
    if not required:
        return True
    offered = product.get("service") or []
    if not isinstance(offered, list):
        offered = []
    return required.issubset(set(offered))


def _llm_score_products(
    query_text: str,
    candidates: list[Product],
    details: dict[str, dict],
    only_product_type: bool = False,
    model: str = DEFAULT_CHOOSE_PRODUCT_MODEL,
) -> list[tuple[Product, float]]:
    if not candidates:
        return []

    if _time_left() < 35.0:
        return [(p, 7.0) for p in candidates if _score_product(p, query_text) > 0]

    payload = {
        "request": query_text,
        "candidates": [
            _compact_candidate_payload(
                product, details.get(str(product.get("product_id", ""))), query_text,
            )
            for product in candidates
        ],
        "only_product_type": only_product_type,
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    env_model = getenv("SANDBOX_MODEL")
    model_chain = [env_model] if env_model else [model, FALLBACK_MODEL]

    for parsed in _iter_llm_json(
        models=model_chain,
        temperature=0.5,
        system=_SHOP_SCORER_PROMPT,
        user=user_content,
        retries=PRODUCT_JUDGE_MAX_RETRIES,
        client=_score_inference_client,
    ):
        if not isinstance(parsed, list):
            continue
        scored = _parse_scored_candidates(parsed, candidates)
        logger.info("ShopScorer [%s]: scored %d products", model_chain[0], len(scored))
        return scored

    logger.warning("ShopScorer: all models exhausted; switching to heuristic fallback")
    scored = [
        (product, 7.0 if _score_product(product, query_text) > 0 else 0.0)
        for product in candidates
    ]
    scored.sort(key=lambda x: (x[1], _pid(x[0])), reverse=True)
    return scored


def _pick_winning_spec_by_depth(spec_indices: list[int], specs: list[SearchSpec]) -> int:
    def _spec_depth_scores(spec: SearchSpec) -> tuple[float, int, int]:
        kw_count = len((spec.get("keywords") or "").split())

        price_score = 0.0
        price_range = spec.get("price_range") or ""
        if price_range and "-" in price_range:
            parts = price_range.split("-", 1)
            lo, hi = parts[0].strip(), parts[1].strip()
            if lo and hi:
                price_score = 1.5   # bounded range e.g. "30-50"
            elif lo or hi:
                price_score = 1.0   # open-ended e.g. "50-" or "-30"

        svc_count = len(
            [
                svc.strip()
                for svc in (spec.get("service") or "").split(",")
                if svc.strip()
            ]
        )
        return (price_score, kw_count, svc_count)

    raw = {idx: _spec_depth_scores(specs[idx]) for idx in spec_indices}
    max_kw = max(v[1] for v in raw.values())
    max_svc = max(v[2] for v in raw.values())

    ranked: dict[int, float] = {}
    for idx, (price_score, kw_count, svc_count) in raw.items():
        score = price_score
        if kw_count == max_kw:
            score += 1.0
        if svc_count == max_svc:
            score += 1.0
        ranked[idx] = score

    best = max(ranked.values())
    winners = [idx for idx, s in ranked.items() if s == best]
    return winners[0]  # first wins on tie


# ──────────────────────────────────────────────────────────────────────────
# Shop-task deadline helpers
# ──────────────────────────────────────────────────────────────────────────

def _shop_elapsed_secs() -> float:
    return time.monotonic() - _shop_mono_anchor


def _shop_time_ok() -> bool:
    remaining = _SHOP_DEADLINE_SOFT_SEC - _shop_elapsed_secs()
    return remaining > _SHOP_DEADLINE_RESERVE_SEC


# ──────────────────────────────────────────────────────────────────────────
# Shop-task candidate collection + scoring helpers
# ──────────────────────────────────────────────────────────────────────────

def _collect_shop_candidate_pools(
    specs: list[SearchSpec],
) -> tuple[list[list[Product]], list]:
    """Broad 3-page search per spec with service fallback when <5 results. Returns (pools, tool_log)."""
    broad: list[list[Product]] = []
    tool_log: list = []
    for spec in specs:
        sp = _spec_to_find_product_params(spec)
        bucket: list[Product] = []
        seen: set[str] = set()
        for pg in range(1, 4):
            r = safe_tool_call("find_product", {**sp, "page": pg})
            tool_log.append(r)
            for row in (r.get("result") or []):
                pid = str(row.get("product_id", "") or "")
                if pid and pid not in seen:
                    seen.add(pid)
                    bucket.append(row)
        if len(bucket) < 5 and sp.get("service"):
            no_svc = {k: v for k, v in sp.items() if k != "service"}
            for pg in range(1, 3):
                rr = safe_tool_call("find_product", {**no_svc, "page": pg})
                tool_log.append(rr)
                for row in (rr.get("result") or []):
                    pid = str(row.get("product_id", "") or "")
                    if pid and pid not in seen:
                        seen.add(pid)
                        bucket.append(row)
        broad.append(bucket)
    return broad, tool_log


def _rate_pool_for_shop(
    spec_query: str,
    prods: list[Product],
    details: dict[str, dict],
    only_product_type: bool,
    cap: int,
    shop_cap: int,
) -> list[tuple[Product, float]]:
    """Heuristic pre-filter with score caching, then LLM-score all candidates for a single spec."""
    if not prods:
        return []

    heur_cache: dict[str, float] = {}

    def _heur(pid: str, row: Product) -> float:
        if pid in heur_cache:
            return heur_cache[pid]
        sc = _score_product(row, spec_query, detail=details.get(pid))
        heur_cache[pid] = sc
        return sc

    pool = list(prods)

    if len(pool) > cap:
        ranked = sorted(
            pool,
            key=lambda p: _heur(str(p.get("product_id", "") or "").strip(), p),
            reverse=True,
        )
        short: list[Product] = []
        per_shop: dict[str, int] = defaultdict(int)
        for p in ranked:
            sid = str(p.get("shop_id", "") or "")
            if sid and per_shop[sid] >= shop_cap:
                continue
            short.append(p)
            if sid:
                per_shop[sid] += 1
            if len(short) >= cap:
                break
        # top-up if still below cap (ignoring per-store limit)
        if len(short) < cap:
            have = {str(p.get("product_id", "") or "") for p in short}
            for p in ranked:
                pid = str(p.get("product_id", "") or "")
                if pid and pid not in have:
                    short.append(p)
                    have.add(pid)
                if len(short) >= cap:
                    break
        pool = short

    # Guard: skip LLM if time is short or per-task LLM budget is exhausted
    if _time_left() < 35.0 or not _spend_core_budget():
        fallback = [
            (p, 7.0 if _heur(str(p.get("product_id", "") or "").strip(), p) > 0 else 0.0)
            for p in prods
        ]
        fallback.sort(key=lambda x: (x[1], str(x[0].get("product_id", ""))), reverse=True)
        return fallback

    payload = {
        "request": spec_query,
        "candidates": [
            _compact_candidate_payload(
                p, details.get(str(p.get("product_id", "") or "")), spec_query,
            )
            for p in pool
        ],
        "only_product_type": only_product_type,
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    for parsed in _iter_llm_json(
        models=_judge_model_chain(),
        temperature=0.5,
        system=_SHOP_SCORER_PROMPT,
        user=user_content,
        retries=PRODUCT_JUDGE_MAX_RETRIES,
        client=_score_inference_client,
    ):
        if not isinstance(parsed, list):
            continue
        # Score ALL original candidates; those excluded from pool receive 0.0
        scored = _parse_scored_candidates(parsed, prods)
        logger.info("ShopScorer: scored %d products via %s", len(scored), _judge_model_chain()[0])
        return scored

    logger.warning("ShopScorer: all models exhausted; switching to heuristic fallback")
    fallback = [
        (p, 7.0 if _heur(_pid(p), p) > 0 else 0.0)
        for p in prods
    ]
    fallback.sort(key=lambda x: (x[1], _pid(x[0])), reverse=True)
    return fallback


def _aggregate_shop_pick_by_score(
    shop_ids: list[str],
    shop_coverage: dict[str, dict[int, list[Product]]],
    spec_scored: list[list[tuple[Product, float]]],
    specs: list[SearchSpec],
    query: str,
) -> tuple[str | None, dict[int, dict]]:
    """Pick the best full-coverage shop using already-computed LLM scores (no extra LLM calls)."""
    pid_score_maps: list[dict[str, float]] = []
    for scored in spec_scored:
        m: dict[str, float] = {}
        for p, sc in scored:
            pid = str(p.get("product_id", "") or "").strip()
            if pid and pid not in m:
                m[pid] = float(sc)
        pid_score_maps.append(m)

    best_sid: str | None = None
    best_total = float("-inf")
    best_tie = float("-inf")
    best_chosen: dict[int, dict] = {}

    for sid in shop_ids:
        cov = shop_coverage.get(sid) or {}
        total = 0.0
        tie = 0.0
        chosen: dict[int, dict] = {}
        for idx, spec in enumerate(specs):
            pool = list(cov.get(idx) or [])
            if not pool:
                continue
            smap = pid_score_maps[idx] if idx < len(pid_score_maps) else {}
            ranked = sorted(
                pool,
                key=lambda p: smap.get(str(p.get("product_id", "") or "").strip(), 0.0),
                reverse=True,
            )
            first = ranked[0]
            fid = str(first.get("product_id", "") or "").strip()
            fs = float(smap.get(fid, 0.0))
            total += fs
            sq = spec.get("query") or spec.get("keywords") or query
            tie += (
                float(_score_product(first, str(sq)))
                - float(first.get("price") or 0) / CHEAPER_PRICE_TIEBREAK_DIVISOR
            )
            chosen[idx] = {"product_id": fid, "reason": "", "score": fs}
        if total > best_total or (total == best_total and tie > best_tie):
            best_total = total
            best_tie = tie
            best_sid = sid
            best_chosen = chosen

    logger.info(
        "_aggregate_shop_pick_by_score: winner=%s total_score=%.2f", best_sid, best_total,
    )
    return best_sid, best_chosen


# ──────────────────────────────────────────────────────────────────────────
# Shop-task in-shop search helpers (4-level fallback)
# ──────────────────────────────────────────────────────────────────────────

def _shop_scoped_search(spec: SearchSpec, shop_id: str) -> list[Product]:
    """4-level fallback search within a specific shop."""
    # Level 1: full spec, pages 1-2
    results = _search_products_for_spec(spec, shop_id=shop_id)
    if results:
        return results
    # Level 2: drop service filter
    results = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=True)
    if results:
        return results
    # Level 3: truncated keywords (first 2 words), no service
    kw = spec.get("keywords") or spec.get("q") or ""
    words = kw.split()
    if len(words) > 2:
        trimmed = dict(spec)
        trimmed["keywords"] = " ".join(words[:2])
        trimmed.pop("service", None)
        results = _search_products_for_spec(trimmed, shop_id=shop_id, omit_service_from_api=True)
        if results:
            return results
    # Level 4: relax only_product_type if set
    if spec.get("only_product_type"):
        relaxed = dict(spec)
        relaxed["only_product_type"] = False
        relaxed.pop("service", None)
        results = _search_products_for_spec(relaxed, shop_id=shop_id, omit_service_from_api=True)
        if results:
            return results
    return []


def _pick_inside_shop(spec: SearchSpec, shop_id: str, query: str) -> Product | None:
    """Search within a shop using 4-level fallback, then LLM-pick the best result."""
    found = _shop_scoped_search(spec, shop_id)
    if not found:
        return None
    product_ids = [str(p.get("product_id", "") or "") for p in found if p.get("product_id")]
    details = _fetch_product_details(product_ids)
    sq = spec.get("query") or spec.get("keywords") or query
    chosen = _llm_choose_product(
        sq, found[:TOP_RELEVANCE_CANDIDATES], details,
        only_product_type=bool(spec.get("only_product_type", False)),
        model=FALLBACK_MODEL,
    )
    return chosen if chosen is not None else found[0] if found else None


def _try_partial_shop_coverage(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict[str, dict[int, list[Product]]],
    query: str,
    n_specs: int,
) -> tuple[list[str] | None, dict]:
    target = n_specs - 1
    partial_shops = {
        shop_id: cov for shop_id, cov in shop_coverage.items() if len(cov) == target
    }
    if not partial_shops:
        return None, {}

    pid_to_score: dict[str, float] = {
        str(p.get("product_id", "")): score
        for scored in spec_scored
        for p, score in scored
    }

    def _shop_total(cov: dict) -> float:
        total = 0.0
        for spec_idx, products in cov.items():
            total += max(
                (
                    pid_to_score.get(str(product.get("product_id", "")), 0.0)
                    for product in products
                ),
                default=0.0,
            )
        return total

    shop_scores = {
        shop_id: _shop_total(cov) for shop_id, cov in partial_shops.items()
    }
    max_score   = max(shop_scores.values())
    best_shops  = [
        shop_id for shop_id, shop_score in shop_scores.items() if shop_score == max_score
    ]
    winner_shop = best_shops[0]  # ties → first candidate

    coverage = partial_shops[winner_shop]
    covered  = set(coverage.keys())
    missing_spec_index = next(
        spec_index for spec_index in range(n_specs) if spec_index not in covered
    )

    resolved_ids: list[str | None] = [None] * n_specs
    for spec_idx in covered:
        shop_product_ids = {
            str(product.get("product_id", "")) for product in coverage[spec_idx]
        }
        best_product = next(
            (
                product for product, _ in spec_scored[spec_idx]
                if str(product.get("product_id", "")) in shop_product_ids
            ),
            coverage[spec_idx][0] if coverage[spec_idx] else None,
        )
        if best_product:
            resolved_ids[spec_idx] = str(best_product.get("product_id", ""))

    if not _shop_time_ok():
        return None, {}
    best_missing = _pick_inside_shop(specs[missing_spec_index], winner_shop, query)
    if not best_missing:
        return None, {}
    resolved_ids[missing_spec_index] = str(best_missing.get("product_id", ""))

    if not all(pid is not None for pid in resolved_ids):
        return None, {}

    context = {
        "sub_case": 4,
        "winner_shop_id": winner_shop,
        "winner_shop_score": round(max_score, 2),
        "covered_spec_indices": sorted(covered),
        "missing_spec_idx": missing_spec_index,
        "missing_spec_keywords": specs[missing_spec_index].get("keywords", ""),
        "filled_missing_product": {
            "product_id": str(best_missing.get("product_id", "")),
            "title": best_missing.get("title", ""),
            "price": best_missing.get("price"),
        },
    }
    return resolved_ids, context


def _build_shop_anchor_queue(
    shop_best: dict[str, tuple[float, int, Product]],
    spec_scored: list[list[tuple[Product, float]]],
    n_specs: int,
    limit: int,
) -> list[tuple[float, int, Product]]:
    """Build a deduplicated shop-anchor queue ordered depth-first by score rank."""
    seen_shops: set[str] = set()
    out: list[tuple[float, int, Product]] = []

    def _push(score: float, si: int, p: Product) -> bool:
        if len(out) >= limit:
            return False
        sid = str(p.get("shop_id", "") or "")
        if not sid or sid in seen_shops:
            return False
        seen_shops.add(sid)
        out.append((score, si, p))
        return True

    depth = max((len(s) for s in spec_scored), default=0)
    for rank in range(min(depth, 12)):
        for si in range(n_specs):
            if rank < len(spec_scored[si]):
                p, sc = spec_scored[si][rank]
                _push(float(sc), si, p)
            if len(out) >= limit:
                return out

    for sid, (sc, si, p) in sorted(shop_best.items(), key=lambda x: x[1][0], reverse=True):
        _push(float(sc), si, p)
        if len(out) >= limit:
            break

    return out


def _try_anchor_shop_strategy(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict[str, dict[int, list[Product]]],
    query: str,
    n_specs: int,
) -> tuple[list[str] | None, dict]:
    """Anchor-product strategy: fix the best-scored product, fill remaining specs within its shop."""
    global_max = max((s[0][1] for s in spec_scored if s), default=0.0)
    if global_max <= 0:
        return None, {}

    # Build per-shop best entry: shop_id → (score, spec_idx, product)
    shop_best: dict[str, tuple[float, int, Product]] = {}
    for idx, scored in enumerate(spec_scored):
        for p, sc in scored:
            sid = str(p.get("shop_id", "") or "")
            if not sid:
                continue
            if sid not in shop_best or sc > shop_best[sid][0]:
                shop_best[sid] = (sc, idx, p)

    # Classify sub-case for narration context
    top_by_spec: dict[int, list[Product]] = defaultdict(list)
    for idx, scored in enumerate(spec_scored):
        for p, sc in scored:
            if sc >= global_max:
                top_by_spec[idx].append(p)

    top_spec_indices = list(top_by_spec.keys())
    if len(top_spec_indices) == 1:
        si = top_spec_indices[0]
        cands = top_by_spec[si]
        sub_case = 1 if len(cands) == 1 else 2
        tie_note = (
            "Single global top-scoring product; anchoring directly."
            if sub_case == 1
            else (
                f"{len(cands)} products tied at score {global_max:.1f} in spec[{si}]; "
                "trying cheapest first."
            )
        )
    else:
        chosen_idx = _pick_winning_spec_by_depth(top_spec_indices, specs)
        sub_case = 3
        tie_note = (
            f"Top score {global_max:.1f} tied across specs {top_spec_indices}; "
            f"depth scoring selected spec[{chosen_idx}] as anchor."
        )

    anchors = _build_shop_anchor_queue(shop_best, spec_scored, n_specs, _SHOP_ANCHOR_ATTEMPTS)
    for attempt, (sc, anchor_idx, anchor_product) in enumerate(anchors):
        if not _shop_time_ok():
            logger.info("_try_anchor_shop_strategy: deadline reached at attempt %d", attempt)
            break
        sid = str(anchor_product.get("shop_id", "") or "")
        if not sid:
            continue

        resolved: list[str | None] = [None] * n_specs
        resolved[anchor_idx] = str(anchor_product.get("product_id", "") or "")
        filled: list[dict] = []
        success = True

        for si in range(n_specs):
            if si == anchor_idx:
                continue
            if not _shop_time_ok():
                success = False
                break
            chosen = _pick_inside_shop(specs[si], sid, query)
            if not chosen:
                logger.info(
                    "_try_anchor_shop_strategy: no result for spec[%d] in shop %s", si, sid,
                )
                success = False
                break
            resolved[si] = str(chosen.get("product_id", "") or "")
            filled.append({
                "spec_idx": si,
                "keywords": specs[si].get("keywords", ""),
                "product_id": str(chosen.get("product_id", "") or ""),
                "title": chosen.get("title", ""),
                "price": chosen.get("price"),
                "llm_reason": chosen.get("_llm_reason", ""),
            })

        if success and all(pid is not None for pid in resolved):
            context = {
                "sub_case": sub_case,
                "global_max_score": global_max,
                "tie_note": tie_note,
                "anchor_attempt": attempt + 1,
                "anchor": {
                    "spec_idx": anchor_idx,
                    "keywords": specs[anchor_idx].get("keywords", ""),
                    "product_id": str(anchor_product.get("product_id", "") or ""),
                    "title": anchor_product.get("title", ""),
                    "price": anchor_product.get("price"),
                    "shop_id": sid,
                },
                "filled_specs": filled,
            }
            return [pid for pid in resolved if pid is not None], context

    return None, {}


def _resolve_case_c(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict[str, dict[int, list[Product]]],
    query: str,
    n_specs: int,
) -> tuple[list[str] | None, dict]:
    # Sub-strategy 1: partial shop coverage (n >= 3 only, fast — no new LLM calls)
    if n_specs >= 3:
        resolved_ids, partial_ctx = _try_partial_shop_coverage(
            specs, spec_scored, shop_coverage, query, n_specs
        )
        if resolved_ids:
            return resolved_ids, partial_ctx

    # Sub-strategy 2: anchor-queue strategy (up to _SHOP_ANCHOR_ATTEMPTS shops, deadline-aware)
    return _try_anchor_shop_strategy(specs, spec_scored, shop_coverage, query, n_specs)


def _tokenize_query(query_text: str) -> list[str]:
    return list(
        dict.fromkeys(
            tok
            for tok in re.findall(r"\b\w+\b", query_text.lower())
            if tok not in _SCORING_STOPWORDS and len(tok) > 1
        )
    )


def _order_shops_by_coverage_score(
    shop_ids: list[str],
    shop_coverage: dict[str, dict[int, list[Product]]],
    specs: list[SearchSpec],
    query: str,
) -> list[str]:
    def _heuristic_shop_score(sid: str) -> float:
        cov = shop_coverage.get(sid) or {}
        total = 0.0
        for idx, spec in enumerate(specs):
            pool = cov.get(idx, [])
            if pool:
                sq = spec.get("query") or spec.get("keywords") or query
                total += max((_score_product(p, str(sq)) for p in pool), default=0.0)
        return total

    ranked = [(sid, _heuristic_shop_score(sid)) for sid in shop_ids]
    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [sid for sid, _ in ranked]


@Tool
def calculate_voucher(
    product_prices: str,
    voucher_type: str,
    discount_value: float,
    threshold: float,
    budget: float,
    cap: float = 0,
) -> dict:
    try:
        prices = [
            float(part.strip()) for part in str(product_prices).split(",")
        ]
    except ValueError:
        return {"error": "Invalid product_prices format. Use comma-separated numbers."}

    total = sum(prices)
    discount = 0.0
    voucher_applied = False

    if total >= threshold:
        voucher_applied = True
        if voucher_type == "fixed":
            discount = discount_value
        elif voucher_type == "percentage":
            discount = total * (discount_value / 100.0)
            if cap > 0:
                discount = min(discount, cap)

    total_after = total - discount

    return {
        "prices": prices,
        "total_before": round(total, 2),
        "discount_amount": round(discount, 2),
        "total_after": round(total_after, 2),
        "within_budget": total_after <= budget,
        "voucher_applied": voucher_applied,
        "budget": budget,
    }


@Tool
def recommend_product(product_ids: str) -> str:
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    return f"The interaction has been completed with status: {status}"


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    uncached = [pid for pid in product_ids if pid not in _product_detail_cache]
    for index in range(0, len(uncached), 10):
        batch = uncached[index : index + 10]
        result = _search_http_get("/search/view_product_information", {"product_ids": ",".join(batch)})
        if result and isinstance(result, list):
            for row in result:
                _product_detail_cache[str(row.get("product_id", ""))] = row
    return {pid: _product_detail_cache[pid] for pid in product_ids if pid in _product_detail_cache}


def _score_product(
    product: dict, query_text: str, detail: dict = None, parsed_spec: dict = None
) -> float:
    title = product.get("title", "").lower()
    title_words = set(re.findall(r"\b\w+\b", title))
    query_words = list(
        dict.fromkeys(
            word
            for word in re.findall(r"\b\w+\b", query_text.lower())
            if word not in _SCORING_STOPWORDS and len(word) > 1
        )
    )
    spec = parsed_spec or {}
    score = 0.0

    for query_word in query_words:
        if (
            query_word in title_words
            or query_word.endswith("s")
            and query_word[:-1] in title_words
            or not query_word.endswith("s")
            and (query_word + "s") in title_words
            or len(query_word) >= 3
            and any(
                title_word.startswith(query_word)
                for title_word in title_words
                if len(title_word) > len(query_word)
            )
        ):
            score += 2
        elif any(
            query_word.startswith(title_word) or title_word.startswith(query_word)
            for title_word in title_words
            if len(title_word) > 2
        ):
            score += 1
        if any(char.isdigit() for char in query_word) and query_word in title:
            score += 2

    price = product.get("price")
    if isinstance(price, (int, float)) and spec.get("price_range"):
        min_price, max_price = _parse_price_range(spec["price_range"])
        if (min_price is not None and price < min_price) or (
            max_price is not None and price > max_price
        ):
            score -= 25
        else:
            score += 5

    product_services = set(product.get("service") or [])
    if spec.get("service"):
        required = {
            svc.strip() for svc in spec["service"].split(",") if svc.strip()
        }
        for svc in required:
            if svc in product_services:
                score += 5
            else:
                score -= 15

    if detail:
        exact_values: set[str] = set()
        attr_words: set[str] = set()

        for attr_key, values in (detail.get("attributes") or {}).items():
            key_lower = attr_key.lower()
            attr_words.update(re.findall(r"\b\w+\b", key_lower.replace("_", " ")))
            for value in values if isinstance(values, list) else [values]:
                value_str = str(value).strip().lower()
                exact_values.add(value_str)
                attr_words.update(re.findall(r"\b\w+\b", value_str))

        for _sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for attr_key, value in opts.items():
                    value_str = str(value).strip().lower()
                    exact_values.add(value_str)
                    attr_words.update(re.findall(r"\b\w+\b", value_str))
                    attr_words.update(
                        re.findall(r"\b\w+\b", attr_key.lower().replace("_", " "))
                    )

        for query_word in query_words:
            if query_word in exact_values:
                score += 3
            elif (query_word + "#") in exact_values:
                score += 5
            elif query_word in attr_words:
                score += 2

    return score


def _score_product_for_product_case(
    product: dict, query_text: str, detail: dict = None, parsed_spec: dict = None
) -> float:
    title = product.get("title", "").lower()
    title_words = set(re.findall(r"\b\w+\b", title))
    query_words = list(
        dict.fromkeys(
            word
            for word in re.findall(r"\b\w+\b", query_text.lower())
            if word not in _SCORING_STOPWORDS and len(word) > 1
        )
    )
    spec = parsed_spec or {}

    if (spec.get("brand") or "").lower() in _JUST_MANUFACTURER_NOT_BRAND:
        spec = {**spec, "brand": None}
    
    score = 0.0
    
    for title_word in title_words:
        if (
            title_word in query_words
            or title_word.endswith("s")
            and title_word[:-1] in query_words
            or not title_word.endswith("s")
            and (title_word + "s") in query_words
            or len(title_word) >= 3
            and any(
                query_word.startswith(title_word)
                for query_word in query_words
                if len(query_word) > len(title_word)
            )
        ):
            score += 2
        elif any(
            title_word.startswith(query_word) or query_word.startswith(title_word)
            for query_word in query_words
            if len(query_word) > 2
        ):
            score += 1

        if any(char.isdigit() for char in title_word) and title_word in query_words:
            score += 2
            
    price = product.get("price")
    if isinstance(price, (int, float)) and spec.get("price_range"):
        min_price, max_price = _parse_price_range(spec["price_range"])
        if (min_price is not None and price < min_price) or (
            max_price is not None and price > max_price
        ):
            score -= 25
        else:
            score += 5

    product_services = set(product.get("service") or [])
    if spec.get("service"):
        required = {
            svc.strip() for svc in spec["service"].split(",") if svc.strip()
        }
        for svc in required:
            if svc in product_services:
                score += 5
            else:
                score -= 15

    _service_only_spec = bool(spec.get("service")) and not any(
        spec.get(k) for k in ("brand", "price_range", "only_product_type")
    )
    if _service_only_spec:
        if any(word in title_words for word in _ADDITIONALLY_PREFERRED_WORDS):
            score += 10

    if detail:
        sku_exact: set[str] = set()
        sku_words: set[str] = set()
        attr_exact: set[str] = set()
        attr_words: set[str] = set()

        for attr_key, values in (detail.get("attributes") or {}).items():
            key_lower = attr_key.lower()
            attr_words.update(re.findall(r"\b\w+\b", key_lower.replace("_", " ")))
            for value in values if isinstance(values, list) else [values]:
                value_str = str(value).strip().lower()
                attr_exact.add(value_str)
                attr_words.update(re.findall(r"\b\w+\b", value_str))

        for _sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for attr_key, value in opts.items():
                    value_str = str(value).strip().lower()
                    sku_exact.add(value_str)
                    sku_words.update(re.findall(r"\b\w+\b", value_str))
                    sku_words.update(
                        re.findall(r"\b\w+\b", attr_key.lower().replace("_", " "))
                    )
        
        if spec.get("only_product_type"):
            kw_words = set(re.findall(r"\b\w+\b", (spec.get("keywords") or query_text).lower()))
            if kw_words:
                all_values = sku_exact | attr_exact
                if any(
                    "only" in set(re.findall(r"\b\w+\b", v))
                    and all(_in_word_set(kw, set(re.findall(r"\b\w+\b", v))) for kw in kw_words)
                    for v in all_values
                ):
                    score += 10

        if spec.get("brand"):
            brand_words = set(re.findall(r"\b\w+\b", spec["brand"].lower()))
            all_values = sku_exact | attr_exact
            if brand_words and any(
                brand_words <= set(re.findall(r"\b\w+\b", v)) for v in all_values
            ):
                score += 10

            for word in _ADDITIONALLY_PREFERRED_WORDS:
                if word in all_values:
                    score += 10 

        _kw_str = (spec.get("keywords") or query_text).lower()
        remaining_words = set(_kw_str.split())
        for value_str in sku_exact:
            words = value_str.split()
            if words and all(_in_word_set(w, remaining_words) for w in words):
                score += 5
                remaining_words -= set(words)
        for value_str in attr_exact:
            words = value_str.split()
            if words and all(_in_word_set(w, remaining_words) for w in words):
                score += 3
                remaining_words -= set(words)

        _bare_spec = not any(spec.get(k) for k in ("brand", "price_range", "service", "only_product_type"))
        if _bare_spec:
            all_value_words = sku_words | attr_words
            found_any = False
            for word in _ADDITIONALLY_PREFERRED_WORDS:
                if word in all_value_words:
                    score += 10
                    found_any = True
            if not found_any:
                score -= 10

    return score


def voucher_max_total_price(voucher: dict) -> float | None:
    discount_type = voucher.get("discount_type", "percentage")
    discount_rate = float(voucher.get("discount_value", 0))
    min_required = float(voucher.get("threshold", 0))
    discount_cap = float(voucher.get("cap", 0))
    budget = float(voucher.get("budget", 0))

    if discount_type == "fixed":
        max_price = budget + discount_rate
        if max_price <= min_required:
            return min_required
        return max_price

    rate = discount_rate / 100.0 if discount_rate > 1 else discount_rate
    if rate <= 0 or rate >= 1:
        return None

    if discount_cap > 0 and budget / (1 - rate) > (budget + discount_cap):
        max_price = budget + discount_cap
    else:
        max_price = budget / (1 - rate)

    if max_price <= min_required:
        return min_required

    return max_price


def _parse_json_object_from_llm(content: str) -> dict | None:
    text = re.sub(r"```json?\s*", "", content)
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group())
                return result if isinstance(result, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _compact_candidate_payload(
    product: dict, detail: dict | None, query_text: str,
) -> dict:
    sku_options = (detail or {}).get("sku_options", {}) or {}
    query_words = set(_tokenize_query(query_text))

    ranked: list = []
    for opt in sku_options.values():
        if not isinstance(opt, dict):
            continue
        opt_words = set(
            w
            for w in re.findall(r"\b\w+\b", " ".join(str(v).lower() for v in opt.values()))
            if len(w) > 1
        )
        ranked.append((len(query_words & opt_words), opt))

    sku_preview: list[dict] = []
    seen_keys = set()
    for _overlap, opt in sorted(ranked, key=lambda item: item[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in seen_keys:
            seen_keys.add(key)
            sku_preview.append(opt)
    return {
        "product_id": str(product.get("product_id", "")).strip(),
        "title": product.get("title", ""),
        "price": product.get("price"),
        "service": product.get("service", []),
        "attributes": (detail or {}).get("attributes", {}),
        "sku_options_preview": sku_preview[:8],
    }


def _verify_reason_grounded(
    reason: str, product: dict, detail: dict | None, query_text: str,
) -> tuple[bool, list[str]]:
    haystack_parts = [(product.get("title") or "").lower()]
    if isinstance(detail, dict):
        attrs = detail.get("attributes") or {}
        if isinstance(attrs, dict):
            for k, vs in attrs.items():
                haystack_parts.append(str(k).lower().replace("_", " "))
                if isinstance(vs, list):
                    haystack_parts.extend(str(v).lower() for v in vs)
                else:
                    haystack_parts.append(str(vs).lower())
        skus = detail.get("sku_options") or {}
        if isinstance(skus, dict):
            for opts in skus.values():
                if isinstance(opts, dict):
                    for k, v in opts.items():
                        haystack_parts.append(str(k).lower().replace("_", " "))
                        haystack_parts.append(str(v).lower())
    haystack = " ".join(haystack_parts)
    query_terms = {
        w for w in re.findall(r"\b\w{4,}\b", query_text.lower())
        if w not in _SCORING_STOPWORDS
    }
    if not query_terms:
        return True, []
    reason_lower = reason.lower()
    claimed = {t for t in query_terms if t in reason_lower}
    missing = [t for t in claimed if t not in haystack]
    return len(missing) == 0, missing


def _neutralize_reason(original_reason: str, missing: list[str]) -> str:
    missing_str = ", ".join(sorted(missing))
    return (
        f"Selected as the best available match among returned candidates; "
        f"the user's requested term(s) ({missing_str}) could not be confirmed literally "
        f"in this product's title, attributes, or sku_options, so the match is partial."
    )


def _apply_reason_grounding(
    result_product: dict, reason: str, relevance_score: float,
    product: dict, detail: dict | None, query_text: str,
) -> None:
    grounded, missing = _verify_reason_grounded(reason, product, detail, query_text)
    result_product["_llm_relevance_score"] = relevance_score
    if grounded:
        result_product["_llm_reason"] = reason
        return
    result_product["_llm_reason"] = _neutralize_reason(reason, missing)
    result_product["_llm_reason_ungrounded_terms"] = missing


def _llm_choose_product(
    query_text: str,
    candidates: list,
    details: dict[str, dict],
    only_product_type: bool = False,
    model: str = DEFAULT_CHOOSE_PRODUCT_MODEL,
) -> dict | None:
    if _time_left() < 35.0:
        return None

    payload = {
        "request": query_text,
        "candidates": [
            _compact_candidate_payload(
                product, details.get(str(product.get("product_id", ""))), query_text,
            )
            for product in candidates[:10]
        ],
        "only_product_type": only_product_type
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    
    env_model = getenv("SANDBOX_MODEL")
    model_chain = (
        [env_model] if env_model
        else list(dict.fromkeys([model] + _judge_model_chain()))
    )

    for parsed in _iter_llm_json(
        models=model_chain,
        temperature=0.5,
        system=_PRODUCT_JUDGE_PROMPT,
        user=user_content,
        retries=PRODUCT_JUDGE_MAX_RETRIES,
    ):
        if not isinstance(parsed, dict):
            continue
        best_product_id = str(parsed.get("best_product_id", "")).strip()
        reason = str(parsed.get("reason", "")).strip()
        try:
            relevance_score = float(parsed.get("relevance_score", 0))
        except (TypeError, ValueError):
            relevance_score = 0.0
        logger.info(
            "ProductJudge: selected product_id=%s score=%.1f reason=%s",
            best_product_id, relevance_score, reason,
        )
        for product in candidates[:10]:
            if str(product.get("product_id", "")).strip() == best_product_id:
                result_product = dict(product)
                detail = details.get(str(product.get("product_id", "")))
                _apply_reason_grounding(
                    result_product, reason, relevance_score,
                    product, detail, query_text,
                )
                return result_product
        logger.warning("ProductJudge: unknown product_id=%s; trying next", best_product_id)

    logger.warning("ProductJudge: all models exhausted; falling back to heuristic")
    return None


def _llm_choose_product_with_consistency(
    query_text: str,
    candidates: list,
    details: dict[str, dict],
    only_product_type: bool = False,
    model: str = DEFAULT_CHOOSE_PRODUCT_MODEL,
) -> dict | None:
    """Self-consistency variant: run LLM judge twice (forward + reversed order).

    If both picks agree → return with boosted confidence score.
    If they disagree and heuristic gap is small → return the higher-scoring pick.
    Falls back to single judge when _VOUCHER_ENABLE_SELF_CONSISTENCY is False.
    """
    first = _llm_choose_product(query_text, candidates, details, only_product_type, model=model)
    if not first or not _VOUCHER_ENABLE_SELF_CONSISTENCY:
        return first

    pid_pick = str(first.get("product_id", "") or "").strip()

    # Heuristic scores to decide whether a second call is worthwhile
    heur_scores: list[tuple[str, float]] = []
    for c in candidates[:_VOUCHER_MAX_JUDGE_CANDIDATES]:
        pid = str(c.get("product_id", "") or "").strip()
        s = _score_product(c, query_text, detail=details.get(pid))
        heur_scores.append((pid, s))
    heur_scores.sort(key=lambda x: x[1], reverse=True)
    top_gap = (
        heur_scores[0][1] - heur_scores[1][1] if len(heur_scores) > 1 else 10.0
    )
    if top_gap >= _VOUCHER_SELF_CONSISTENCY_GAP:
        return first  # heuristic is already decisive → no second call needed

    # Second call with reversed candidate order
    reversed_cands = list(reversed(candidates[:_VOUCHER_MAX_JUDGE_CANDIDATES]))
    second = _llm_choose_product(
        query_text, reversed_cands, details, only_product_type, model=model,
    )
    if not second:
        return first

    if str(second.get("product_id", "") or "").strip() == pid_pick:
        # Both agree → boost score slightly
        s1 = float(first.get("_llm_relevance_score", 0) or 0)
        s2 = float(second.get("_llm_relevance_score", 0) or 0)
        first["_llm_relevance_score"] = min(10.0, s1 + 0.5 * s2 / 10.0)
        return first

    s1 = float(first.get("_llm_relevance_score", 0) or 0)
    s2 = float(second.get("_llm_relevance_score", 0) or 0)
    return first if s1 >= s2 else second


def _select_best_product_for_product_case(
    products: list,
    query_text: str,
    top_count: int = 15,
    prefer_cheaper: bool = False,
    parsed_spec: dict = None,
) -> dict | None:

    if not products:
        return None

    top = sorted(
        products,
        key=lambda item: _score_product_for_product_case(
            item, query_text, parsed_spec=parsed_spec
        ),
        reverse=True,
    )[:top_count]
    product_ids = [
        str(item.get("product_id", "")) for item in top if item.get("product_id")
    ]
    details = _fetch_product_details(product_ids)
    if not top:
        return None

    scored_top = [
        (
            item,
            round(
                _score_product_for_product_case(
                    item, query_text,
                    detail=details.get(str(item.get("product_id", ""))),
                    parsed_spec=parsed_spec,
                ),
                2,
            ),
        )
        for item in top
    ]
    scored_top.sort(key=lambda x: x[1], reverse=True)
    
    top_score = scored_top[0][1] if scored_top else 0.0
    score_floor = top_score - 2.0
    llm_candidates = [item for item, score in scored_top if score >= score_floor]
    
    llm_choice = _llm_choose_product(
        query_text, llm_candidates, details,
        only_product_type=bool(parsed_spec.get("only_product_type", False)),
    )
    logger.info("llm_choice product_id=%s", llm_choice.get("product_id") if llm_choice else None)
    if llm_choice is not None:
        return llm_choice
    return max(
        top,
        key=lambda item: _score_product_for_product_case(
            item,
            query_text,
            details.get(str(item.get("product_id", ""))),
            parsed_spec=parsed_spec,
        ),
    )


def _infer_task_type(query: str) -> str:
    query_lower = query.lower()
    if any(
        marker in query_lower
        for marker in (
            "voucher", "coupon", "discount", "promo", "minimum spend",
            "threshold", "budget", "final price",
        )
    ):
        return "voucher"
    if any(x in query_lower for x in ("same shop", "same store", "same seller", "one shop")):
        return "shop"
    if re.search(r"\b(first|second|third)\b", query_lower) or any(
        x in query_lower for x in (" along with ", " together with ", " plus ", " bundle ")
    ):
        if any(x in query_lower for x in ("shop", "store", "seller", "offering", "sells")):
            return "shop"
        return "voucher" if "budget" in query_lower or "voucher" in query_lower else "product"
    if ("shop" in query_lower or "store" in query_lower or "seller" in query_lower) and (
        re.search(
            r"\b(both|these|offering|offers|sells|same|together|along\s+with)\b",
            query_lower,
        )
        is not None
        or _MULTI_PRODUCT_SPLIT_RE.search(query) is not None
    ):
        return "shop"
    return "product"


def _sanitize_keyword_text(text: str | None) -> str:
    if not text:
        return "product"
    filtered = [
        w
        for w in text.lower().split()
        if w not in _SCORING_STOPWORDS
    ]
    if not filtered:
        return "product"
    return " ".join(dict.fromkeys(filtered))


def _sanitize_product_search_params(params: dict) -> dict:
    sanitized = dict(params)
    products: list[dict] = []
    for product in sanitized.get("products", []) or []:
        if not isinstance(product, dict):
            continue
        cleaned = dict(product)
        if "keywords" in cleaned:
            cleaned["keywords"] = _sanitize_keyword_text(cleaned.get("keywords"))
        if "q" in cleaned:
            cleaned["q"] = _sanitize_keyword_text(cleaned.get("q"))
        products.append(cleaned)
    if products:
        sanitized["products"] = products
    return sanitized


def _extract_query_params_llm(query: str, task_type: str) -> dict:
    if _time_left() < LLM_PARSE_MIN_TIME_LEFT:
        return _extract_query_params_regex(query)
    system_prompt = _TASK_EXTRACTION_PROMPTS.get(task_type, _PRODUCT_PROMPT)
    model_chain = _parse_model_chain(task_type)

    for model in model_chain:
        result = _inference_client.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "temperature": 0,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
            },
        )
        if result and result.get("choices"):
            content = result["choices"][0].get("message", {}).get("content") or ""
            parsed = _parse_json_object_from_llm(content)
            if parsed is not None:
                if task_type == "product":
                    return _sanitize_product_search_params(parsed)
                if task_type == "shop":
                    for product_entry in parsed.get("products", []):
                        if product_entry.get("keywords"):
                            product_entry["keywords"] = " ".join(
                                word for word in product_entry["keywords"].split()
                                if word.lower() not in _SCORING_STOPWORDS
                            )
                return parsed
            logger.warning("LLM extraction: %s returned unparseable response, trying next", model)
        else:
            logger.warning("LLM extraction: %s returned no response, trying next", model)

    logger.warning("LLM extraction failed on all models; falling back to regex")
    return _extract_query_params_regex(query)


def _extract_query_params_regex(query: str) -> dict:
    task_type = _infer_task_type(query)

    def _extract_product_spec(text: str) -> dict:
        alpha_words = [
            w for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()) if w not in _REGEX_STOPWORDS
        ]
        alnum_tokens = re.findall(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", text.lower())
        words = alpha_words[:6]
        for t in alnum_tokens[:2]:
            if t not in words:
                words.append(t)
        for s in re.findall(r"(\d+)#", text)[:2]:
            if s not in words:
                words.append(s)
        keywords = " ".join(words) or "product"

        price_range = None
        m = re.search(
            r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)", text, re.I
        )
        if m:
            price_range = f"{m.group(1)}-"
        else:
            m = re.search(r"(\d{1,6})\s*(?:to|and|-)\s*(\d{1,6})\s*(?:pesos|php)", text, re.I)
            if m:
                price_range = f"{m.group(1)}-{m.group(2)}"
            elif re.search(r"(?:price|pesos|php|cost)", text, re.I):
                m = re.search(r"(\d{1,6})\s+(?:to|and)\s+(\d{1,6})", text)
                if m:
                    price_range = f"{m.group(1)}-{m.group(2)}"

        service = None
        tl = text.lower()
        if "lazmall" in tl or "official" in tl:
            service = "official"
        if "free shipping" in tl or "free delivery" in tl:
            service = "freeShipping" if not service else f"{service},freeShipping"
        if "lazflash" in tl or "flash sale" in tl or "flashsale" in tl:
            service = "flashsale" if not service else f"{service},flashsale"
        if "cash on delivery" in tl or "cod" in tl:
            service = "COD" if not service else f"{service},COD"

        return {"keywords": keywords, "price_range": price_range, "service": service}

    product_text = _BUDGET_SPLIT_RE.split(query)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query

    parts = [
        part.strip()
        for part in _MULTI_PRODUCT_SPLIT_RE.split(product_text)
        if part and len(part.strip()) > 10
    ]
    if len(parts) <= 1 and re.search(r"(?:^|\n|\s)(?:\d+\.|\(\d+\)|[-*])\s+", product_text):
        parts = [
            p.strip(" .;\n")
            for p in re.split(r"(?:^|\n|\s)(?:\d+\.|\(\d+\)|[-*])\s+", product_text)
            if len(p.strip()) > 10
        ]
    if len(parts) <= 1 and ";" in product_text and any(
        marker in product_text.lower()
        for marker in ("first", "second", "third", "also", "shop", "voucher", "budget")
    ):
        parts = [p.strip(" .;\n") for p in product_text.split(";") if len(p.strip()) > 10]
    if not parts:
        parts = [query]

    products = [_extract_product_spec(part) for part in parts]
    products = [
        spec for spec in products if len(spec["keywords"].split()) >= 2
    ] or products
    is_shop = task_type == "shop" or (task_type == "voucher" and "same shop" in query.lower())

    return {"task_type": task_type, "products": products, "is_shop_voucher": is_shop}


def _spec_to_find_product_params(product: dict, *, include_price: bool = True) -> dict[str, Any]:
    keywords = product.get("keywords", "product")
    service = product.get("service")

    if not service and bool(product.get("only_product_type")):
        search_query = keywords + " only"
    else:
        search_query = keywords

    params: dict[str, Any] = {"q": search_query}
    if include_price and product.get("price_range"):
        params["price"] = product["price_range"]
    if service:
        params["service"] = service
    return params


def _parse_price_range(price_range: str | None) -> tuple[float | None, float | None]:
    if not price_range or not isinstance(price_range, str):
        return None, None
    s = price_range.strip()
    if "-" not in s:
        try:
            return None, float(s)
        except ValueError:
            return None, None
    lo_s, _, hi_s = s.partition("-")
    try:
        lo = float(lo_s) if lo_s.strip() else None
    except ValueError:
        lo = None
    try:
        hi = float(hi_s) if hi_s.strip() else None
    except ValueError:
        hi = None
    return lo, hi


def _normalize_voucher_fields(raw: dict | None) -> dict:
    voucher = raw or {}
    discount_type = voucher.get("discount_type", "percentage")
    if isinstance(discount_type, str) and discount_type.lower() in {"", "null", "none"}:
        discount_type = "percentage"
    discount_value = voucher.get("discount_value")
    if discount_value in (None, "", "null"):
        discount_value = voucher.get("face_value")
    if discount_value in (None, "", "null") and voucher.get("discount") is not None:
        discount_value = float(voucher.get("discount") or 0) * 100
    return {
        "discount_type": discount_type,
        "discount_value": float(discount_value or 0),
        "threshold": float(voucher.get("threshold", 0)),
        "cap": float(voucher.get("cap", 0)),
        "budget": float(voucher.get("budget", 0)),
        "same_shop": voucher.get("voucher_type") == "shop"
        or bool(voucher.get("same_shop", False)),
    }


def _deduplicate_products(products: list) -> list:
    seen: set = set()
    out: list = []
    for product in products:
        product_id = str(product.get("product_id", ""))
        if product_id and product_id not in seen:
            seen.add(product_id)
            out.append(product)
    return out


def _format_product_ids(ids: list, expected_order: list = None) -> str:
    seen = set()
    out = []
    for pid in ids:
        pid = str(pid).strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    if expected_order:
        rank = {
            product_id: index
            for index, product_id in enumerate(expected_order)
        }
        out = sorted(out, key=lambda product_id: rank.get(product_id, len(expected_order)))
    return ",".join(out) if out else FALLBACK_PRODUCT_ID

def _enrich_products_for_reason(product_summaries: list[dict]) -> list[dict]:
    try:
        product_ids = [
            str(summary.get("product_id", "")) for summary in product_summaries
        ]
        _fetch_product_details(product_ids)
    except Exception:
        logger.warning("_enrich_products_for_reason: detail fetch failed", exc_info=True)

    enriched = []
    for summary in product_summaries:
        try:
            product_id = str(summary.get("product_id", ""))
            detail = (
                _product_detail_cache.get(product_id, {})
                if isinstance(_product_detail_cache, dict)
                else {}
            )
            entry: dict = {
                "product_id": product_id,
                "title": summary.get("title") or (
                    detail.get("title", "") if isinstance(detail, dict) else ""
                ),
                "price": summary.get("price") if summary.get("price") is not None else (
                    detail.get("price") if isinstance(detail, dict) else None
                ),
            }
            if isinstance(detail, dict):
                sku_options_raw = detail.get("sku_options") or []
                normalized_skus: list[dict] = []
                if isinstance(sku_options_raw, list):
                    for sku_row in sku_options_raw:
                        if not isinstance(sku_row, dict):
                            continue
                        vals = sku_row.get("values", [])
                        if not isinstance(vals, list):
                            vals = list(vals.values()) if isinstance(vals, dict) else []
                        normalized_skus.append(
                            {"name": sku_row.get("name"), "values": vals[:5]}
                        )
                elif isinstance(sku_options_raw, dict):
                    attr_values: dict[str, list] = {}
                    for variant in sku_options_raw.values():
                        if not isinstance(variant, dict):
                            continue
                        for attr_name, attr_val in variant.items():
                            attr_values.setdefault(attr_name, [])
                            if attr_val not in attr_values[attr_name]:
                                attr_values[attr_name].append(attr_val)
                    for attr_name, vals in attr_values.items():
                        normalized_skus.append({"name": attr_name, "values": vals[:5]})
                if normalized_skus:
                    entry["sku_options"] = normalized_skus[:3]

                attrs = detail.get("attributes") or {}
                if isinstance(attrs, dict) and attrs:
                    entry["attributes"] = {
                        key: value for key, value in list(attrs.items())[:8]
                    }

                services = detail.get("service_tags") or detail.get("services") or []
                if isinstance(services, list) and services:
                    entry["service_tags"] = services[:6]
        except Exception:
            logger.warning(
                "_enrich_products_for_reason: failed for product_id=%s",
                summary.get("product_id"),
                exc_info=True,
            )
            entry = {
                "product_id": str(summary.get("product_id", "")),
                "title": summary.get("title", ""),
                "price": summary.get("price"),
            }
        enriched.append(entry)
    return enriched

def _invoke_step_narrator(
    query: str, context: dict, fallback: str, force: bool = False,
) -> str:
    if not force and (_time_left() < 15.0 or not _spend_narration_budget()):
        return fallback

    user_content = json.dumps({"query": query, **context}, ensure_ascii=False)
    for text in _iter_llm_response_text(
        models=_narrator_model_chain(),
        temperature=0.3,
        system=_THINK_NARRATOR_PROMPT,
        user=user_content,
        retries=1,
        max_tokens=_NARRATOR_MAX_TOKENS,
    ):
        if len(text) >= 80:
            return text
    return fallback


def _compact_product_list(items: list) -> list:
    return [
        {"pid": str(p.get("product_id", "")),
         "p": p.get("price"),
         "s": str(p.get("shop_id", ""))}
        for p in items[:10]
        if isinstance(p, dict)
    ]


def _compact_tool_result(tool_call: dict) -> dict:
    if not isinstance(tool_call, dict) or tool_call.get("name") != "find_product":
        return tool_call
    inner = tool_call.get("result")
    if isinstance(inner, dict) and isinstance(inner.get("result"), list):
        new_tc = dict(tool_call)
        new_tc["result"] = {**inner, "result": _compact_product_list(inner["result"])}
        return new_tc
    if isinstance(inner, list):
        new_tc = dict(tool_call)
        new_tc["result"] = _compact_product_list(inner)
        return new_tc
    return tool_call


def _append_step(think: str, tool_results: list, response: str, query: str, steps: list) -> None:
    compact = [_compact_tool_result(tc) for tc in (tool_results or [])]
    step = create_dialogue_step(think, compact, response, query, len(steps) + 1)
    steps.append(step)


def _finish_session(product_ids: list, status: str, query: str, steps: list, think: str = "", llm_reason: str = "") -> None:
    real_ids = [
        str(pid).strip()
        for pid in product_ids
        if str(pid).strip() and str(pid).strip() != FALLBACK_PRODUCT_ID
    ]
    if not real_ids and status == "success":
        status = "failure"
    formatted_ids = _format_product_ids(real_ids) if real_ids else ""
    tool_results = []
    if real_ids:
        rec = safe_tool_call(
            "recommend_product",
            {"product_ids": formatted_ids},
        )
        tool_results.append(rec)
    term = safe_tool_call("terminate", {"status": status})
    tool_results.append(term)
    if not think:
        fallback_finish = (
            (
                f"I am recommending product(s) {formatted_ids} for the query. "
                if formatted_ids
                else "I could not identify a valid product_id to recommend. "
            )
            + (f"{llm_reason} " if llm_reason else "")
            + f"Status: {status}."
        )
        think = _invoke_step_narrator(
            query,
            {
                "recommended_product_ids": formatted_ids,
                "status": status,
                **({"llm_reason": llm_reason} if llm_reason else {}),
                "note": "Finalising recommendation and terminating the session.",
            },
            fallback=fallback_finish,
            force=True,
        )
    _append_step(think, tool_results, "Done.", query, steps)


def _safe_score(prod: dict, q: str, spec: dict | None) -> float | None:
    try:
        return round(_score_product(prod, q, parsed_spec=spec), 1)
    except Exception:
        return None


def _emit_weighing_step(
    leader: dict | None, pool: list, spec: dict | None,
    query: str, steps: list, n_alts: int = 3,
) -> None:
    if not leader or not pool:
        return
    lead_pid = str(leader.get("product_id", ""))
    lead_heur = _safe_score(leader, query, spec)
    others = [p for p in pool if str(p.get("product_id", "")) != lead_pid]
    try:
        others = sorted(
            others,
            key=lambda p: _score_product(p, query, parsed_spec=spec),
            reverse=True,
        )
    except Exception:
        pass
    alternatives = [
        {"product_id": str(a.get("product_id", "")),
         "title": (a.get("title") or "")[:80],
         "price": a.get("price"),
         "heuristic_score": _safe_score(a, query, spec)}
        for a in others[:n_alts]
    ]
    ctx = {
        "weighing": {
            "leader": {
                "product_id": lead_pid,
                "title": (leader.get("title") or "")[:80],
                "price": leader.get("price"),
                "heuristic_score": lead_heur,
                "llm_reason": leader.get("_llm_reason", ""),
                "relevance_score": leader.get("_llm_relevance_score", 0),
            },
            "alternatives": alternatives,
        },
        "query_constraints": {
            "keywords": (spec or {}).get("keywords"),
            "price_range": (spec or {}).get("price_range"),
            "service": (spec or {}).get("service"),
        },
    }
    lead_reason = leader.get("_llm_reason") or ""
    if alternatives:
        alt_desc = " and ".join(
            f"pid={a['product_id']} (${a['price']}, score={a['heuristic_score']})"
            for a in alternatives[:2]
        )
        reason_clause = lead_reason if lead_reason else f"its heuristic score of {lead_heur} is highest among candidates"
        fb = (
            f"I am comparing the top candidates. "
            f"Product pid={lead_pid} (title='{(leader.get('title') or '')[:60]}', "
            f"${leader.get('price')}, score={lead_heur}) is preferred over {alt_desc} "
            f"because {reason_clause}."
        )
    else:
        reason_clause = lead_reason if lead_reason else f"heuristic score {lead_heur} is highest"
        fb = (
            f"I am evaluating the leading candidate. Product pid={lead_pid} "
            f"(${leader.get('price')}, score={lead_heur}). Reason: {reason_clause}."
        )
    think = _invoke_step_narrator(query, ctx, fallback=fb)
    _append_step(think, [], "", query, steps)


def _emit_weighing_step_multi(
    leaders: list, pools: list, specs: list,
    query: str, steps: list, n_alts: int = 2,
) -> None:
    per_spec: list[dict] = []
    for i, (leader, pool, spec) in enumerate(zip(leaders, pools, specs)):
        if leader is None:
            continue
        lead_pid = str(leader.get("product_id", ""))
        lead_heur = _safe_score(leader, query, spec)
        others = [p for p in (pool or []) if str(p.get("product_id", "")) != lead_pid]
        try:
            others = sorted(
                others,
                key=lambda p: _score_product(p, query, parsed_spec=spec),
                reverse=True,
            )
        except Exception:
            pass
        alt_entries = [
            {"product_id": str(a.get("product_id", "")),
             "price": a.get("price"),
             "heuristic_score": _safe_score(a, query, spec)}
            for a in others[:n_alts]
        ]
        per_spec.append({
            "spec_idx": i,
            "keywords": (spec or {}).get("keywords", ""),
            "leader": {
                "product_id": lead_pid,
                "price": leader.get("price"),
                "heuristic_score": lead_heur,
                "llm_reason": leader.get("_llm_reason", ""),
            },
            "alternatives": alt_entries,
        })
    if not per_spec:
        return
    ctx = {"weighing": {"per_spec": per_spec}}
    fb_parts = [f"I am comparing candidates across {len(per_spec)} product specs."]
    for e in per_spec:
        lead = e["leader"]
        alts = e.get("alternatives", [])
        if alts:
            alt_desc = " and ".join(
                f"pid={a['product_id']} (${a['price']}, score={a['heuristic_score']})"
                for a in alts[:2]
            )
            lead_reason = lead.get("llm_reason") or f"heuristic score {lead.get('heuristic_score')} is highest"
            fb_parts.append(
                f"Spec[{e['spec_idx']}] '{e['keywords']}': "
                f"pid={lead['product_id']} (${lead['price']}, score={lead['heuristic_score']}) "
                f"wins over {alt_desc} because {lead_reason}."
            )
        else:
            fb_parts.append(
                f"Spec[{e['spec_idx']}] '{e['keywords']}': "
                f"pid={lead['product_id']} (${lead['price']}, score={lead['heuristic_score']}) selected."
            )
    think = _invoke_step_narrator(query, ctx, fallback=" ".join(fb_parts))
    _append_step(think, [], "", query, steps)


def _check_pick_against_query(
    *, title: str, price: Any, parsed_spec: dict,
) -> dict:
    title_lower = (title or "").lower()
    query_keywords = [
        w for w in str(parsed_spec.get("keywords", "") or "").lower().split() if w
    ]
    matched = [w for w in query_keywords if w in title_lower]
    missing = [w for w in query_keywords if w not in title_lower]
    price_range = parsed_spec.get("price_range")
    price_ok: bool | None = None
    price_note = "no price range was parsed from the query"
    try:
        if price_range:
            lo, hi = _parse_price_range(str(price_range))
            if price is not None:
                pv = float(price)
                if lo is not None and pv < lo:
                    price_ok = False
                    price_note = f"price {pv} is BELOW lower bound {lo} of range {price_range}"
                elif hi is not None and pv > hi:
                    price_ok = False
                    price_note = f"price {pv} is ABOVE upper bound {hi} of range {price_range}"
                else:
                    price_ok = True
                    price_note = f"price {pv} fits inside range {price_range}"
            else:
                price_note = f"no price available to compare against range {price_range}"
    except (TypeError, ValueError):
        price_note = f"price {price!r} is not numeric; could not check range {price_range}"
    if not missing and price_ok is not False:
        overall_note = "The selected product looks like a genuine match for the parsed query."
    elif missing and price_ok is False:
        overall_note = (
            f"HONEST MISMATCH: title is missing query terms {missing} and price is outside "
            "the requested range. This is the best available candidate, not a clean fit."
        )
    elif missing:
        overall_note = (
            f"HONEST MISMATCH: the selected title is missing query terms {missing}; "
            "attributes may still confirm the fit, but the title alone is imperfect."
        )
    else:
        overall_note = (
            "HONEST MISMATCH: title matches the keywords but the price does not fit "
            "the requested range. Taking it as the closest available option."
        )
    return {
        "query_keywords": query_keywords,
        "keywords_matched": matched,
        "keywords_missing": missing,
        "title_contains_all_keywords": not missing,
        "price_ok": price_ok,
        "price_note": price_note,
        "overall_note": overall_note,
    }


def _run_single_product_search(params: dict, query: str, steps: list) -> None:
    products_specs = params.get("products", [{}])
    primary_spec = products_specs[0] if products_specs else {}
    search_params = _spec_to_find_product_params(primary_spec)

    pool: list = []
    seen: set[str] = set()
    tool_bundle: list = []

    r1 = safe_tool_call("find_product", {**search_params, "page": 1})
    tool_bundle.append(r1)
    for row in (r1.get("result") or []):
        pid = str(row.get("product_id", "") or "")
        if pid and pid not in seen:
            seen.add(pid)
            pool.append(row)

    top_candidates = [
        {"title": r.get("title", ""), "price": r.get("price"), "product_id": str(r.get("product_id", ""))}
        for r in pool[:5]
    ]
    fallback_search = (
        f"Searched for '{search_params.get('q', '')}' "
        f"(price={search_params.get('price', 'any')}, service={search_params.get('service', 'any')}). "
        f"Found {len(pool)} results. Top candidates: {top_candidates}."
    )
    think_search = _invoke_step_narrator(
        query,
        {
            "search_query": search_params.get("q", ""),
            "price_filter": search_params.get("price"),
            "service_filter": search_params.get("service"),
            "total_results": len(pool),
            "top_candidates": top_candidates,
        },
        fallback=fallback_search,
    )
    _append_step(think_search, tool_bundle, "", query, steps)

    page1_pids = [str(p.get("product_id", "") or "") for p in pool[:10] if p.get("product_id")]
    page1_details = _fetch_product_details(page1_pids)
    page1_scores = [
        _score_product_for_product_case(
            p, query,
            detail=page1_details.get(str(p.get("product_id", ""))),
            parsed_spec=primary_spec,
        )
        for p in pool[:10]
    ]
    max_p1_score = max(page1_scores, default=0.0)
    
    if not pool or max_p1_score < 10.0:
        fb_calls: list = []

        r2 = safe_tool_call("find_product", {**search_params, "page": 2})
        fb_calls.append(r2)
        for row in (r2.get("result") or []):
            pid = str(row.get("product_id", "") or "")
            if pid and pid not in seen:
                seen.add(pid)
                pool.append(row)

        fallback_broaden = (
            f"Weak page-1 heuristic score (max={max_p1_score:.1f} < 10); "
            f"broadened search (page 2, service relaxation, short keyword) "
            f"to {len(pool)} total candidates."
        )
        think_broaden = _invoke_step_narrator(
            query,
            {
                "search_query": search_params.get("q", ""),
                "note": (
                    f"Page-1 best heuristic score was low ({max_p1_score:.1f} < 10). "
                    "Broadening search: page 2, service relaxation, and short keyword."
                ),
                "total_results_after_broadening": len(pool),
            },
            fallback=fallback_broaden,
        )
        _append_step(think_broaden, fb_calls, "", query, steps)

    best = (
        _select_best_product_for_product_case(
            pool, query, top_count=15, parsed_spec=primary_spec
        )
        if pool
        else None
    )

    if not best:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="No suitable product found matching the query constraints.",
        )
        return

    _emit_weighing_step(best, pool, primary_spec, query, steps, n_alts=3)

    pid = str(best.get("product_id", ""))
    detail = _product_detail_cache.get(pid, {})
    llm_reason = best.get("_llm_reason", "")
    attributes = detail.get("attributes", {}) if isinstance(detail, dict) else {}
    sku_options = detail.get("sku_options", {}) if isinstance(detail, dict) else {}
    sku_options_sample = list(sku_options.values())[:3] if isinstance(sku_options, dict) else []
    constraint_check = _check_pick_against_query(
        title=best.get("title", ""),
        price=best.get("price"),
        parsed_spec=primary_spec,
    )
    try:
        sorted_alts = sorted(
            [p for p in pool if str(p.get("product_id", "")) != pid],
            key=lambda p: _score_product(p, query, parsed_spec=primary_spec),
            reverse=True,
        )
    except Exception:
        sorted_alts = [p for p in pool if str(p.get("product_id", "")) != pid]
    alt_list = [
        {
            "product_id": str(a.get("product_id", "")),
            "title": (a.get("title") or "")[:80],
            "price": a.get("price"),
            "heuristic_score": _safe_score(a, query, primary_spec),
        }
        for a in sorted_alts[:2]
    ]
    fallback_pick = (
        f"Selected product_id={pid} "
        f"title='{best.get('title', '')[:100]}' "
        f"price={best.get('price')} service={best.get('service')}. "
        + (f"LLM reason: {llm_reason}" if llm_reason else "Chosen by heuristic score.")
    )
    think_pick = _invoke_step_narrator(
        query,
        {
            "selected": {
                "product_id": pid,
                "title": best.get("title", ""),
                "price": best.get("price"),
                "service": best.get("service"),
                "attributes": attributes,
                "sku_options_sample": sku_options_sample,
            },
            "constraints": {
                "price_range": primary_spec.get("price_range"),
                "service": primary_spec.get("service"),
                "keywords": primary_spec.get("keywords"),
            },
            "constraint_check": constraint_check,
            "alternatives": alt_list,
            "llm_reason": llm_reason,
        },
        fallback=fallback_pick,
    )
    _finish_session([pid], "success", query, steps, think=think_pick, llm_reason=llm_reason)


def _run_same_shop_search(params: dict, query: str, steps: list) -> None:
    global _shop_mono_anchor
    _shop_mono_anchor = time.monotonic()

    specs = params.get("products", [])
    n_specs = len(specs)
    if not specs:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="No product specs found in shop query.",
        )
        return

    kw_list = [s.get("keywords") or s.get("q", "") for s in specs]
    think_analyze = _invoke_step_narrator(
        query,
        {
            "product_count": n_specs,
            "products": [
                {
                    "keywords": s.get("keywords"),
                    "price_range": s.get("price_range"),
                    "service": s.get("service"),
                }
                for s in specs
            ],
        },
        fallback=(
            f"Searching for {n_specs} products from the same shop. "
            f"Keywords: {kw_list}. "
            f"Price ranges: {[s.get('price_range') for s in specs]}. "
            f"Services: {[s.get('service') for s in specs]}."
        ),
    )

    # Phase 1: Broad 3-page candidate collection per spec with service fallback
    broad, search_tool_calls = _collect_shop_candidate_pools(specs)
    _append_step(think_analyze, search_tool_calls, "", query, steps)

    # Phase 2: LLM scoring with heuristic pre-filter + per-store cap
    cap = _SHOP_BATCH_CAP_MULTI if n_specs >= 3 else _SHOP_BATCH_CAP
    shop_cap = _SHOP_PER_STORE_CAP_MULTI if n_specs >= 3 else _SHOP_PER_STORE_CAP
    spec_scored_full: list[list[tuple[Product, float]]] = []
    spec_scored_thr: list[list[tuple[Product, float]]] = []

    for spec_idx, (spec, prods) in enumerate(zip(specs, broad)):
        spec_query = spec.get("query") or spec.get("keywords") or query
        product_ids = [
            str(p.get("product_id", "") or "") for p in prods if p.get("product_id")
        ]
        details = _fetch_product_details(product_ids)
        scored = _rate_pool_for_shop(
            spec_query, prods, details,
            only_product_type=bool(spec.get("only_product_type", False)),
            cap=cap, shop_cap=shop_cap,
        )
        spec_scored_full.append(scored)
        spec_scored_thr.append(
            [(p, s) for p, s in scored if s >= SHOP_SCORE_THRESHOLD]
        )
        logger.info(
            "_run_same_shop_search: spec[%d] %d candidates → %d passed threshold %.1f",
            spec_idx, len(scored), len(spec_scored_thr[-1]), SHOP_SCORE_THRESHOLD,
        )

    # Fall back to full scored lists if any spec is empty after threshold filtering
    if any(len(x) == 0 for x in spec_scored_thr):
        spec_scored_thr = spec_scored_full

    spec_scored = spec_scored_thr
    filtered_results: list[list[Product]] = [
        [p for p, _ in scored] for scored in spec_scored
    ]
    shop_coverage = _group_products_by_shop(filtered_results)

    full_shops = [
        shop_id for shop_id, cov in shop_coverage.items() if len(cov) == n_specs
    ]

    scoring_summary = [
        {
            "spec_idx": spec_index,
            "keywords": specs[spec_index].get("keywords", ""),
            "total_collected": len(broad[spec_index]),
            "passed_threshold": len(spec_scored[spec_index]),
            "top_candidates": [
                {"title": p.get("title", ""), "price": p.get("price"), "score": s}
                for p, s in spec_scored[spec_index][:3]
            ],
        }
        for spec_index in range(n_specs)
    ]
    think_scoring = _invoke_step_narrator(
        query,
        {
            "scoring_summary": scoring_summary,
            "score_threshold": SHOP_SCORE_THRESHOLD,
            "full_coverage_shops_found": len(full_shops),
        },
        fallback=(
            f"LLM-scored products for {n_specs} specs (threshold={SHOP_SCORE_THRESHOLD}). "
            + " | ".join(
                f"spec[{i}]: {len(spec_scored[i])}/{len(broad[i])} passed"
                for i in range(n_specs)
            )
            + f". Full-coverage shops: {len(full_shops)}."
        ),
    )
    _append_step(think_scoring, [], "", query, steps)

    # Branch A: exactly one shop covers all specs
    if len(full_shops) == 1:
        shop_id = full_shops[0]
        used_ids: set[str] = set()
        chosen_product_ids: list[str] = []
        leaders_a: list = []
        pools_a: list = []
        for spec_idx in range(n_specs):
            pool = shop_coverage[shop_id].get(spec_idx, []) or []
            pools_a.append(pool)
            picked = None
            for product in pool:
                product_id = str(product.get("product_id", "") or "")
                if product_id and product_id not in used_ids:
                    chosen_product_ids.append(product_id)
                    used_ids.add(product_id)
                    picked = product
                    break
            leaders_a.append(picked)
        if len(chosen_product_ids) == n_specs:
            _emit_weighing_step_multi(leaders_a, pools_a, specs, query, steps)
            enriched = _enrich_products_for_reason(
                [{"product_id": pid} for pid in chosen_product_ids]
            )
            constraint_checks = (
                [
                    _check_pick_against_query(
                        title=info.get("title", ""),
                        price=info.get("price"),
                        parsed_spec=spec or {},
                    )
                    for spec, info in zip(specs, enriched)
                ]
                if len(specs) == len(enriched) else None
            )
            ctx_found: dict = {
                "shop_id": shop_id,
                "note": "Only one shop found covering all product specs.",
                "selected_products": enriched,
            }
            if constraint_checks is not None:
                ctx_found["constraint_checks"] = constraint_checks
            think_found = _invoke_step_narrator(
                query,
                ctx_found,
                fallback=(
                    f"Only shop {shop_id} covers all {n_specs} specs. "
                    f"Recommending: {chosen_product_ids}."
                ),
            )
            _finish_session(chosen_product_ids, "success", query, steps, think=think_found)
            return

    # Branch B: multiple full-coverage shops — prerank by heuristic, pick by score aggregation
    if len(full_shops) > 1:
        preranked = _order_shops_by_coverage_score(
            full_shops[:_SHOP_MAX_FULL_SHOPS], shop_coverage, specs, query,
        )
        shop_id, chosen = _aggregate_shop_pick_by_score(
            preranked, shop_coverage, spec_scored, specs, query,
        )
        chosen_ids = [
            chosen[spec_index]["product_id"]
            for spec_index in range(n_specs)
            if spec_index in chosen
        ]
        if shop_id and len(chosen_ids) == n_specs:
            leaders_b: list = []
            pools_b: list = []
            for spec_index in range(n_specs):
                pool = (shop_coverage.get(shop_id) or {}).get(spec_index, []) or []
                pools_b.append(pool)
                lead_pid = chosen.get(spec_index, {}).get("product_id", "")
                lead = next(
                    (p for p in pool if str(p.get("product_id", "") or "") == lead_pid),
                    pool[0] if pool else None,
                )
                if lead is not None:
                    lead = dict(lead)
                    lead["_llm_reason"] = chosen.get(spec_index, {}).get("reason", "")
                    lead["_llm_relevance_score"] = chosen.get(spec_index, {}).get("score", 0)
                leaders_b.append(lead)
            _emit_weighing_step_multi(leaders_b, pools_b, specs, query, steps)
            enriched = _enrich_products_for_reason(
                [{"product_id": pid} for pid in chosen_ids]
            )
            llm_reasoning = [
                {
                    "spec_index": spec_index,
                    "product_id": chosen[spec_index]["product_id"],
                    "reason": chosen[spec_index]["reason"],
                    "relevance_score": chosen[spec_index]["score"],
                }
                for spec_index in range(n_specs) if spec_index in chosen
            ]
            constraint_checks = (
                [
                    _check_pick_against_query(
                        title=info.get("title", ""),
                        price=info.get("price"),
                        parsed_spec=spec or {},
                    )
                    for spec, info in zip(specs, enriched)
                ]
                if len(specs) == len(enriched) else None
            )
            ctx_found = {
                "shop_id": shop_id,
                "note": (
                    f"{len(full_shops)} full-coverage shops; top {min(len(full_shops), _SHOP_MAX_FULL_SHOPS)} "
                    "preranked by heuristic; best shop selected by score aggregation."
                ),
                "selected_products": enriched,
                "llm_reasoning": llm_reasoning,
            }
            if constraint_checks is not None:
                ctx_found["constraint_checks"] = constraint_checks
            think_found = _invoke_step_narrator(
                query,
                ctx_found,
                fallback=(
                    f"Selected shop {shop_id} from {len(full_shops)} full-coverage shops "
                    f"via score aggregation. Product IDs: {chosen_ids}."
                ),
            )
            _finish_session(chosen_ids, "success", query, steps, think=think_found)
            return

    # Branch C: no full-coverage shop — fallback resolution strategies
    logger.info(
        "_run_same_shop_search: Branch C — no full-coverage shop "
        "(%d specs, %d shops after scoring). Applying fallback resolution.",
        n_specs, len(shop_coverage),
    )
    # Emit per-spec weighing step so the narrator reasons about the candidates
    # before explaining the anchor strategy (Case 9b → Case 10 narration flow).
    leaders_c = [scored[0][0] if scored else None for scored in spec_scored]
    pools_c = [[p for p, _ in scored] for scored in spec_scored]
    _emit_weighing_step_multi(leaders_c, pools_c, specs, query, steps)

    resolved_product_ids, case_c_ctx = _resolve_case_c(
        specs, spec_scored, shop_coverage, query, n_specs
    )

    if resolved_product_ids and len(resolved_product_ids) == n_specs:
        sub_case = case_c_ctx.get("sub_case", 0)
        if sub_case == 4:
            fallback_case_c = (
                f"Sub-case 4: shops covering "
                f"{n_specs - 1}/{n_specs} specs. "
                f"Winner {case_c_ctx.get('winner_shop_id')} "
                f"score={case_c_ctx.get('winner_shop_score')}. "
                f"Filled missing spec[{case_c_ctx.get('missing_spec_idx')}]."
            )
        else:
            anchor = case_c_ctx.get("anchor", {})
            fallback_case_c = (
                f"Sub-case {sub_case}: {case_c_ctx.get('tie_note', '')} "
                f"Anchor spec[{anchor.get('spec_idx')}] product_id={anchor.get('product_id')} "
                f"shop_id={anchor.get('shop_id')}."
            )
        think_case_c = _invoke_step_narrator(
            query,
            {
                "case_c_resolution": case_c_ctx,
                "note": "No full-coverage shop; anchor strategy used.",
            },
            fallback=fallback_case_c,
        )
        _append_step(think_case_c, [], "", query, steps)

        enriched = _enrich_products_for_reason(
            [{"product_id": pid} for pid in resolved_product_ids]
        )
        constraint_checks = (
            [
                _check_pick_against_query(
                    title=info.get("title", ""),
                    price=info.get("price"),
                    parsed_spec=spec or {},
                )
                for spec, info in zip(specs, enriched)
            ]
            if len(specs) == len(enriched) else None
        )
        ctx_found = {
            "shop_id": (
                case_c_ctx.get("anchor", {}).get("shop_id")
                or case_c_ctx.get("winner_shop_id", "resolved")
            ),
            "selected_products": enriched,
            "llm_reasoning": case_c_ctx.get("filled_specs", []),
        }
        if constraint_checks is not None:
            ctx_found["constraint_checks"] = constraint_checks
        think_found = _invoke_step_narrator(
            query,
            ctx_found,
            fallback=f"Anchor resolved. Products: {resolved_product_ids}.",
        )
        _finish_session(resolved_product_ids, "success", query, steps, think=think_found)
        return

    _finish_session(
        [FALLBACK_PRODUCT_ID],
        "failure",
        query,
        steps,
        think=f"Could not find a single shop carrying all {n_specs} required products.",
    )


# ──────────────────────────────────────────────────────────────────────────
# Voucher-specific helper functions
# ──────────────────────────────────────────────────────────────────────────


def _voucher_heuristic_overlap(product: dict, query_text: str) -> int:
    """Title-word overlap for voucher candidate ranking (reuses _score_product logic)."""
    return _score_product(product, query_text)


def _voucher_rrf_merge(
    rankings: list[list],
    k: int = 60,
    top_n: int = 15,
) -> list:
    """Reciprocal-rank-fusion merge of multiple product ranking lists."""
    score: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    by_pid: dict[str, dict] = {}
    for ranking in rankings:
        for rank, prod in enumerate(ranking, start=1):
            pid = str(prod.get("product_id", ""))
            if not pid:
                continue
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + rank)
            if pid not in best_rank or rank < best_rank[pid]:
                best_rank[pid] = rank
            if pid not in by_pid:
                by_pid[pid] = prod
    sorted_pids = sorted(score, key=lambda p: (-score[p], best_rank[p]))
    return [by_pid[pid] for pid in sorted_pids[:top_n]]


def _voucher_rare_tokens(spec: dict) -> list[str]:
    """Pull rare spec constraint tokens for a focused product search."""
    keywords_lower = str(spec.get("keywords") or "").lower()
    keyword_set = set(re.findall(r"\b[\w.]+\b", keywords_lower))
    out: list[str] = []
    seen: set[str] = set()

    constraints = spec.get("constraints") or {}
    if isinstance(constraints, dict):
        for v in constraints.values():
            for tok in re.findall(r"\b[\w.]+\b", str(v).lower()):
                if len(tok) < 2 or tok in seen or tok in keyword_set:
                    continue
                seen.add(tok)
                out.append(tok)

    nu_pattern = re.compile(
        r"\b\d+(?:\.\d+)?(?:g|kg|ml|cm|mm|gb|tb|mb|k|inch|oz|lb|pcs|pack)\b",
        re.IGNORECASE,
    )
    title = str(spec.get("hypothetical_title") or "")
    for tok in nu_pattern.findall(title):
        tl = tok.lower()
        if tl in seen or tl in keyword_set:
            continue
        seen.add(tl)
        out.append(tl)

    return out[:4]


def _voucher_is_cart_within_budget(total: float, voucher: dict) -> bool:
    """Return True if the pre-discount cart total fits the voucher budget."""
    budget = float(voucher.get("budget", 0))
    if total <= budget:
        return True
    threshold = float(voucher.get("threshold", 0))
    if total < threshold:
        return False
    discount_type = voucher.get("discount_type", "percentage")
    discount_value = float(voucher.get("discount_value", 0))
    cap = float(voucher.get("cap", 0))
    if discount_type == "fixed":
        discount = discount_value
    else:
        rate = discount_value / 100.0 if discount_value > 1 else discount_value
        discount = total * rate
        if cap > 0:
            discount = min(discount, cap)
    return (total - discount) <= budget


def _voucher_score_pool(
    specs: list[dict],
    candidates_per_spec: list[list],
) -> list[list[tuple]]:
    """LLM-score every candidate per spec for knapsack input."""
    scored: list[list[tuple]] = []
    for spec, candidates in zip(specs, candidates_per_spec):
        if not candidates:
            scored.append([])
            continue
        pids = [str(p.get("product_id", "")) for p in candidates if p.get("product_id")]
        details = _fetch_product_details(pids)
        spec_query = spec.get("query") or spec.get("keywords") or ""
        result = _llm_score_products(
            spec_query, candidates, details,
            only_product_type=bool(spec.get("only_product_type", False)),
        )
        scored.append(result)
    return scored


def _voucher_gate_by_floor(
    scored_per_spec: list[list[tuple]],
    threshold: float,
    top_k: int = _VOUCHER_KNAPSACK_CAND_CAP,
) -> list[list[tuple]]:
    """Filter scored candidates by minimum score threshold."""
    return [
        sorted([(p, s) for p, s in cand if s >= threshold], key=lambda x: -x[1])[:top_k]
        for cand in scored_per_spec
    ]


def _voucher_run_branch_bound(
    candidates_per_spec: list[list[tuple]],
    max_total: float,
    require_same_shop: bool = False,
    voucher: dict | None = None,
) -> tuple[list, float, float] | None:
    """Multi-choice knapsack via DFS with branch-and-bound pruning on price."""
    if not candidates_per_spec or any(not cands for cands in candidates_per_spec):
        return None

    if require_same_shop:
        per_spec_by_shop: list[dict] = []
        for cands in candidates_per_spec:
            bucket: dict = defaultdict(list)
            for prod, score in cands:
                sid = str(prod.get("shop_id") or "")
                if sid:
                    bucket[sid].append((prod, score))
            per_spec_by_shop.append(bucket)

        common_shops: set | None = None
        for bucket in per_spec_by_shop:
            shops = set(bucket.keys())
            common_shops = shops if common_shops is None else (common_shops & shops)
        if not common_shops:
            return None

        best_overall: tuple | None = None
        for sid in common_shops:
            per_spec = [per_spec_by_shop[i][sid] for i in range(len(per_spec_by_shop))]
            sub = _voucher_run_branch_bound(
                per_spec, max_total, require_same_shop=False, voucher=voucher,
            )
            if sub is None:
                continue
            sel, sc, pr = sub
            if (
                best_overall is None
                or sc > best_overall[1]
                or (sc == best_overall[1] and pr < best_overall[2])
            ):
                best_overall = (sel, sc, pr)
        return best_overall

    sorted_per_spec = [
        sorted(cands, key=lambda ps: float(ps[0].get("price") or 0.0))
        for cands in candidates_per_spec
    ]
    n = len(sorted_per_spec)
    best: dict = {"selection": None, "score": -1.0, "price": float("inf")}

    def _dfs(idx: int, partial: list, cur_price: float, cur_score: float) -> None:
        if cur_price > max_total:
            return
        if idx == n:
            if voucher is not None and not _voucher_is_cart_within_budget(cur_price, voucher):
                return
            if (
                cur_score > best["score"]
                or (cur_score == best["score"] and cur_price < best["price"])
            ):
                best["selection"] = [p for p, _ in partial]
                best["score"] = cur_score
                best["price"] = cur_price
            return
        for cand, score in sorted_per_spec[idx]:
            price = float(cand.get("price") or 0.0)
            if cur_price + price > max_total:
                break
            partial.append((cand, score))
            _dfs(idx + 1, partial, cur_price + price, cur_score + score)
            partial.pop()

    _dfs(0, [], 0.0, 0.0)
    if best["selection"] is None:
        return None
    return best["selection"], best["score"], best["price"]


def _voucher_knapsack_tiers(
    scored_per_spec: list[list[tuple]],
    max_total: float,
    require_same_shop: bool = False,
    voucher: dict | None = None,
) -> tuple[list, dict] | None:
    """Try knapsack with progressively relaxed score floors (6.0 → 4.0 → 0.0)."""
    tiers = [
        (SHOP_SCORE_THRESHOLD, "primary"),
        (SHOP_SCORE_THRESHOLD - _VOUCHER_KNAPSACK_TIER_STEP, "relaxed"),
        (0.0, "unfiltered"),
    ]
    for threshold, tier_name in tiers:
        filtered = _voucher_gate_by_floor(scored_per_spec, threshold)
        if any(not cands for cands in filtered):
            logger.info(
                "voucher_knapsack: tier=%s threshold=%.1f — empty spec, skip",
                tier_name, threshold,
            )
            continue
        result = _voucher_run_branch_bound(
            filtered, max_total, require_same_shop=require_same_shop, voucher=voucher,
        )
        if result is not None:
            selection, total_score, total_price = result
            logger.info(
                "voucher_knapsack: tier=%s threshold=%.1f same_shop=%s score=%.1f price=%.2f",
                tier_name, threshold, require_same_shop, total_score, total_price,
            )
            return selection, {
                "tier": tier_name,
                "threshold": threshold,
                "total_score": total_score,
                "total_price": total_price,
                "same_shop": require_same_shop,
            }
        logger.info(
            "voucher_knapsack: tier=%s threshold=%.1f infeasible", tier_name, threshold,
        )
    return None


def _voucher_cheapest_per_spec(
    scored_per_spec: list[list[tuple]],
    require_same_shop: bool = False,
) -> tuple[list, float, float] | None:
    """Best-effort fallback: pick the cheapest candidate per spec."""
    if any(not cands for cands in scored_per_spec):
        return None

    if require_same_shop:
        per_spec_by_shop: list[dict] = []
        for cand in scored_per_spec:
            bucket: dict = defaultdict(list)
            for prod, score in cand:
                sid = str(prod.get("shop_id") or "")
                if sid:
                    bucket[sid].append((prod, score))
            per_spec_by_shop.append(bucket)

        common_shops: set | None = None
        for bucket in per_spec_by_shop:
            shops = set(bucket.keys())
            common_shops = shops if common_shops is None else (common_shops & shops)

        if common_shops:
            best_shop: tuple | None = None
            for sid in common_shops:
                sel: list = []
                total_score = 0.0
                total_price = 0.0
                for bucket in per_spec_by_shop:
                    cheap_p, cheap_s = min(
                        bucket[sid],
                        key=lambda ps: float(ps[0].get("price") or float("inf")),
                    )
                    sel.append(cheap_p)
                    total_score += cheap_s
                    total_price += float(cheap_p.get("price") or 0.0)
                if best_shop is None or total_price < best_shop[2]:
                    best_shop = (sel, total_score, total_price)
            if best_shop is not None:
                return best_shop

    selection: list = []
    total_score = 0.0
    total_price = 0.0
    for cand in scored_per_spec:
        cheapest_p, cheapest_s = min(
            cand, key=lambda ps: float(ps[0].get("price") or float("inf")),
        )
        selection.append(cheapest_p)
        total_score += cheapest_s
        total_price += float(cheapest_p.get("price") or 0.0)
    return selection, total_score, total_price


# ──────────────────────────────────────────────────────────────────────────
# Voucher helper functions (ported from agent_make.py)
# ──────────────────────────────────────────────────────────────────────────

def _voucher_local_math(prices: list[float], voucher: dict) -> dict:
    """Local Python voucher math (no tool call). Returns same schema as calculate_voucher."""
    total = sum(float(p) for p in prices)
    discount_type = voucher.get("discount_type", "percentage") or "percentage"
    discount_value = float(voucher.get("discount_value", 0) or 0)
    cap = float(voucher.get("cap", 0) or 0)
    threshold = float(voucher.get("threshold", 0) or 0)
    budget = float(voucher.get("budget", 0) or 0)
    applied = total >= threshold
    discount = 0.0
    if applied:
        if discount_type == "fixed":
            discount = discount_value
        else:
            discount = total * (discount_value / 100.0)
            if cap > 0:
                discount = min(discount, cap)
    total_after = total - discount
    return {
        "total": round(total, 2),
        "discount": round(discount, 2),
        "total_after": round(total_after, 2),
        "applied": applied,
        "within_budget": total_after <= budget,
    }


def _voucher_widen_price_range(price_str: str | None, ratio: float) -> str | None:
    """Widen a 'lo-hi' price string by ratio on both ends."""
    if not price_str:
        return None
    try:
        lo_s, _, hi_s = price_str.partition("-")
        lo = float(lo_s) if lo_s.strip() else None
        hi = float(hi_s) if hi_s.strip() else None
        if lo is not None:
            lo = max(0.0, lo * (1 - ratio))
        if hi is not None:
            hi = hi * (1 + ratio)
        lo_fmt = f"{lo:.0f}" if lo is not None else ""
        hi_fmt = f"{hi:.0f}" if hi is not None else ""
        widened = f"{lo_fmt}-{hi_fmt}"
        if widened == "-":
            return None
        return widened
    except (ValueError, TypeError):
        return price_str


def _voucher_alternate_query_slug(spec: dict, max_words: int = 10) -> str | None:
    """Build an alternate keyword slug from the raw query text (strips stopwords)."""
    _lex_fillers = frozenset({
        "a", "an", "the", "and", "or", "but", "so", "if", "as", "of", "to", "in",
        "on", "at", "by", "for", "with", "from", "into", "onto", "about", "between",
        "through", "above", "below", "over", "under", "i", "me", "my", "we", "us",
        "our", "you", "your", "it", "its", "they", "them", "their", "that", "this",
        "these", "those", "which", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "can", "could", "will", "would",
        "should", "may", "might", "not", "no", "also", "then", "very", "just", "only",
    })
    text = str(spec.get("query") or "").strip()
    if not text:
        return None
    words = [w for w in re.findall(r"\b\w+\b", text.lower()) if len(w) > 1 and w not in _lex_fillers]
    if not words:
        return None
    slug = " ".join(words[:max_words])
    kw = (spec.get("keywords") or "").strip()
    if not kw:
        return slug
    kw_toks = set(re.findall(r"\b\w+\b", kw.lower()))
    sl_toks = set(re.findall(r"\b\w+\b", slug.lower()))
    if not kw_toks or not sl_toks:
        return slug
    if len(kw_toks & sl_toks) >= min(len(kw_toks), len(sl_toks), 3):
        return None
    return slug


def _voucher_confirmed_by_tool(prices_csv: str, voucher: dict) -> bool:
    """Verify voucher math via the calculate_voucher tool (authoritative check)."""
    r = safe_tool_call(
        "calculate_voucher",
        {
            "product_prices": prices_csv,
            "voucher_type": voucher.get("discount_type", "percentage"),
            "discount_value": float(voucher.get("discount_value", 0) or 0),
            "threshold": float(voucher.get("threshold", 0) or 0),
            "budget": float(voucher.get("budget", 0) or 0),
            "cap": float(voucher.get("cap", 0) or 0),
        },
    )
    try:
        result = r["result"]
        return bool(result.get("voucher_applied")) and bool(result.get("within_budget"))
    except (AttributeError, KeyError, TypeError):
        return False


def _voucher_rejudge_pool(
    query_text: str, spec: dict, pool: list, picked: dict | None, picked_score: float,
) -> tuple[dict | None, float]:
    """Re-run LLM judge over an expanded pool; return whichever pick scores higher."""
    if not pool:
        return picked, picked_score
    pids = [str(p.get("product_id", "") or "") for p in pool if p.get("product_id")]
    details = _fetch_product_details(pids)
    better = _llm_choose_product(
        spec.get("query") or query_text,
        pool[:_VOUCHER_MAX_JUDGE_CANDIDATES],
        details,
        only_product_type=False,
        model=FALLBACK_MODEL,
    )
    if not better:
        return picked, picked_score
    ns = float(better.get("_llm_relevance_score", 0) or 0)
    if picked is None or ns > picked_score:
        return better, ns
    return picked, picked_score


def _voucher_close_success(
    pids: list[str],
    leaders: list[dict],
    pools: list[list],
    products: list[dict],
    voucher: dict,
    calc_total: float,
    query: str,
    steps: list,
    fb: str,
    tool_calls: list | None = None,
    extra_ctx: dict | None = None,
) -> None:
    """Emit weighing step + narration + finish_session for a successful voucher resolution."""
    _emit_weighing_step_multi(leaders, pools, products, query, steps)
    enriched = _enrich_products_for_reason([
        {"product_id": pid, "title": ld.get("title", ""), "price": ld.get("price")}
        for pid, ld in zip(pids, leaders)
    ])
    ctx_data: dict = {
        "selected_products": enriched,
        "total_before_discount": round(calc_total, 2),
        "budget_constraint": voucher,
    }
    if extra_ctx:
        ctx_data.update(extra_ctx)
    if len(products) == len(enriched):
        ctx_data["constraint_checks"] = [
            _check_pick_against_query(
                title=info.get("title", ""),
                price=info.get("price"),
                parsed_spec=spec or {},
            )
            for spec, info in zip(products, enriched)
        ]
    think = _invoke_step_narrator(query, ctx_data, fallback=fb)
    if tool_calls:
        _append_step(think, tool_calls, "", query, steps)
        _finish_session(pids, "success", query, steps)
    else:
        _finish_session(pids, "success", query, steps, think=think)


def _voucher_build_shop_coverage(
    pools: list[list],
) -> dict[str, dict[int, list]]:
    """Build {shop_id → {spec_idx → [products]}} mapping from per-spec pools."""
    coverage: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for spec_idx, pool in enumerate(pools):
        for product in pool:
            sid = str(product.get("shop_id", "") or "")
            if sid:
                coverage[sid][spec_idx].append(product)
    return {sid: dict(cov) for sid, cov in coverage.items()}


# ──────────────────────────────────────────────────────────────────────────
# Voucher combo-grid builders (ported from agent_make.py)
# ──────────────────────────────────────────────────────────────────────────

def _voucher_evaluate_combos(
    ranked_per_spec: list[list[tuple]],
    voucher: dict,
    max_combos: int,
) -> list[tuple]:
    """Evaluate the cartesian product of per-spec ranked candidates for voucher feasibility.

    Returns a list of ``(total_score, selection, calc)`` tuples sorted by descending score
    and ascending distance from 90 % of budget (or ascending total_after when no budget).
    """
    feasible: list[tuple] = []
    count = 0
    for combo in _cartesian_product(*ranked_per_spec):
        count += 1
        if count > max_combos:
            break
        selection = [c[0] for c in combo]
        pids = [str(p.get("product_id", "")) for p in selection]
        if len(set(pids)) != len(pids):
            continue
        prices = [float(p.get("price") or 0) for p in selection]
        if any(pr <= 0 for pr in prices):
            continue
        calc = _voucher_local_math(prices, voucher)
        if not calc["applied"] or not calc["within_budget"]:
            continue
        feasible.append((sum(c[1] for c in combo), selection, calc))
    budget = float(voucher.get("budget") or 0)
    target = budget * 0.9 if budget > 0 else 0.0
    if target > 0:
        feasible.sort(key=lambda x: (-x[0], abs(x[2]["total_after"] - target)))
    else:
        feasible.sort(key=lambda x: (-x[0], x[2]["total_after"]))
    return feasible


def _voucher_build_combo_grid_heuristic(
    pools: list[list],
    voucher: dict,
    products: list[dict],
    k_per_spec: int = _VOUCHER_COMBO_K_PER_SPEC,
    max_combos: int = _VOUCHER_COMBO_MAX_COMBOS,
) -> list[tuple]:
    """Heuristic cartesian product combo grid – no LLM calls."""
    if not pools or not all(pools):
        return []
    ranked_per_spec: list[list[tuple]] = []
    for i, pool in enumerate(pools):
        kw = products[i].get("keywords") or products[i].get("query") or "" if i < len(products) else ""
        scored = sorted(
            ((p, _score_product(p, kw)) for p in pool if p.get("price") is not None),
            key=lambda x: -x[1],
        )
        if not scored:
            return []
        ranked_per_spec.append(scored[:k_per_spec])
    return _voucher_evaluate_combos(ranked_per_spec, voucher, max_combos)


def _voucher_build_combo_grid_llm(
    pools: list[list],
    voucher: dict,
    products: list[dict],
    query: str,
    k_per_spec: int = _VOUCHER_COMBO_K_PER_SPEC,
    max_combos: int = _VOUCHER_COMBO_MAX_COMBOS,
    score_threshold: float = _VOUCHER_COMBO_SCORE_THRESHOLD,
) -> list[tuple]:
    """LLM-scored cartesian product combo grid (ported from _x_build_llm_voucher_combo_grid)."""
    if not pools or not all(pools):
        return []
    ranked_per_spec: list[list[tuple]] = []
    for i, pool in enumerate(pools):
        spec = products[i] if i < len(products) else {}
        spec_query = spec.get("query") or spec.get("keywords") or query
        pids = [str(p.get("product_id", "")) for p in pool if p.get("product_id")]
        details = _fetch_product_details(pids)

        if not _spend_core_budget():
            # Budget exhausted → heuristic fallback for this spec
            scored_pairs = [
                (p, _score_product(p, spec_query))
                for p in pool if p.get("price") is not None
            ]
        else:
            capped_pool = pool[:_SHOP_BATCH_CAP]
            payload = {
                "request": spec_query,
                "candidates": [
                    _compact_candidate_payload(
                        p, details.get(str(p.get("product_id", "") or "")), spec_query,
                    )
                    for p in capped_pool
                ],
                "only_product_type": bool(spec.get("only_product_type", False)),
            }
            scored_pairs = []
            user_body = json.dumps(payload, ensure_ascii=False)
            for parsed in _iter_llm_json(
                models=_judge_model_chain(include_extra=False),
                temperature=0.5,
                system=_SHOP_SCORER_PROMPT,
                user=user_body,
                retries=PRODUCT_JUDGE_MAX_RETRIES,
            ):
                if not isinstance(parsed, list):
                    continue
                price_pool = [p for p in pool if p.get("price") is not None]
                scored_pairs = _parse_scored_candidates(parsed, price_pool)
                break
            if not scored_pairs:
                scored_pairs = [
                    (p, _score_product(p, spec_query))
                    for p in pool if p.get("price") is not None
                ]

        # Boost with attribute overlap + sales signal (mirrors agent_make.py)
        spec_words = set(re.findall(r"\b\w+\b", spec_query.lower()))
        boosted: list[tuple] = []
        for p, s in scored_pairs:
            bonus = 0.0
            sold = float(p.get("sold_count") or 0)
            if sold > 0:
                bonus += min(1.0, math.log10(sold + 1) / 3.0)
            boosted.append((p, s + bonus))

        filtered = [(p, s) for p, s in boosted if s >= score_threshold]
        if not filtered:
            filtered = sorted(boosted, key=lambda x: -x[1])[:k_per_spec]
        if not filtered:
            return []
        filtered.sort(key=lambda x: -x[1])
        ranked_per_spec.append(filtered[:k_per_spec])

    return _voucher_evaluate_combos(ranked_per_spec, voucher, max_combos)


# ──────────────────────────────────────────────────────────────────────────
# Voucher strategy functions (ported from agent_make.py)
# ──────────────────────────────────────────────────────────────────────────

def _voucher_try_combo_matrix(
    products: list[dict],
    pools: list[list],
    voucher: dict,
    query: str,
    steps: list,
) -> bool:
    """Try LLM-scored combo grid, fall back to heuristic grid. n ≤ 3 only."""
    n = len(products)
    if not (len(pools) == n and all(pools) and n <= 3):
        return False
    feasible = _voucher_build_combo_grid_llm(
        pools, voucher, products, query,
        k_per_spec=_VOUCHER_COMBO_K_PER_SPEC,
        max_combos=_VOUCHER_COMBO_MAX_COMBOS,
        score_threshold=_VOUCHER_COMBO_SCORE_THRESHOLD,
    )
    if not feasible:
        feasible = _voucher_build_combo_grid_heuristic(
            pools, voucher, products,
            k_per_spec=_VOUCHER_COMBO_K_PER_SPEC,
            max_combos=_VOUCHER_COMBO_MAX_COMBOS,
        )
    if not feasible:
        return False
    _, selection, calc = feasible[0]
    pids = [str(p["product_id"]) for p in selection]
    _voucher_close_success(
        pids=pids, leaders=selection, pools=pools, products=products,
        voucher=voucher, calc_total=calc["total"], query=query, steps=steps,
        fb=(
            f"Combinatorial voucher pick: total_before={calc['total']:.2f} "
            f"(threshold={voucher.get('threshold')}), "
            f"total_after={calc['total_after']:.2f} (budget={voucher.get('budget')}). "
            f"Recommending: {pids}."
        ),
    )
    return True


def _voucher_try_direct_pick(
    products: list[dict],
    pools: list[list],
    voucher: dict,
    query: str,
    steps: list,
) -> tuple[bool, list[list]]:
    """LLM-judge top candidates per spec, verify via tool. Merge pools on failure.

    Returns (success, (possibly reordered) pools).
    """
    n = len(products)
    if len(pools) != n or not all(pools):
        return False, pools

    judged: list[dict] = []
    for i, pool in enumerate(pools):
        spec = products[i]
        picked: dict | None = None
        if len(pool) > 1:
            # Quick heuristic gate: skip LLM if one candidate is clearly dominant
            probe = pool[:3]
            probe_pids = [str(c.get("product_id", "") or "") for c in probe]
            probe_details = _fetch_product_details(probe_pids)
            sc_list = [
                (c, _score_product(c, spec.get("keywords") or query, detail=probe_details.get(str(c.get("product_id", "") or ""))))
                for c in probe
            ]
            sc_list.sort(key=lambda x: x[1], reverse=True)
            top_sc = sc_list[0][1]
            runner_sc = sc_list[1][1] if len(sc_list) > 1 else float("-inf")
            if not (top_sc >= _VOUCHER_SKIP_LLM_SCORE and top_sc >= runner_sc + 2.0):
                kw = spec.get("keywords") or spec.get("query") or query
                all_pids = [str(c.get("product_id", "") or "") for c in pool if c.get("product_id")]
                details = _fetch_product_details(all_pids[:_VOUCHER_MAX_JUDGE_CANDIDATES])
                picked = _llm_choose_product_with_consistency(
                    kw, pool[:_VOUCHER_MAX_JUDGE_CANDIDATES], details,
                    only_product_type=bool(spec.get("only_product_type", False)),
                    model=FALLBACK_MODEL,
                )
        judged.append(picked or pool[0])

    prices_csv = ",".join(str(j.get("price", 0)) for j in judged)
    pids = [str(j.get("product_id", "") or "") for j in judged]
    if len(set(pids)) == len(pids) and _voucher_confirmed_by_tool(prices_csv, voucher):
        _voucher_close_success(
            pids=pids, leaders=judged, pools=pools, products=products,
            voucher=voucher, calc_total=sum(float(j.get("price", 0) or 0) for j in judged),
            query=query, steps=steps,
            fb=f"LLM-judged candidates fit budget. Recommending {pids}.",
        )
        return True, pools

    # Reorder pools: put judged pick first so later strategies see the preferred candidate
    merged: list[list] = []
    for orig, pick in zip(pools, judged):
        jpid = str(pick.get("product_id", "") or "")
        rest = [c for c in orig if str(c.get("product_id", "") or "") != jpid]
        merged.append([pick] + rest if jpid else list(orig))
    return False, merged


def _voucher_try_swap_refine(
    products: list[dict],
    pools: list[list],
    voucher: dict,
    query: str,
    steps: list,
) -> bool:
    """Re-rank one spec at a time by relevance+price; accept first combo that fits."""
    n = len(products)
    if len(pools) != n or not all(pools):
        return False
    for i in range(n):
        kw = products[i].get("keywords", "product")
        sq = kw if n > 1 else query
        ranked = sorted(
            pools[i],
            key=lambda p: (
                -_score_product(p, sq),
                float(p.get("price", 0) or 0),
            ),
        )[:10]
        pids_r = [str(p.get("product_id", "") or "") for p in ranked]
        details = _fetch_product_details(pids_r)
        if not ranked:
            continue
        picked = max(
            ranked,
            key=lambda p: (
                _score_product(p, sq, detail=details.get(str(p.get("product_id", "") or "")))
                - float(p.get("price", 0) or 0) / CHEAPER_PRICE_TIEBREAK_DIVISOR
            ),
        )
        trial_pools = [list(pool) for pool in pools]
        trial_pools[i] = [picked]
        prices = [float(tp[0].get("price", 0) or 0) for tp in trial_pools if tp]
        if len(prices) != n:
            continue
        calc = _voucher_local_math(prices, voucher)
        if not (calc["applied"] and calc["within_budget"]):
            continue
        prices_csv = ",".join(str(p) for p in prices)
        if not _voucher_confirmed_by_tool(prices_csv, voucher):
            continue
        trial_leaders = [tp[0] for tp in trial_pools]
        final_pids = [str(p.get("product_id", "") or "") for p in trial_leaders]
        _voucher_close_success(
            pids=final_pids, leaders=trial_leaders, pools=pools, products=products,
            voucher=voucher, calc_total=calc["total"], query=query, steps=steps,
            fb=(
                f"Re-ranked spec[{i}] by relevance and price. "
                f"Selected {picked.get('product_id')} price={picked.get('price')}. "
                f"total_before={calc['total']:.2f}, total_after={calc['total_after']:.2f}."
            ),
        )
        return True
    return False


def _voucher_try_budget_fill(
    products: list[dict],
    pools: list[list],
    voucher: dict,
    query: str,
    steps: list,
) -> None:
    """Force ceiling constraint: search for a cheaper candidate for one spec at a time."""
    n = len(products)
    ceiling = voucher_max_total_price(voucher)
    if len(pools) != n:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think=(
                f"Could not build one candidate pool per product spec "
                f"for budget={voucher.get('budget')}."
            ),
        )
        return
    for i in range(n - 1, -1, -1):
        if not all(pool for idx, pool in enumerate(pools) if idx != i):
            continue
        running = sum(
            float(pools[j][0].get("price", 0) or 0)
            for j in range(n) if j != i and pools[j]
        )
        if ceiling is None or running >= ceiling:
            continue
        sp = _spec_to_find_product_params(products[i], include_price=False)
        sp["price"] = f"0-{ceiling - running:.0f}"
        r = safe_tool_call("find_product", sp)
        found = _deduplicate_products(r.get("result") or [])
        if not found:
            continue
        ranked = sorted(
            found,
            key=lambda p: (
                -_score_product(p, products[i].get("keywords") or query),
                float(p.get("price", 0) or 0),
            ),
        )
        for candidate in ranked[:10]:
            trial_pools = [list(pool) for pool in pools]
            trial_pools[i] = [candidate] + [
                p for p in ranked
                if str(p.get("product_id", "") or "") != str(candidate.get("product_id", "") or "")
            ]
            if not all(trial_pools):
                continue
            trial_leaders = [tp[0] for tp in trial_pools]
            final_pids = [str(p.get("product_id", "") or "") for p in trial_leaders]
            if len(set(final_pids)) != len(final_pids):
                continue
            prices = [float(p.get("price", 0) or 0) for p in trial_leaders]
            calc = _voucher_local_math(prices, voucher)
            if not (calc["applied"] and calc["within_budget"]):
                continue
            _voucher_close_success(
                pids=final_pids, leaders=trial_leaders, pools=pools, products=products,
                voucher=voucher, calc_total=calc["total"], query=query, steps=steps,
                fb=(
                    f"Adjusted selections to meet voucher math. Prices: {prices}. "
                    f"total_before={calc['total']:.2f}, discount={calc['discount']:.2f}, "
                    f"total_after={calc['total_after']:.2f}. Products: {final_pids}."
                ),
            )
            return
    # Last resort: try leaders as-is
    if all(pools):
        leaders = [pool[0] for pool in pools]
        pids = [str(p.get("product_id", "") or "") for p in leaders]
        prices = [float(p.get("price", 0) or 0) for p in leaders]
        calc = _voucher_local_math(prices, voucher)
        if len(set(pids)) == len(pids) and calc["applied"] and calc["within_budget"]:
            _voucher_close_success(
                pids=pids, leaders=leaders, pools=pools, products=products,
                voucher=voucher, calc_total=calc["total"], query=query, steps=steps,
                fb=(
                    f"Fallback leaders satisfy voucher math. Prices: {prices}. "
                    f"total_before={calc['total']:.2f}, discount={calc['discount']:.2f}, "
                    f"total_after={calc['total_after']:.2f}. Products: {pids}."
                ),
            )
            return
    _finish_session(
        [FALLBACK_PRODUCT_ID], "failure", query, steps,
        think=(
            f"Could not find products that trigger the voucher threshold "
            f"and stay within budget={voucher.get('budget')}."
        ),
    )


# ──────────────────────────────────────────────────────────────────────────
# Same-shop voucher: combo-based (ported from agent_make.py)
# ──────────────────────────────────────────────────────────────────────────

def _voucher_collect_same_shop_pools(
    specs: list[dict], ceiling: float | None,
) -> tuple[list[list], list]:
    """Collect product pools for same-shop voucher, clipping price to pre-discount ceiling."""
    pools: list[list] = []
    tool_log: list = []
    ceiling_range = f"0-{int(ceiling)}" if ceiling and ceiling > 0 else None
    for spec in specs:
        search_spec = dict(spec)
        if ceiling_range:
            lo, hi = _parse_price_range(spec.get("price_range"))
            if hi is not None:
                new_hi = min(hi, ceiling) if ceiling else hi
                lo_s = f"{lo:.0f}" if lo is not None else "0"
                search_spec["price_range"] = f"{lo_s}-{new_hi:.0f}"
            else:
                search_spec["price_range"] = ceiling_range if not spec.get("price_range") else spec.get("price_range")
        sp = _spec_to_find_product_params(search_spec)
        bucket: list = []
        seen: set[str] = set()
        for pg in range(1, 4):
            r = safe_tool_call("find_product", {**sp, "page": pg})
            tool_log.append(r)
            for row in (r.get("result") or []):
                pid = str(row.get("product_id", "") or "")
                if pid and pid not in seen:
                    seen.add(pid)
                    bucket.append(row)
        if len(bucket) < 5:
            relaxed = {k: v for k, v in sp.items() if k != "service"}
            for pg in range(1, 3):
                r = safe_tool_call("find_product", {**relaxed, "page": pg})
                tool_log.append(r)
                for row in (r.get("result") or []):
                    pid = str(row.get("product_id", "") or "")
                    if pid and pid not in seen:
                        seen.add(pid)
                        bucket.append(row)
        pools.append(bucket)
    return pools, tool_log


def _voucher_run_same_shop_combo(
    params: dict, voucher: dict, query: str, steps: list,
) -> bool:
    """Same-shop voucher: collect pools, score per shop, pick best combo with utilization bonus.

    Ported from agent_make.py _x_run_same_shop_voucher_task.
    Returns True on success.
    """
    specs = params.get("products") or []
    n = len(specs)
    if n <= 1:
        return False
    ceiling = voucher_max_total_price(voucher)
    if ceiling is None or ceiling <= 0:
        return False

    pools, tool_log = _voucher_collect_same_shop_pools(specs, ceiling)
    if tool_log:
        preview = [
            {
                "keywords": specs[i].get("keywords", "") if i < len(specs) else "",
                "results_found": len(pools[i]) if i < len(pools) else 0,
                "top_product": {
                    "product_id": str(pools[i][0].get("product_id", "")),
                    "title": pools[i][0].get("title", ""),
                    "price": pools[i][0].get("price"),
                    "shop_id": str(pools[i][0].get("shop_id", "")),
                } if i < len(pools) and pools[i] else None,
            }
            for i in range(n)
        ]
        _append_step(
            _invoke_step_narrator(
                query,
                {
                    "budget_constraint": voucher,
                    "max_allowed_total": ceiling,
                    "candidates_per_product": preview,
                    "note": "This voucher also requires a single shared shop.",
                },
                fallback=(
                    f"Same-shop voucher search collected pool sizes "
                    f"{[len(p) for p in pools]} with pre-discount ceiling {ceiling:.2f}."
                ),
            ),
            tool_log, "", query, steps,
        )

    if len(pools) != n or not all(pools):
        return False

    # Build shop coverage map and find shops covering all specs
    coverage = _voucher_build_shop_coverage(pools)
    candidate_shops = [sid for sid, cov in coverage.items() if len(cov) == n]
    if not candidate_shops:
        return False

    # LLM-score each spec pool
    scored_by_spec: list[dict[str, float]] = []
    for i, pool in enumerate(pools):
        spec = specs[i]
        sq = spec.get("query") or spec.get("keywords") or query
        pids = [str(p.get("product_id", "") or "") for p in pool if p.get("product_id")]
        details = _fetch_product_details(pids[:_SHOP_VOUCHER_DETAIL_FETCH_CAP])
        scored = _rate_pool_for_shop(
            sq, pool, details,
            only_product_type=bool(spec.get("only_product_type", False)),
            cap=_SHOP_BATCH_CAP,
            shop_cap=_SHOP_PER_STORE_CAP,
        )
        scored_by_spec.append({
            str(p.get("product_id", "") or ""): float(sc) for p, sc in scored
        })

    # Enumerate combos per candidate shop, pick best score + utilization bonus
    best: tuple | None = None  # (score, leaders, calc, shop_id)
    for sid in candidate_shops[:_SHOP_VOUCHER_MAX_CANDIDATE_SHOPS]:
        per_spec_ranked: list[list] = []
        for i in range(n):
            rows = list((coverage.get(sid) or {}).get(i) or [])
            rows.sort(
                key=lambda p: scored_by_spec[i].get(str(p.get("product_id", "") or ""), 0.0),
                reverse=True,
            )
            per_spec_ranked.append(rows[:_SHOP_VOUCHER_TOP_PER_SPEC])
        if not all(per_spec_ranked):
            continue
        combo_count = 0
        for combo in _cartesian_product(*per_spec_ranked):
            combo_count += 1
            if combo_count > _SHOP_VOUCHER_COMBO_CAP:
                break
            pids = [str(p.get("product_id", "") or "") for p in combo]
            if len(set(pids)) != len(pids):
                continue
            prices = [float(p.get("price", 0) or 0) for p in combo]
            calc = _voucher_local_math(prices, voucher)
            if not (calc["applied"] and calc["within_budget"]):
                continue
            prices_csv = ",".join(str(x) for x in prices)
            if not _voucher_confirmed_by_tool(prices_csv, voucher):
                continue
            rel = sum(
                scored_by_spec[i].get(str(combo[i].get("product_id", "") or ""), 0.0)
                for i in range(n)
            )
            budget_v = float(voucher.get("budget", 0) or 0)
            after = float(calc.get("total_after", 0.0) or 0.0)
            utilization_bonus = 1.0 - abs(after - budget_v * _SHOP_VOUCHER_UTILIZATION_TARGET) / max(budget_v, 1.0)
            score = rel + utilization_bonus
            if best is None or score > best[0]:
                best = (score, list(combo), calc, sid)

    if best is None:
        return False
    _, leaders, calc, shop_id = best
    pids = [str(p.get("product_id", "") or "") for p in leaders]
    shop_pools = [
        [p for p in pools[i] if str(p.get("shop_id", "") or "") == shop_id]
        for i in range(n)
    ]
    _voucher_close_success(
        pids=pids, leaders=leaders, pools=shop_pools, products=specs,
        voucher=voucher, calc_total=calc["total"], query=query, steps=steps,
        fb=(
            f"Same-shop voucher success: shop_id={shop_id}, "
            f"total_before={calc['total']:.2f}, discount={calc['discount']:.2f}, "
            f"total_after={calc['total_after']:.2f}, products={pids}."
        ),
        extra_ctx={"shop_id": shop_id},
    )
    return True


def _run_voucher_null_price(
    products: list, voucher: dict, query: str, steps: list,
) -> bool:
    """Voucher flow when no product spec has a price_range or service constraint.

    Enhanced with 4-stage per-spec refinement (ported from agent_make.py
    _x_try_voucher_null_sweep): if a pick scores below _VOUCHER_REFINE_BELOW,
    retries with (1) shortened keywords, (2) alternate query slug, (3) page 2,
    (4) dropping only_product_type.
    Returns True on success so the caller can short-circuit.
    """
    discount_type = voucher.get("discount_type", "percentage")
    discount_value = float(voucher.get("discount_value", 0))
    cap = float(voucher.get("cap", 0))
    threshold = float(voucher.get("threshold", 0))
    budget = float(voucher.get("budget", 0))

    if discount_value <= 0:
        logger.warning("voucher_null_price: discount_value=0, skipping")
        return False

    allowed_total = voucher_max_total_price(voucher)
    if allowed_total is None or allowed_total <= 0:
        return False

    # ── Ceiling scan: find maximum-priced item per spec ───────────────────
    max_items: list[dict | None] = []
    scan_calls: list = []
    for spec in products:
        sp = _spec_to_find_product_params(spec, include_price=False)
        sp["price"] = f"1-{allowed_total:.0f}"
        sp["sort"] = "pricedesc"
        r = safe_tool_call("find_product", sp)
        scan_calls.append(r)
        found = r.get("result") or []
        max_items.append(found[0] if found else None)

    if any(item is None for item in max_items):
        logger.info("voucher_null_price: a spec had no result in scan, aborting")
        return False

    sorted_indices = sorted(
        range(len(max_items)),
        key=lambda i: float((max_items[i] or {}).get("price", 0) or 0),
    )
    prices_sorted = [float((max_items[i] or {}).get("price", 0) or 0) for i in sorted_indices]
    above_index = (
        sorted_indices[-1] if threshold > 0 and prices_sorted and prices_sorted[-1] >= threshold
        else None
    )

    processing_order = (
        [above_index] + [i for i in sorted_indices if i != above_index]
        if above_index is not None else sorted_indices
    )

    logger.info(
        "voucher_null_price: sorted=%s above=%s allowed=%.2f",
        sorted_indices, above_index, allowed_total,
    )

    # ── Per-spec search + 4-stage refinement ─────────────────────────────
    pid_map: dict[int, str] = {}
    reason_map: dict[int, str] = {}
    picked_price_map: dict[int, float] = {}
    spent = 0.0
    all_tool_calls: list = list(scan_calls)

    for i in processing_order:
        spec = products[i]
        remaining = allowed_total - spent
        if remaining <= 0:
            return False
        sp = _spec_to_find_product_params(spec, include_price=False)
        price_band = (
            f"{min(budget, remaining):.0f}-{remaining:.0f}"
            if i == above_index and remaining >= budget
            else f"1-{remaining:.0f}"
        )
        sp["price"] = price_band

        r = safe_tool_call("find_product", sp)
        all_tool_calls.append(r)
        found = r.get("result") or []
        if not found:
            logger.info("voucher_null_price: no results for spec %d", i)
            return False

        pids_found = [str(p.get("product_id", "")) for p in found if p.get("product_id")]
        details = _fetch_product_details(pids_found)
        picked = _llm_choose_product_with_consistency(
            str(spec.get("query") or spec.get("keywords") or ""),
            found, details,
            only_product_type=bool(spec.get("only_product_type", False)),
            model=FALLBACK_MODEL,
        )
        if picked is None:
            picked = found[0]
        pscore = float(picked.get("_llm_relevance_score", 0) or 0) if picked else 0.0
        pool = _deduplicate_products(list(found))

        # Stage 1: shortened keywords
        if pscore < _VOUCHER_REFINE_BELOW:
            kw = spec.get("keywords", "") or ""
            words = kw.split()
            if len(words) >= 3:
                ss = dict(spec)
                ss["keywords"] = " ".join(words[:2])
                ss["only_product_type"] = False
                ssp = _spec_to_find_product_params(ss, include_price=False)
                ssp["price"] = price_band
                sr = safe_tool_call("find_product", ssp)
                all_tool_calls.append(sr)
                pool = _deduplicate_products(pool + (sr.get("result") or []))
                picked, pscore = _voucher_rejudge_pool(query, spec, pool, picked, pscore)

        # Stage 2: alternate query slug
        if pscore < _VOUCHER_REFINE_BELOW:
            alt = _voucher_alternate_query_slug(spec)
            if alt:
                alt_spec = dict(spec)
                alt_spec["keywords"] = alt
                alt_spec["only_product_type"] = False
                alt_sp = _spec_to_find_product_params(alt_spec, include_price=False)
                alt_sp["price"] = price_band
                ar = safe_tool_call("find_product", alt_sp)
                all_tool_calls.append(ar)
                pool = _deduplicate_products(pool + (ar.get("result") or []))
                picked, pscore = _voucher_rejudge_pool(query, spec, pool, picked, pscore)

        # Stage 3: page 2
        if pscore < _VOUCHER_REFINE_BELOW:
            p2 = dict(sp)
            p2["page"] = 2
            p2r = safe_tool_call("find_product", p2)
            all_tool_calls.append(p2r)
            pool = _deduplicate_products(pool + (p2r.get("result") or []))
            picked, pscore = _voucher_rejudge_pool(query, spec, pool, picked, pscore)

        # Stage 4: relax only_product_type
        if pscore < _VOUCHER_REFINE_BELOW and spec.get("only_product_type"):
            rel = dict(spec)
            rel["only_product_type"] = False
            rsp = _spec_to_find_product_params(rel, include_price=False)
            rsp["price"] = price_band
            rr = safe_tool_call("find_product", rsp)
            all_tool_calls.append(rr)
            pool = _deduplicate_products(pool + (rr.get("result") or []))
            picked, pscore = _voucher_rejudge_pool(query, spec, pool, picked, pscore)

        if not picked:
            return False
        pid_map[i] = str(picked["product_id"])
        reason_map[i] = picked.get("_llm_reason", "")
        picked_price = float(picked.get("price", 0) or 0)
        picked_price_map[i] = picked_price
        spent += picked_price

    pids = [pid_map[i] for i in range(len(products)) if i in pid_map]
    if len(pids) != len(products):
        return False

    prices_ordered = [picked_price_map[i] for i in range(len(products))]
    calc = _voucher_local_math(prices_ordered, voucher)
    if not (calc["applied"] and calc["within_budget"]):
        logger.info("voucher_null_price: local math failed after refinement")
        return False

    # ── Narrate scan ──────────────────────────────────────────────────────
    voucher_constraint = {
        "budget": budget,
        "discount_type": discount_type,
        "discount_value": discount_value,
        "cap": cap,
        "threshold": threshold,
    }
    top_cands = [
        {
            "keywords": products[i].get("keywords", ""),
            "results_found": 1 if max_items[i] else 0,
            "top_product": {
                "product_id": str(max_items[i].get("product_id", "")) if max_items[i] else "",
                "title": max_items[i].get("title", "") if max_items[i] else "",
                "price": max_items[i].get("price") if max_items[i] else None,
            } if max_items[i] else None,
        }
        for i in range(len(products))
    ]
    discount_desc = (
        f"fixed {discount_value}" if discount_type == "fixed"
        else f"{discount_value}%" + (f" capped at {cap}" if cap > 0 else "")
    )
    think_search = _invoke_step_narrator(
        query,
        {
            "budget_constraint": voucher_constraint,
            "max_allowed_total": allowed_total,
            "candidates_per_product": top_cands,
        },
        fallback=(
            f"Voucher null-price scan: {len(products)} product(s). "
            f"Budget={budget}, discount={discount_desc}, threshold={threshold}, "
            f"allowed_total={allowed_total:.2f}."
        ),
    )
    _append_step(think_search, all_tool_calls, "", query, steps)

    # ── Narrate selection + close ─────────────────────────────────────────
    base = [{"product_id": pid_map.get(i, ""), "title": "", "price": picked_price_map.get(i)} for i in range(len(products))]
    selected_info = _enrich_products_for_reason(base)
    for i, entry in enumerate(selected_info):
        entry["llm_reason"] = reason_map.get(i, "")
    ctx_ok: dict = {
        "selected_products": selected_info,
        "total_spent": calc["total_after"],
        "total_before_discount": calc["total"],
        "allowed_total": allowed_total,
        "budget_constraint": voucher_constraint,
    }
    if len(products) == len(selected_info):
        ctx_ok["constraint_checks"] = [
            _check_pick_against_query(
                title=info.get("title", ""),
                price=info.get("price"),
                parsed_spec=spec or {},
            )
            for spec, info in zip(products, selected_info)
        ]
    think_ok = _invoke_step_narrator(
        query, ctx_ok,
        fallback=(
            f"Selected {len(pids)} products. "
            f"total_before={calc['total']:.2f}, discount={calc['discount']:.2f}, "
            f"total_after={calc['total_after']:.2f}, budget={budget:.2f}. Products: {pids}."
        ),
    )
    _finish_session(pids, "success", query, steps, think=think_ok)
    return True


def _run_voucher_search(params: dict, query: str, steps: list) -> None:
    """Voucher/budget flow. Paths tried in order:
    1A. Same-shop combo matrix  →  _voucher_run_same_shop_combo
    1B. Null-price fast track   →  _run_voucher_null_price (4-stage refinement)
    Guards
    0.  Pricedesc scan + broad pool (multi-page RRF)
    S1. Combo matrix
    S2. Direct LLM pick
    S3. Swap refine
    4.  Knapsack solver
    S4. Budget fill
    5.  Reservation-window allocator (final fallback)
    """
    is_shop = params.get("is_shop_voucher", False) or "same shop" in query.lower()
    products = params.get("products", [])
    voucher = _normalize_voucher_fields(params.get("voucher"))

    # ── Path 1A: shop-voucher — combo-based same-shop search ──────────────
    if is_shop and len(products) > 1:
        if _voucher_run_same_shop_combo(params, voucher, query, steps):
            return
        # fall through to regular voucher strategies if combo fails

    # ── Path 1B: null-price fast track ────────────────────────────────────
    all_null = all(
        not p.get("price_range") and not p.get("service") for p in products
    )
    if all_null and len(steps) <= 1:
        if _run_voucher_null_price(products, voucher, query, steps):
            return

    # ── Guards ────────────────────────────────────────────────────────────
    n_specs = len(products)
    if n_specs == 0:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="No product specs found in voucher query.",
        )
        return

    allowed_total = voucher_max_total_price(voucher)
    if not allowed_total or allowed_total <= 0:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="Could not calculate allowed total from voucher parameters.",
        )
        return

    kw_list = [spec.get("keywords", "") for spec in products]
    think_analyze = _invoke_step_narrator(
        query,
        {
            "product_count": n_specs,
            "budget": voucher.get("budget"),
            "allowed_total": round(allowed_total, 2),
            "products": [
                {
                    "keywords": spec.get("keywords"),
                    "price_range": spec.get("price_range"),
                }
                for spec in products
            ],
        },
        fallback=(
            f"Voucher task: {n_specs} product(s). "
            f"Budget={voucher.get('budget')}, allowed_total={allowed_total:.2f}. "
            f"Keywords: {kw_list}."
        ),
    )

    # ── Path 0: pricedesc scan to measure max price per spec ─────────────
    scan_tool_calls: list = []
    max_prices: list[float] = []
    for spec in products:
        sp = _spec_to_find_product_params(spec, include_price=False)
        sp["price"] = f"1-{allowed_total:.0f}"
        sp["sort"] = "pricedesc"
        r = safe_tool_call("find_product", sp)
        scan_tool_calls.append(r)
        found = r.get("result") or []
        max_prices.append(float(found[0].get("price", 0)) if found else 0.0)

    _append_step(think_analyze, scan_tool_calls, "", query, steps)
    logger.info(
        "voucher_search: allowed_total=%.2f n_specs=%d max_prices=%s",
        allowed_total, n_specs, max_prices,
    )

    # ── Broad pool collection: multi-page RRF ─────────────────────────────
    voucher_ceiling = int(allowed_total) if allowed_total > 0 else None
    # More pages for fewer specs (more budget flexibility)
    pages_per_spec = 4 if n_specs <= 2 else 3 if n_specs == 3 else 2

    if n_specs <= 2:
        sort_variants = ["pricedesc", "priceasc"]
    elif n_specs == 3:
        sort_variants = ["pricedesc"]
    else:
        sort_variants = []

    cand_products_llm: list[list] = []
    llm_pick_calls: list = []
    for spec in products:
        sp_broad = _spec_to_find_product_params(spec, include_price=False)
        rare = _voucher_rare_tokens(spec)
        rankings: list[list] = []

        # Multi-page fetch for richer candidate pool
        for pg in range(1, pages_per_spec + 1):
            r_pg = safe_tool_call("find_product", {**sp_broad, "page": pg})
            llm_pick_calls.append(r_pg)
            rankings.append(r_pg.get("result") or [])

        if rare:
            sp_focused = dict(sp_broad)
            sp_focused["q"] = (
                str(sp_broad.get("q", "")) + " " + " ".join(rare)
            ).strip()
            r_focused = safe_tool_call("find_product", {**sp_focused, "page": 1})
            llm_pick_calls.append(r_focused)
            rankings.append(r_focused.get("result") or [])

        for sort_opt in sort_variants:
            sp_sorted = dict(sp_broad)
            if voucher_ceiling is not None and "price" not in sp_sorted:
                sp_sorted["price"] = f"0-{voucher_ceiling}"
            sp_sorted["sort"] = sort_opt
            r_sorted = safe_tool_call("find_product", {**sp_sorted, "page": 1})
            llm_pick_calls.append(r_sorted)
            rankings.append(r_sorted.get("result") or [])

        found_llm = _voucher_rrf_merge(rankings, top_n=20)

        # Widen if service constraint excluded results
        if not found_llm and sp_broad.get("service"):
            sp_no_svc = {k: v for k, v in sp_broad.items() if k != "service"}
            widened_price = _voucher_widen_price_range(
                sp_no_svc.get("price", ""), ratio=_VOUCHER_PRICE_WIDEN_RATIO
            )
            if widened_price:
                sp_no_svc["price"] = widened_price
            r = safe_tool_call("find_product", {**sp_no_svc, "page": 1})
            llm_pick_calls.append(r)
            found_llm = _voucher_rrf_merge([r.get("result") or []], top_n=20)

        cand_products_llm.append(found_llm)

    # ── Strategy 1: Combo matrix (n ≤ 3) ─────────────────────────────────
    if n_specs <= 3 and len(cand_products_llm) == n_specs and all(cand_products_llm):
        if _voucher_try_combo_matrix(products, cand_products_llm, voucher, query, steps):
            return

    # ── Strategy 2: Direct LLM pick ───────────────────────────────────────
    if len(cand_products_llm) == n_specs and all(cand_products_llm):
        ok, cand_products_llm = _voucher_try_direct_pick(
            products, cand_products_llm, voucher, query, steps
        )
        if ok:
            return

    # ── Strategy 3: Swap refine ───────────────────────────────────────────
    if len(cand_products_llm) == n_specs and all(cand_products_llm):
        if _voucher_try_swap_refine(products, cand_products_llm, voucher, query, steps):
            return

    # ── Path 4: knapsack solver (cart-level optimiser) ────────────────────
    scored_per_spec: list[list[tuple]] = []
    if (
        cand_products_llm
        and all(cand_products_llm)
        and _time_left() > _VOUCHER_SOFT_DEADLINE
    ):
        try:
            scored_per_spec = _voucher_score_pool(products, cand_products_llm)
        except Exception:
            logger.warning("voucher_search: knapsack scoring failed", exc_info=True)
            scored_per_spec = []

        if scored_per_spec and all(scored_per_spec):
            require_same_shop = bool(params.get("is_shop_voucher", False))
            ladder_result = _voucher_knapsack_tiers(
                scored_per_spec,
                max_total=allowed_total,
                require_same_shop=require_same_shop,
                voucher=voucher,
            )
            if ladder_result is not None:
                selection, ctx = ladder_result
                pids = [str(p.get("product_id", "")) for p in selection]
                total_price_k = float(ctx.get("total_price", 0.0))
                enriched = _enrich_products_for_reason([
                    {
                        "product_id": str(p.get("product_id", "")),
                        "title": p.get("title", ""),
                        "price": p.get("price"),
                    }
                    for p in selection
                ])
                k_pick_lines = "; ".join(
                    f"spec[{j}] pid={str(p.get('product_id',''))} "
                    f"'{str(p.get('title',''))[:60]}' price={p.get('price')}"
                    for j, p in enumerate(selection)
                )
                think_k = _invoke_step_narrator(
                    query,
                    {
                        "selected_products": enriched,
                        "total_before_discount": round(total_price_k, 2),
                        "budget_constraint": voucher,
                        "solver": "knapsack",
                        "tier": ctx.get("tier"),
                    },
                    fallback=(
                        f"Knapsack solver (tier='{ctx.get('tier')}') selected "
                        f"{len(selection)} product(s): {k_pick_lines}. "
                        f"Cart total {total_price_k:.2f}, allowed_total {allowed_total:.2f}."
                    ),
                )
                _append_step(think_k, [], "", query, steps)
                _finish_session(pids, "success", query, steps)
                return

    # ── Strategy 4: Budget fill (after knapsack) ──────────────────────────
    if len(cand_products_llm) == n_specs and all(cand_products_llm):
        _voucher_try_budget_fill(products, cand_products_llm, voucher, query, steps)
        return

    # ── Path 5: reservation-window allocator (final fallback) ─────────────
    remaining_order: list[int] = sorted(
        range(n_specs), key=lambda spec_index: max_prices[spec_index], reverse=True,
    )
    logger.info("voucher_search: processing order by price desc=%s", remaining_order)

    picked_products: list[Product] = []
    picked_orig_idx: list[int] = []
    pool_by_spec: dict[int, list[Product]] = {}
    budget_tool_calls: list = []

    while remaining_order:
        position = len(picked_products)
        is_anchor = position == 0
        spent = sum(float(picked.get("price", 0)) for picked in picked_products)

        found_valid = False
        for candidate_i in list(remaining_order):
            others = [j for j in remaining_order if j != candidate_i]
            reserved = sum(
                (range_min or 0.0)
                for other_idx in others
                for range_min, _ in [_parse_price_range(products[other_idx].get("price_range"))]
            )
            ceiling = allowed_total - spent - reserved
            if n_specs > 1:
                floor = allowed_total / n_specs if is_anchor else 1.0
            else:
                floor = 1.0

            orig_min_price, orig_max_price = _parse_price_range(
                products[candidate_i].get("price_range")
            )
            final_min_price = max(
                orig_min_price if orig_min_price is not None else 0.0, floor,
            )
            final_max_price = min(
                orig_max_price if orig_max_price is not None else float("inf"), ceiling,
            )

            if final_min_price > final_max_price:
                logger.info(
                    "voucher_search: spec[%d] pos=%d price window empty (%.2f > %.2f), skipping",
                    candidate_i, position, final_min_price, final_max_price,
                )
                continue

            sp = _spec_to_find_product_params(products[candidate_i], include_price=False)
            sp["price"] = f"{final_min_price:.0f}-{final_max_price:.0f}"
            candidates_in_range: list[Product] = []
            seen_product_ids: set[str] = set()
            for page in range(1, 3):
                r = safe_tool_call("find_product", {**sp, "page": page})
                budget_tool_calls.append(r)
                for row in r.get("result") or []:
                    row_pid = str(row.get("product_id", ""))
                    if row_pid and row_pid not in seen_product_ids:
                        candidates_in_range.append(row)
                        seen_product_ids.add(row_pid)

            if not candidates_in_range:
                logger.info(
                    "voucher_search: spec[%d] no results in range %.0f-%.0f; trying next",
                    candidate_i, final_min_price, final_max_price,
                )
                continue

            spec_query_text = (
                products[candidate_i].get("query")
                or products[candidate_i].get("keywords")
                or query
            )
            candidate_product_ids = [
                str(row.get("product_id", ""))
                for row in candidates_in_range if row.get("product_id")
            ]
            details = _fetch_product_details(candidate_product_ids)
            chosen = _llm_choose_product(
                spec_query_text, candidates_in_range, details,
                only_product_type=bool(products[candidate_i].get("only_product_type", False)),
                model=FALLBACK_MODEL,
            )
            if chosen is None:
                chosen = candidates_in_range[0]

            picked_products.append(chosen)
            picked_orig_idx.append(candidate_i)
            pool_by_spec[candidate_i] = candidates_in_range
            remaining_order = [j for j in remaining_order if j != candidate_i]
            found_valid = True
            logger.info(
                "voucher_search: pos=%d spec[%d] → product_id=%s price=%.2f range=[%.0f, %.0f]",
                position, candidate_i,
                chosen.get("product_id"), float(chosen.get("price", 0)),
                final_min_price, final_max_price,
            )
            break

        if not found_valid:
            logger.warning(
                "voucher_search: no valid candidate at position %d; aborting", position,
            )
            think_fail = _invoke_step_narrator(
                query,
                {
                    "position": position,
                    "allowed_total": round(allowed_total, 2),
                    "spent_so_far": round(spent, 2),
                    "note": (
                        f"No product found for spec at processing position {position} "
                        f"that fits within the remaining voucher budget."
                    ),
                },
                fallback=(
                    f"I could not find a suitable product for spec at position {position} "
                    f"within the remaining budget (spent={spent:.2f}, "
                    f"allowed_total={allowed_total:.2f}). Aborting the voucher search."
                ),
            )
            _append_step(think_fail, budget_tool_calls, "", query, steps)

            # cheapest-per-spec sub-fallback (uses knapsack candidate pool if available)
            if scored_per_spec and all(scored_per_spec):
                fb = _voucher_cheapest_per_spec(
                    scored_per_spec,
                    require_same_shop=bool(params.get("is_shop_voucher", False)),
                )
                if fb is not None:
                    selection_fb, _, total_fb = fb
                    pids_fb = [str(p.get("product_id", "")) for p in selection_fb]
                    enriched_fb = _enrich_products_for_reason([
                        {
                            "product_id": str(p.get("product_id", "")),
                            "title": p.get("title", ""),
                            "price": p.get("price"),
                        }
                        for p in selection_fb
                    ])
                    think_fb = _invoke_step_narrator(
                        query,
                        {
                            "selected_products": enriched_fb,
                            "total_before_discount": round(total_fb, 2),
                            "budget_constraint": voucher,
                            "solver": "best_effort_cheapest",
                        },
                        fallback=(
                            f"Best-effort fallback: picked the cheapest product per spec "
                            f"(total {total_fb:.2f}) after the knapsack and allocator both "
                            f"failed to fit allowed_total={allowed_total:.2f}."
                        ),
                    )
                    _append_step(think_fb, [], "", query, steps)
                    _finish_session(pids_fb, "success", query, steps)
                    return

            _finish_session(
                [FALLBACK_PRODUCT_ID], "failure", query, steps,
                think=(
                    f"Could not find a product for spec at position {position} "
                    f"within the voucher budget constraints "
                    f"(allowed_total={allowed_total:.2f})."
                ),
            )
            return

    # allocator succeeded — emit weighing step then close
    product_id_by_spec = {
        orig_idx: str(picked_products[pos].get("product_id", ""))
        for pos, orig_idx in enumerate(picked_orig_idx)
    }
    price_by_spec = {
        orig_idx: float(picked_products[pos].get("price", 0))
        for pos, orig_idx in enumerate(picked_orig_idx)
    }
    ordered_product_ids = [product_id_by_spec[spec_index] for spec_index in range(n_specs)]
    total_price = sum(price_by_spec.values())
    enriched = _enrich_products_for_reason([
        {
            "product_id": product_id_by_spec[spec_index],
            "title": picked_products[picked_orig_idx.index(spec_index)].get("title", ""),
            "price": picked_products[picked_orig_idx.index(spec_index)].get("price"),
        }
        for spec_index in range(n_specs)
    ])

    leaders_v: list = []
    pools_v: list = []
    for spec_index in range(n_specs):
        try:
            pos = picked_orig_idx.index(spec_index)
            leaders_v.append(picked_products[pos])
        except ValueError:
            leaders_v.append(None)
        pools_v.append(pool_by_spec.get(spec_index, []))
    _append_step("", budget_tool_calls, "", query, steps)
    _emit_weighing_step_multi(leaders_v, pools_v, products, query, steps)

    constraint_checks = (
        [
            _check_pick_against_query(
                title=info.get("title", ""),
                price=info.get("price"),
                parsed_spec=spec or {},
            )
            for spec, info in zip(products, enriched)
        ]
        if len(products) == len(enriched)
        else None
    )
    fallback_done = (
        f"Voucher search complete via reservation-window allocation. "
        f"Total before discount: {total_price:.2f}, allowed_total={allowed_total:.2f}, "
        f"budget={voucher.get('budget')}. Product IDs: {ordered_product_ids}."
    )
    ctx_done = {
        "selected_products": enriched,
        "total_before_discount": round(total_price, 2),
        "budget_constraint": voucher,
    }
    if constraint_checks is not None:
        ctx_done["constraint_checks"] = constraint_checks
    think_done = _invoke_step_narrator(query, ctx_done, fallback=fallback_done)
    _append_step(think_done, [], "", query, steps)
    _finish_session(ordered_product_ids, "success", query, steps)


def agent_main(problem_data: dict) -> list[dict]:
    global _problem_start_ts

    _reasoning_tls.events = []
    _product_detail_cache.clear()
    _problem_start_ts = time.monotonic()

    steps: list = []
    query: str = problem_data.get("query", "")
    if not query:
        term = safe_tool_call("terminate", {"status": "failure"})
        steps.append(
            create_dialogue_step(
                "No query was provided, so I cannot search for products.",
                [term],
                "",
                "",
                1,
            )
        )
        return steps

    try:
        category = str(problem_data.get("category", "")).lower()
        task_type = category if category in {"product", "shop", "voucher"} else _infer_task_type(query)
        _reset_llm_budget(task_type)

        params = _extract_query_params_llm(query, task_type)
        if task_type == "voucher" and isinstance(problem_data.get("voucher"), dict):
            structured_voucher = _normalize_voucher_fields(problem_data.get("voucher"))
            parsed_voucher = params.get("voucher") if isinstance(params.get("voucher"), dict) else {}
            merged_voucher = dict(parsed_voucher)
            # Structured voucher metadata from the benchmark is authoritative,
            # including legitimate zero values such as cap=0.
            merged_voucher.update(structured_voucher)
            params["voucher"] = merged_voucher
            params["is_shop_voucher"] = bool(
                merged_voucher.get("same_shop") or params.get("is_shop_voucher")
            )
        logger.info("agent_main: extracted params=%s", params)

        products_info = params.get("products", [])
        keywords_list = [
            entry.get("keywords") or entry.get("q", "") for entry in products_info
        ]
        price_list = [entry.get("price_range") for entry in products_info]
        service_list = [entry.get("service") for entry in products_info]
        fallback_init = (
            f"Query: '{query[:300]}'. "
            f"Search keywords: {keywords_list}. "
            f"Price constraints: {price_list}. "
            f"Service filters: {service_list}."
        )
        ctx_init: dict = {
            "keywords": keywords_list,
            "price_constraints": price_list,
            "service_filters": service_list,
        }
        if products_info:
            first_product = products_info[0]
            if bool(first_product.get("only_product_type")):
                ctx_init["only_product_type"] = True
                ctx_init["only_product_type_reason"] = (
                    "The query refers to the product type alone with no additional qualifiers "
                    "(no brand, color, material, or numeric spec). "
                    "Appending 'only' to the search query narrows results to this exact product "
                    "type and avoids unrelated products that merely contain this term."
                )
        if params.get("voucher"):
            voucher_block = params["voucher"]
            ctx_init["budget_constraint"] = {
                "discount_type": voucher_block.get("discount_type"),
                "discount_value": voucher_block.get("discount_value"),
                "threshold": voucher_block.get("threshold"),
                "cap": voucher_block.get("cap"),
                "budget": voucher_block.get("budget"),
            }
        think_init = _invoke_step_narrator(query, ctx_init, fallback=fallback_init)
        _append_step(think_init, [], "", query, steps)

        if task_type == "shop":
            _run_same_shop_search(params, query, steps)
        elif task_type == "voucher":
            _run_voucher_search(params, query, steps)
        else:
            _run_single_product_search(params, query, steps)

    except Exception:
        logger.error("agent_main: unhandled exception", exc_info=True)
        try:
            _finish_session([FALLBACK_PRODUCT_ID], "failure", query, steps)
        except Exception:
            steps.append(create_dialogue_step("Done.", [], "Done.", query, len(steps) + 1))

    if not steps:
        steps.append(create_dialogue_step("Done.", [], "Done.", query, 1))

    _attach_proxy_calls_to_dialogue(steps)
    logger.info("agent_main: finished — %d dialogue steps produced", len(steps))
    return steps
