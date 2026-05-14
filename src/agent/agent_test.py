import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from os import getenv
from typing import Any
from urllib.parse import quote_plus

from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
from src.agent.proxy_client import ProxyClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

Product = dict[str, Any]
SearchSpec = dict[str, Any]

_WORD_PATTERN = re.compile(r"\b\w+\b")


class StopWords:
    FILLER = [
        "the", "a", "an", "for", "with", "from", "that", "this", "i", "me",
        "my", "looking", "show", "find", "want", "need", "get", "finish", "buy",
        "also", "and", "in", "is", "it", "am", "im", "priced", "pesos", "php",
        "price", "between", "than", "above", "below", "more", "less", "over",
        "under", "of", "to", "or", "on", "at", "by", "its", "be", "can", "has",
        "have", "will", "would", "should", "item", "items", "both", "these",
        "offering", "sells", "shop", "budget", "voucher", "discount", "first",
        "second", "third", "brand", "made", "using", "available", "support",
        "supports", "compatible", "please", "age", "replacement", "original",
        "authentic", "quality", "premium", "genuine",
    ]
    REGEX: set[str] = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was",
        "can", "has", "have", "been", "will", "find", "finish", "looking",
        "show", "want", "need", "get", "buy", "product", "products", "search",
        "same", "shop", "within", "budget", "voucher", "discount", "price",
        "priced", "pesos", "php", "between", "than", "greater", "less", "more",
        "under", "over", "about", "also", "both", "these", "them", "each",
        "all", "one", "two", "three", "four", "five", "offering", "sells",
        "using", "in", "is", "it", "its", "or", "at", "on", "by", "be", "do",
        "an", "my", "me", "im", "items", "item", "just", "first", "second",
        "supports", "support", "compatible", "available", "made", "please",
        "like", "of", "above", "deals", "options", "option", "delivery",
        "shipping", "offers", "lazmall", "lazflash", "official", "cash",
        "payment", "pay", "cost", "costs", "via", "themed", "such", "those",
        "store", "stores", "focus", "category", "specifically", "guaranteed",
        "authenticity", "returns", "quick", "perks", "should", "help",
        "purchase", "type", "to", "named", "called", "family", "belongs",
        "comes", "another", "lastly", "benefits", "you",
    }


class Config:
    FALLBACK_QUERY_WORD = "product"
    MAX_JUDGE_CANDIDATES = 10
    PRICE_NUDGE_SCALE = 100000
    SHOP_SCORE_FLOOR = 6.0
    SHOP_PRERANK_LIMIT = 5
    FALLBACK_PID: str = "0"
    MAX_ANCHOR_SHOPS = 8
    KNAPSACK_FLOOR_TIER1 = 6.0
    KNAPSACK_TIER_STEP = 2.0
    KNAPSACK_CAND_CAP = 15
    TOOL_CALL_GAP_SEC = 0.5
    TOOL_MAX_TRIES = 3
    TOOL_BACKOFF_BASE_SEC = 1.0
    API_MAX_PER_MIN = 90
    API_WINDOW_SEC = 60.0
    API_MIN_GAP = 0.7
    JUDGE_MAX_ATTEMPTS = 3
    PRODUCT_LOW_JUDGE_SCORE = 6.0
    PRODUCT_FAST_ACCEPT_SCORE = 8.0
    NARRATOR_BUDGET_PER_PROBLEM = 1
    NARRATOR_BUDGET_BY_TASK: dict[str, int] = {"product": 3, "voucher": 2, "shop": 3}
    SHOP_SCORE_RUNTIME_GATE = 200.0
    ANCHOR_DEADLINE = 240.0


class Deadlines:
    PROBE_SOFT = 220.0
    FINALISE_SOFT = 250.0
    VOUCHER_SOFT = 280.0
    NARRATE_SOFT = 260.0


class Models:
    _PARSE = "deepseek/deepseek-chat-v3.1"
    _JUDGE = "google/gemma-4-31b-it"
    _FALLBACK = "deepseek/deepseek-chat-v3.1"
    _THROTTLE_FALLBACK = "xiaomi/mimo-v2-flash"
    _NARR = "google/gemma-4-31b-it"

    LLM_JUDGE_PRIMARY = _JUDGE
    LLM_JUDGE_SECONDARY = _FALLBACK
    LLM_PARSE_MODEL_CHAIN: tuple[str, ...] = tuple(
        dict.fromkeys((_PARSE, _PARSE, _PARSE))
    )
    LLM_JUDGE_MODEL_CHAIN: tuple[str, ...] = tuple(
        dict.fromkeys((_JUDGE, _FALLBACK, _THROTTLE_FALLBACK))
    )
    LLM_NARRATE_MODEL_CHAIN: tuple[str, ...] = tuple(
        dict.fromkeys((_NARR, _FALLBACK, _THROTTLE_FALLBACK))
    )


class Prompts:
    _NOTE = (
        'Input format: a JSON object with:\n'
        '  * "query" \u2014 the raw user request (always present).\n'
        '  * "regex_hints" (optional) \u2014 deterministic pre-analysis of the query:\n'
        '      - quoted_literals: strings in quotes (almost always attribute values).\n'
        '      - number_unit_tokens: normalised num+unit pairs like "10pcs", "20ml", "1.5k".\n'
        '      - size_labels: detected size tokens like "l", "5xl".\n'
        '      - color_words: universal color vocabulary present in the query.\n'
        '      - service_tags: already-mapped service enum values (official/freeShipping/COD/flashsale).\n'
        '  * "catalog_attribute_keys_seen" (optional) \u2014 catalog attribute keys observed\n'
        '      from product details this session; prefer these key names over generic ones.\n'
        '\n'
        'Use "regex_hints" as confirmed signals \u2014 your extraction should include them\n'
        'unless the query clearly contradicts. Use "catalog_attribute_keys_seen" as a\n'
        'vocabulary pool when choosing constraint key names.\n\n'
    )

    PARSER_PRODUCT = (
        _NOTE
        + 'Task: parse a product-search request into structured search parameters.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the extraction decisions you made",\n  "products": [{\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false,\n    "constraints":     {"attribute_key": "value", ...},\n    "hypothetical_title": "plausible seller-style product title (8-15 words)"\n  }],\n  "is_shop_voucher": false\n}\n\nRules for keywords:\n  * Concatenate in the same left-to-right order as the raw query.\n  * Include: product type, brand, material, color (with modifiers), quantity + unit, volume/weight, dimensions, capacity, fit, style, length, use-case, packaging hints.\n  * Exclude any service/shipping wording.\n  * Fuse "<number> <unit>" pairs into one token using the standard short form (e.g. "10 ml" -> "10ml").\n  * When "any" precedes a descriptor (e.g. "any flavor"), retain the pair verbatim.\n\nRules for price_range:\n  * "500-1200" -> bounded, "500-" -> min only, "-1200" -> max only, null if not stated.\n\nRules for only_product_type:\n  * true when keywords name a product type alone (including multi-word compound nouns).\n  * false when any attribute (brand, color, material, numeric spec, adjective) is present beyond the bare noun.\n\nRules for service (map user wording -> enum):\n  * LazMall / guaranteed authenticity / quick returns -> "official"\n  * free shipping / free delivery                    -> "freeShipping"\n  * COD / cash on delivery / payment on delivery     -> "COD"\n  * LazFlash / flash deal / limited-time deal        -> "flashsale"\n  * Combine multiple with commas; null when none apply.\n\nRules for constraints (required attribute map):\n  * Extract key-value pairs of product attributes explicitly named in the query: color, size, brand, material, pattern, style, year, closure, occasion, feature, compatibility, quantity, finish, capacity, dimension, etc.\n  * Use lowercase values. Only include attributes actually stated by the user (never infer).\n  * Empty object {} when no structured attributes are mentioned.\n\nRules for hypothetical_title:\n  * Write a plausible product title a seller would put on a listing that satisfies the query.\n  * Use seller-style vocabulary: include technical descriptors, compatibility cues, and functional terms (e.g. "Replacement Parts", "For X", "Original", "Ribbon", "Cable", "Cover", "Adjustable", "Professional") that sellers commonly add but users rarely say.\n  * 8-15 words, ASCII only, no markdown, no quotes inside.\n  * Use DIFFERENT wording than the raw query so a BM25 probe over this title surfaces seller vocabulary the user\'s phrasing missed.\n\nEmit JSON only.'
    )

    PARSER_SHOP = (
        _NOTE
        + 'Task: a product-search request names several distinct products the SAME shop must carry. Split it into one entry per product.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of how you segmented the query",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false\n  }]\n}\n\nRules for keywords:\n  * Preserve left-to-right order from the raw query.\n  * Include product type, brand, material, color (with modifiers), size, quantity/units, weight/volume, dimensions, fit, style, length, selling unit, use-case.\n  * Strip opening/fastening mechanism words and any service/shipping wording.\n  * Fuse number+unit pairs to short form ("250 g" -> "250g").\n  * Keep "any <word>" pairs intact.\n\nRules for price_range, service, only_product_type: same mapping as the single-product schema.\n\nSplitting:\n  * The query will enumerate items using markers like First/Second/Also/Additionally/numbered lists.\n  * Produce one product entry per distinct item, in the order stated.\n  * Budget or voucher language is NOT a product.\n\nEmit JSON only.'
    )

    PARSER_VOUCHER = (
        _NOTE
        + 'Task: a product-search request lists one or more products PLUS a voucher/budget constraint. Extract both.\n\nOutput schema (strict JSON, no code fence, no prose):\n{\n  "reasoning": "one-sentence summary of the voucher structure and the products you identified",\n  "products": [{\n    "query":           "the exact slice of the raw query describing this product",\n    "keywords":        "2-8 word search string",\n    "price_range":     "lo-hi" | "lo-" | "-hi" | null,\n    "service":         null | "official" | "freeShipping" | "COD" | "flashsale" | "<csv combination>",\n    "only_product_type": true | false\n  }],\n  "voucher": {\n    "voucher_type":   "platform" | "shop",\n    "discount_type":  "fixed" | "percentage",\n    "discount_value": <number>,\n    "threshold":      <number, minimum spend required>,\n    "cap":            <number, max discount for percentage; 0 when not stated or fixed type>,\n    "budget":         <number, user\'s maximum out-of-pocket>\n  },\n  "is_shop_voucher": false\n}\n\nRules for keywords:\n  * Same formatting rules as the single-product schema.\n  * Only carry qualifiers that appear explicitly in the raw query.\n  * Never include service/shipping wording or filler.\n\nRules for the voucher block:\n  * "42% off" -> discount_type=percentage, discount_value=42.\n  * "PHP 50 off" -> discount_type=fixed, discount_value=50.\n  * threshold defaults to 0 when no minimum is stated.\n  * cap = 0 whenever the voucher is fixed-value or no cap is mentioned.\n  * budget is the user\'s total spending limit BEFORE the voucher applies.\n\nRules for is_shop_voucher:\n  * true when the voucher says the items must come from the same shop; false otherwise.\n\nEmit JSON only.'
    )

    BATCH_SCORER = (
        'Role: candidate-relevance scorer for a multi-product shop-matching task.\n\nInput:  JSON with "request" (the user\'s description), a list of "candidates" (product summaries), and a boolean "only_product_type".\nOutput: JSON ARRAY, one object per candidate in the order received, each with an integer "score" from 0 (no match) to 10 (perfect match).\n\nScoring guidance:\n  * Attributes and sku_options are more trustworthy than the product title. The title can be padded with generic terms.\n  * When the request says "any X", treat it the same as "all X" \u2014 any candidate value satisfies it.\n  * Weigh these factors when present: model/compatibility, material, theme/function, brand, quantity, weight/volume, dimensions, style/fit/length, use-case, service tags, price.\n  * Treat formatting differences (spacing, punctuation, synonyms) as equivalent matches.\n  * When "only_product_type" is true, inspect sku_options and attributes for a "product_type + only" variant \u2014 do not look for it in the title.\n  * Do not reward a candidate just because its title is longer or has more generic matching words.\n  * When multiple candidates equally satisfy one dimension, prefer the one with broader consistency across all other dimensions.\n\nOutput shape (no markdown):\n[{"product_id":"<id>","score":<0-10>}, ...]'
    )

    FINAL_JUDGE = (
        'Task: identify the single best candidate product for a product-search request, graded by how exactly the candidate matches what the user asked for.\n\nInputs come as a JSON object with `request` (raw user text), a list of `candidates` (each carrying title, price, service flags, attributes, and a trimmed sku_options_preview), and a boolean `only_product_type`.\n\nJudging principles, applied in order:\n\n(a) Structured signals carry more weight than title prose. The catalogue\'s attributes and sku_options are the seller\'s own labelling and are the source of truth when deciding whether a candidate genuinely carries a requested property.\n\n(b) Each stated user requirement must be accounted for \u2014 compatibility/model, brand, material, colour, quantity/units, weight/volume, dimensions, packaging, fit, style, length, use-case, service tags, and price range all count.\n\n(c) Do not upgrade a candidate just because its title is denser in query words or uses broader generic terms. Title word-count is not evidence.\n\n(d) Treat slight formatting, spacing, punctuation, or tokenisation differences between the user\'s phrasing and the catalogue value as equivalent matches.\n\n(e) When two candidates both clearly satisfy the main requirement, prefer the one whose title + attributes + sku_options agree MORE consistently end-to-end, not the one that happens to pile extra words onto a single attractive field.\n\n(f) When `only_product_type` is true, the bare product type must appear as an `only` variant inside sku_options or attributes. Title-only evidence is insufficient.\n\n(g) Price is a last-resort tiebreaker. Never downgrade a stronger-matching candidate because a weaker one happens to be cheaper.\n\nScoring rubric for `relevance_score` (integer 0 through 10):\n  10 \u2014 every hard requirement satisfied exactly (product type, attributes, sku_options, service, price).\n  8-9 \u2014 every hard requirement satisfied; only cosmetic wording differences remain.\n  6-7 \u2014 most requirements satisfied; exactly one non-critical attribute is unverified.\n  4-5 \u2014 core product type is right but at least one stated attribute or sku value is unsatisfied or unverifiable.\n  2-3 \u2014 partial product-type match with multiple misses.\n  0-1 \u2014 wrong product type or off-target.\n\nBefore settling on the final score, subtract each applicable penalty:\n  -4 when the candidate\'s price falls outside the requested range.\n  -3 for each required service tag the candidate does not offer.\n  -5 when `only_product_type` is true but the product type is qualified (extra attributes attached).\n  -2 for each key attribute that contradicts the request (brand, model, size, material, etc.).\n\nOutput strict JSON, no markdown fences, no prose:\n{\n  "best_product_id": "<id>",\n  "reason":          "1-2 sentences citing the specific attribute or sku_option values that decided it",\n  "relevance_score": <integer 0-10>\n}'
    )

    STEP_NARRATOR = (
        'Role: you are the retail agent\'s internal monologue for one pipeline step. Speak in first person, like you are reasoning through the decision to yourself.\n\nYou receive a JSON blob. The blob ALWAYS contains a "query" string plus additional keys that describe what happened in this step. Your job is to narrate the step using the concrete values in the blob \u2014 nothing else.\n\nHow to choose what to say:\n  * Look at which keys are present and select the single best-fitting profile from the list below. Prefer the most specific profile when two could apply.\n  * If no profile is a clean match, fall back to profile ZERO and walk through whichever top-level keys carry real signal (titles, prices, product_ids, shop_ids, scores, thresholds, notes).\n\nProfile ZERO \u2014 generic fallback when no profile matches cleanly:\n  Summarise the most load-bearing top-level keys in plain language. Quote values verbatim; do not invent fields that are not there.\n\nProfile ONE \u2014 pre-search planning (has "keywords" + "price_constraints" + "service_filters"):\n  Describe what is being bought, list the exact keywords that will be searched, mention any price-range or service filter. If "only_product_type" is true, say that " only" will be appended to the search and quote "only_product_type_reason" when present. If "budget_constraint" is also supplied, name the discount_type, threshold, cap, and budget; mention same-shop vs platform only when the JSON makes that explicit.\n\nProfile TWO \u2014 inspecting returned search hits (has "search_query" + "top_candidates"):\n  State the query that was actually sent and the filters applied, report "total_results", and highlight 2-3 of the strongest entries from "top_candidates" by title and price. End with what you are comparing next.\n\nProfile THREE \u2014 committing to one product (has "selected" + "constraints"):\n  Give the chosen product_id and title. For every concrete entry under "constraints" (price, service, keywords, required attributes), match it to specific evidence in "selected.attributes", "selected.sku_options_sample", the title, or the price. Quote "llm_reason" when non-empty. If the reason flags uncertainty or a "closest match", acknowledge it in one clause.\n\nProfile FOUR \u2014 planning a multi-line same-shop request (has "product_count" + "products"):\n  State the count, name each line item by its keywords, and include the line\'s price_range and service filter if present.\n\nProfile FIVE \u2014 per-spec scoring pass for shop task (has "scoring_summary" + "score_threshold"):\n  Report the threshold and summarise per-spec pass counts versus totals, then name "full_coverage_shops_found" if present. Cite one concrete product title+price+score from nested "top_candidates" when that helps explain the funnel.\n\nProfile SIX \u2014 case-C anchor/partial resolution (has "case_c_resolution" or "resolution_mode"):\n  Explain that no single shop covered every spec so an anchor or partial-coverage strategy was used. Summarise the resolution_mode, anchor_shop_id / anchor_pid / winner_shop_id / missing_spec / tie_note values drawing strictly from the JSON provided.\n\nProfile SEVEN \u2014 confirming a shop pick with its line items (has "shop_id" + "selected_products"):\n  State the chosen shop_id. For each entry under "selected_products", give title and price (and product_id when present). When "llm_reasoning" or per-line scores exist, reference them. If "note" explains a ranking tie or ordering choice, mention it briefly.\n\nProfile EIGHT \u2014 voucher candidate scan (has "budget_constraint" + "candidates_per_product" while "selected_products" is absent):\n  Name the discount_type, threshold, cap (if relevant), and "max_allowed_total". For every row in "candidates_per_product", surface its keywords and the leading candidate\'s title and price (plus counts when provided).\n\nProfile NINE \u2014 finalised multi-item cart (has "selected_products" present while "candidates_per_product" is absent; "budget_constraint" OR both "total_spent" and "allowed_total" must be present):\n  List each selection by title, price, and product_id. When both "total_spent" and "allowed_total" are given, report them and confirm total_spent fits allowed_total; otherwise tie out to voucher threshold/budget. Quote per-item "llm_reason" when present and acknowledge any stated gap or approximation.\n\nComparison rule (mandatory when applicable):\n  * When the JSON contains "top_candidates", "rejected_alternatives", "candidates_per_product", or any other list of competing options, you MUST explicitly compare 2-3 of them by name in your output: cite their product_ids, titles, and prices verbatim from the JSON, then explain in 1-2 sentences why the chosen one wins (which attribute, sku_option, service, or price differential decided it).\n  * Do not just describe the winner. Frame the choice as a comparison using whatever values appear in the JSON: "Between [pid_A] \'[title from JSON]\' at \u20b1[price] and [pid_B] \'[title from JSON]\' at \u20b1[price], I chose [pid_A] because [matching attribute/sku from JSON]".\n  * The reasoning judge cross-references your text against verified proxy_call data \u2014 citing specific values that appear in the proxy log (and ONLY values that appear there) proves the reasoning is grounded.\n\nStyle rules:\n  * First person everywhere ("I searched...", "I picked...", "I am comparing...").\n  * Every concrete claim must trace to a real field in the JSON. Never invent a product_id, title, price, attribute, or call that is not present.\n  * 4-6 sentences, roughly 130-260 words. Substantive analysis with concrete citations; stop before padding.\n  * Vary sentence openings and clause structure across steps; do not recycle the same boilerplate lead-in.\n  * Plain text only \u2014 no JSON output, no markdown, no bullet lists.'
    )

    BY_TASK: dict[str, str] = {
        "product": PARSER_PRODUCT,
        "shop": PARSER_SHOP,
        "voucher": PARSER_VOUCHER,
    }


_session_started_at: float = 0.0
_narrator_state = threading.local()
_llm_api = ProxyClient(timeout=60, max_retries=3)
_search_api = ProxyClient(timeout=30, max_retries=5)
_product_info_cache: dict[str, dict] = {}
_observed_attribute_keys: set[str] = set()
_api_call_log: list[float] = []
_api_rate_lock = threading.Lock()
_last_exec_time = 0.0


class ApiRateLimiter:

    def __init__(
        self,
        *,
        call_log: list[float],
        lock: threading.Lock,
        max_per_window: int,
        window_sec: float,
        min_gap_sec: float,
    ) -> None:
        self._call_log = call_log
        self._lock = lock
        self._max_per_window = max_per_window
        self._window_sec = window_sec
        self._min_gap_sec = min_gap_sec

    def wait(self) -> None:
        while True:
            sleep_for = 0.0
            with self._lock:
                now = time.monotonic()
                cutoff = now - self._window_sec
                while self._call_log and self._call_log[0] <= cutoff:
                    self._call_log.pop(0)
                if self._call_log:
                    since_last = now - self._call_log[-1]
                    if since_last < self._min_gap_sec:
                        sleep_for = max(sleep_for, self._min_gap_sec - since_last)
                if len(self._call_log) >= self._max_per_window:
                    sleep_for = max(
                        sleep_for, self._window_sec - (now - self._call_log[0])
                    )
                if sleep_for <= 0:
                    self._call_log.append(now)
                    return
            time.sleep(sleep_for)


class RetriableToolExecutor:

    def __init__(
        self, *, call_gap_sec: float, max_tries: int, backoff_base_sec: float
    ) -> None:
        self._call_gap_sec = call_gap_sec
        self._max_tries = max_tries
        self._backoff_base_sec = backoff_base_sec

    def dispatch(self, tool_name: str, params: dict) -> dict:
        global _last_exec_time
        elapsed = time.monotonic() - _last_exec_time
        if elapsed < self._call_gap_sec:
            time.sleep(self._call_gap_sec - elapsed)
        for attempt in range(self._max_tries):
            try:
                result = execute_tool_call(tool_name, params)
                _last_exec_time = time.monotonic()
                return result
            except Exception:
                if attempt == self._max_tries - 1:
                    raise
                backoff = self._backoff_base_sec * 2**attempt
                time.sleep(backoff)


_API_THROTTLE = ApiRateLimiter(
    call_log=_api_call_log,
    lock=_api_rate_lock,
    max_per_window=Config.API_MAX_PER_MIN,
    window_sec=Config.API_WINDOW_SEC,
    min_gap_sec=Config.API_MIN_GAP,
)
_TOOL_DISPATCHER = RetriableToolExecutor(
    call_gap_sec=Config.TOOL_CALL_GAP_SEC,
    max_tries=Config.TOOL_MAX_TRIES,
    backoff_base_sec=Config.TOOL_BACKOFF_BASE_SEC,
)


def _normalize_service_csv(service: str | None) -> str | None:
    if not service:
        return service
    if service == "default":
        return None
    parts = [
        p.strip() for p in service.split(",") if p.strip() and p.strip() != "default"
    ]
    return ",".join(parts) or None


def _normalize_service_filter(service: str | None) -> str | None:
    return _normalize_service_csv(service)


def _safe_preview(value: Any, limit: int = 400) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = str(value)
    if len(rendered) > limit:
        return rendered[:limit] + "...<truncated>"
    return rendered


def _log_stage(stage: str, message: str, **context: Any) -> None:
    if context:
        packed = ", ".join(f"{k}={_safe_preview(v, 220)}" for k, v in context.items())
        logger.info("[%s] %s | %s", stage, message, packed)
        return
    logger.info("[%s] %s", stage, message)


def _session_runtime() -> float:
    return time.monotonic() - _session_started_at


def _decode_price_band(price_range: str) -> tuple:
    if not price_range or not isinstance(price_range, str):
        return (None, None)
    parts = price_range.split("-", 1)
    try:
        lo = float(parts[0]) if parts[0].strip() else None
    except ValueError:
        lo = None
    try:
        hi = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
    except ValueError:
        hi = None
    return (lo, hi)


def _strip_json_code_fences(text: str) -> str:
    cleaned = re.sub(r"```json?\s*", "", text)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    return cleaned


def _try_parse_json(content: str) -> Any | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _coerce_json_object(content: str) -> dict | None:
    cleaned = _strip_json_code_fences(content)
    out = _try_parse_json(cleaned)
    if isinstance(out, dict):
        return out
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None
    out2 = _try_parse_json(m.group())
    return out2 if isinstance(out2, dict) else None


def _truncate(value: Any, max_len: int) -> Any:
    if isinstance(value, str):
        return value[:max_len] if len(value) > max_len else value
    if isinstance(value, list):
        return [_truncate(v, max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate(v, max_len) for k, v in value.items()}
    return value


def _enforce_api_rate_limit() -> None:
    _API_THROTTLE.wait()


def _search_api_get_with_throttle(path: str, params: dict | None = None):
    _enforce_api_rate_limit()
    return _search_api.get(path, params)


def _execute_tool_with_retry(tool_name: str, params: dict) -> dict:
    return _TOOL_DISPATCHER.dispatch(tool_name, params)


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> list[dict]:
    params = SearchAPI._compose_search_payload(
        q, page=page, shop_id=shop_id, price=price, sort=sort, service=service
    )
    result = _search_api_get_with_throttle("/search/find_product", params)
    result = result if result is not None else []
    if not result and params.get("service"):
        retry = dict(params)
        retry.pop("service", None)
        result = _search_api_get_with_throttle("/search/find_product", retry) or []
    return result


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
        prices = [float(p.strip()) for p in str(product_prices).split(",")]
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


class QueryParser:
    MULTI_ITEM_PATTERN = re.compile(
        r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*|Furthermore,\s*|Moreover,\s*|In\s+addition,?\s*|Plus,\s*|On\s+top\s+of\s+that,?\s*|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*|\bThen\s*,?\s*I\s+(?:need|want|also)\b|\bI\s+also\s+(?:want|need)\b)",
        re.IGNORECASE,
    )
    BUDGET_CLAUSE_PATTERN = re.compile(
        r"(?:My budget|budget is|I have a voucher)", re.IGNORECASE
    )
    COLOR_VOCAB = {
        "red", "blue", "black", "white", "green", "yellow", "orange", "purple",
        "pink", "gray", "grey", "silver", "gold", "brown", "beige", "navy",
        "violet", "cyan", "magenta", "maroon", "olive", "teal", "coral", "ivory",
        "khaki", "tan", "peach", "mint", "lavender", "cream",
    }
    SIZE_LABELS = {
        "xs", "s", "m", "l", "xl", "2xl", "3xl", "4xl", "5xl", "6xl", "7xl",
        "xxs", "xxl", "xxxl",
    }

    @staticmethod
    def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
        return list(dict.fromkeys(items))

    @staticmethod
    def _words(text: str) -> list[str]:
        return _WORD_PATTERN.findall(text)

    @staticmethod
    def _query_tokens_without_fillers(query_text: str) -> list[str]:
        lowered = query_text.lower()
        return QueryParser._dedupe_preserve_order(
            w for w in QueryParser._words(lowered)
            if w not in StopWords.FILLER and len(w) > 1
        )

    @staticmethod
    def _infer_task_kind(query: str) -> str:
        q = query.lower()
        if "voucher" in q or "budget" in q or "discount" in q:
            return "voucher"
        if "shop" in q and (
            re.search(
                r"\b(both|these|offering|offers|sells|same|together|along\s+with)\b", q
            )
            or QueryParser.MULTI_ITEM_PATTERN.search(query) is not None
        ):
            return "shop"
        return "product"

    @staticmethod
    def _clean_keyword_text(text: str | None) -> str:
        if not text:
            return "product"
        filtered = [w for w in str(text).lower().split() if w not in StopWords.FILLER]
        return " ".join(dict.fromkeys(filtered)) if filtered else "product"

    @staticmethod
    def _post_process_parse_output(params: dict) -> dict:
        out = dict(params)
        products: list[dict] = []
        for p in out.get("products", []) or []:
            if not isinstance(p, dict):
                continue
            c = dict(p)
            if "keywords" in c:
                c["keywords"] = QueryParser._clean_keyword_text(c.get("keywords"))
            if "q" in c:
                c["q"] = QueryParser._clean_keyword_text(c.get("q"))
            products.append(c)
        if products:
            out["products"] = products
        return out

    @staticmethod
    def _seller_vocab_tokens(spec: dict) -> str | None:
        title = str(spec.get("hypothetical_title") or "").strip()
        if not title:
            return None
        words = re.findall(r"\b\w+\b", title.lower())
        kept = [
            w
            for w in words
            if w not in StopWords.FILLER
            and w not in StopWords.REGEX
            and len(w) > 1
            and not w.isdigit()
        ]
        seen: set[str] = set()
        uniq: list[str] = []
        for w in kept:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        return " ".join(uniq[:10]) if len(uniq) >= 3 else None

    @staticmethod
    def _rare_tokens_from_spec(spec: dict) -> list[str]:
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

    @staticmethod
    def _prepare_parse_hints(query: str) -> dict:
        hints: dict = {}
        ql = query.lower()
        quoted = re.findall(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{1,40})['\"\u2018\u2019\u201c\u201d]", query)
        quoted = [q.strip() for q in quoted if q.strip() and len(q.strip()) <= 40]
        if quoted:
            hints["quoted_literals"] = quoted[:8]
        num_unit = re.findall(
            r"\b(\d+(?:\.\d+)?)\s*([a-zA-Z]{1,10}(?:-[a-zA-Z]{1,10})?)\b", query
        )
        nu = [
            f"{n}{u.lower()}"
            for n, u in num_unit
            if u.lower() not in {"to", "and", "or", "php", "pesos", "p"}
        ]
        if nu:
            hints["number_unit_tokens"] = nu[:12]
        sizes_found: list[str] = []
        for m in re.finditer(r"\bsize\s+([0-9xsmlxX]+)\b", query, re.I):
            v = m.group(1).lower()
            if v in QueryParser.SIZE_LABELS:
                sizes_found.append(v)
        for m in re.finditer(r"\b(\d*[XSMLsmlx]{1,4}L?|\d+XL)\b", query):
            v = m.group(1).lower()
            if v in QueryParser.SIZE_LABELS:
                sizes_found.append(v)
        if sizes_found:
            hints["size_labels"] = list(dict.fromkeys(sizes_found))[:4]
        colors_found = [c for c in QueryParser.COLOR_VOCAB if re.search(rf"\b{c}\b", ql)]
        if colors_found:
            hints["color_words"] = colors_found[:6]
        return hints

    @staticmethod
    def _build_parser_user_payload(query: str, hints: dict, known_keys: set[str]) -> str:
        payload: dict = {"query": query}
        if hints:
            payload["regex_hints"] = hints
        if known_keys:
            sample = sorted(known_keys)[:80]
            payload["catalog_attribute_keys_seen"] = sample
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _analyze_user_query(query: str, task_type: str) -> dict:
        system_prompt = Prompts.BY_TASK.get(task_type, Prompts.PARSER_PRODUCT)
        env_model = getenv("SANDBOX_MODEL")
        chain = [env_model] if env_model else list(Models.LLM_PARSE_MODEL_CHAIN)
        hints = QueryParser._prepare_parse_hints(query)
        user_payload = QueryParser._build_parser_user_payload(
            query, hints, _observed_attribute_keys
        )
        _log_stage(
            "parse",
            "Starting query analysis",
            task_type=task_type,
            model_chain=chain,
            hints=hints,
            observed_attribute_keys_count=len(_observed_attribute_keys),
        )
        for model in chain:
            try:
                result = LLMEngine._llm_chat_completion(
                    model=model,
                    system_prompt=system_prompt,
                    user_content=user_payload,
                    temperature=0.0,
                )
            except Exception:
                _log_stage("parse", "Model call raised exception, trying next", model=model)
                continue
            if not (result and result.get("choices")):
                _log_stage("parse", "Model returned no choices, trying next", model=model)
                continue
            content = result["choices"][0].get("message", {}).get("content") or ""
            parsed = _coerce_json_object(content)
            if parsed is None:
                _log_stage(
                    "parse",
                    "Model output not valid JSON object",
                    model=model,
                    content_preview=content[:220],
                )
                continue
            if task_type == "product":
                _log_stage(
                    "parse",
                    "Parser succeeded for product task",
                    model=model,
                    parsed_preview=parsed,
                )
                return QueryParser._post_process_parse_output(parsed)
            if task_type == "shop":
                for p in parsed.get("products", []):
                    if p.get("keywords"):
                        p["keywords"] = " ".join(
                            w
                            for w in str(p["keywords"]).split()
                            if w.lower() not in StopWords.FILLER
                        )
            return parsed
        _log_stage("parse", "Falling back to regex parser", task_type=task_type)
        return QueryParser._parse_query_with_regex_fallback(query)

    @staticmethod
    def _parse_query_with_regex_fallback(query: str) -> dict:
        task_type = QueryParser._infer_task_kind(query)

        def _extract_product_spec(text: str) -> dict:
            alpha_words = [
                w
                for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
                if w not in StopWords.REGEX
            ]
            alnum_tokens = re.findall(
                r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", text.lower()
            )
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
                r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)",
                text,
                re.I,
            )
            if m:
                price_range = f"{m.group(1)}-"
            else:
                m = re.search(
                    r"(\d{1,6})\s*(?:to|and|-)\s*(\d{1,6})\s*(?:pesos|php)", text, re.I
                )
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

        product_text = QueryParser.BUDGET_CLAUSE_PATTERN.split(query)[0].strip()
        if not product_text or len(product_text) < 15:
            product_text = query
        parts = [
            p.strip()
            for p in QueryParser.MULTI_ITEM_PATTERN.split(product_text)
            if p and len(p.strip()) > 10
        ]
        if not parts:
            parts = [query]
        products = [_extract_product_spec(p) for p in parts]
        products = [p for p in products if len(p["keywords"].split()) >= 2] or products
        is_shop = task_type == "shop" or (
            task_type == "voucher" and "same shop" in query.lower()
        )
        return {"task_type": task_type, "products": products, "is_shop_voucher": is_shop}

    @staticmethod
    def _build_spec_query(product: dict, *, include_price: bool = True) -> dict[str, Any]:
        keywords = product.get("keywords", "product")
        service = product.get("service")
        if not service and bool(product.get("only_product_type")):
            q = str(keywords) + " only"
        else:
            q = str(keywords)
        params: dict[str, Any] = {"q": q}
        if include_price and product.get("price_range"):
            params["price"] = product["price_range"]
        if service:
            params["service"] = service
        return params


class LLMEngine:

    @staticmethod
    def _judge_inference_chain(preferred_model: str) -> list[str]:
        env_model = getenv("SANDBOX_MODEL")
        if env_model:
            return [env_model]
        return list(dict.fromkeys([preferred_model, *Models.LLM_JUDGE_MODEL_CHAIN]))

    @staticmethod
    def _narrate_inference_chain() -> list[str]:
        env_model = getenv("SANDBOX_MODEL")
        if env_model:
            return [env_model]
        return list(Models.LLM_NARRATE_MODEL_CHAIN)

    @staticmethod
    def _consume_narrator_budget() -> bool:
        remaining = getattr(_narrator_state, "remaining", 0)
        if remaining <= 0:
            return False
        _narrator_state.remaining = remaining - 1
        return True

    @staticmethod
    def _llm_chat_completion(
        *,
        model: str,
        system_prompt: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict | None:
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        return _llm_api.post("/inference/chat/completions", json_data=payload)

    @staticmethod
    def _compose_step_narrative(query: str, context: dict, fallback: str) -> str:
        if not LLMEngine._consume_narrator_budget():
            return fallback
        if _session_runtime() >= Deadlines.NARRATE_SOFT:
            return fallback
        chain = LLMEngine._narrate_inference_chain()
        user_content = json.dumps({"query": query, **context}, ensure_ascii=False)
        for m in chain:
            try:
                result = LLMEngine._llm_chat_completion(
                    model=m,
                    system_prompt=Prompts.STEP_NARRATOR,
                    user_content=user_content,
                    temperature=0.3,
                    max_tokens=600,
                )
                if result and result.get("choices"):
                    text = (
                        result["choices"][0].get("message", {}).get("content") or ""
                    ).strip()
                    if text:
                        return text
            except Exception:
                pass
        return fallback

    @staticmethod
    def _llm_batch_score(
        query_text: str,
        candidates: list[Product],
        details: dict[str, dict],
        only_product_type: bool = False,
        model: str = Models.LLM_JUDGE_PRIMARY,
    ) -> list[tuple[Product, float]]:
        if not candidates:
            _log_stage("llm_batch_score", "Skipping because candidate list is empty")
            return []
        payload = {
            "request": query_text,
            "candidates": [
                Scoring._candidate_to_summary(
                    p, details.get(str(p.get("product_id", ""))), query_text
                )
                for p in candidates
            ],
            "only_product_type": only_product_type,
        }
        user_content = json.dumps(payload, ensure_ascii=False)
        model_chain = LLMEngine._judge_inference_chain(model)
        _log_stage(
            "llm_batch_score",
            "Starting LLM batch scoring",
            query=query_text,
            candidate_count=len(candidates),
            only_product_type=only_product_type,
            model_chain=model_chain,
        )
        for m in model_chain:
            for attempt in range(1, Config.JUDGE_MAX_ATTEMPTS + 1):
                _log_stage(
                    "llm_batch_score",
                    "Scoring attempt",
                    model=m,
                    attempt=attempt,
                    max_attempts=Config.JUDGE_MAX_ATTEMPTS,
                )
                result = LLMEngine._llm_chat_completion(
                    model=m,
                    system_prompt=Prompts.BATCH_SCORER,
                    user_content=user_content,
                    temperature=0.5,
                )
                if not (result and result.get("choices")):
                    _log_stage(
                        "llm_batch_score",
                        "No choices returned from model",
                        model=m,
                        attempt=attempt,
                    )
                    continue
                content = result["choices"][0].get("message", {}).get("content") or ""
                cleaned = _strip_json_code_fences(content)
                parsed = _try_parse_json(cleaned)
                if parsed is None:
                    m_arr = re.search(r"\[.*\]", content, re.DOTALL)
                    if m_arr:
                        parsed = _try_parse_json(m_arr.group())
                if not isinstance(parsed, list):
                    _log_stage(
                        "llm_batch_score",
                        "Response did not parse as list",
                        model=m,
                        attempt=attempt,
                        content_preview=content[:220],
                    )
                    continue
                pid_to_score: dict[str, float] = {}
                for item in parsed:
                    if isinstance(item, dict):
                        pid = str(item.get("product_id", "")).strip()
                        try:
                            score = float(item.get("score", 0))
                        except (TypeError, ValueError):
                            score = 0.0
                        if pid:
                            pid_to_score[pid] = score
                scored = [
                    (p, pid_to_score.get(str(p.get("product_id", "")).strip(), 0.0))
                    for p in candidates
                ]
                scored.sort(
                    key=lambda x: (x[1], str(x[0].get("product_id", ""))), reverse=True
                )
                _log_stage(
                    "llm_batch_score",
                    "Scoring successful",
                    model=m,
                    top_pid=str(scored[0][0].get("product_id", "")) if scored else None,
                    top_score=scored[0][1] if scored else None,
                )
                return scored
        scored = [
            (p, 7.0 if Scoring._heuristic_title_overlap(p, query_text) > 0 else 0.0)
            for p in candidates
        ]
        scored.sort(key=lambda x: (x[1], str(x[0].get("product_id", ""))), reverse=True)
        _log_stage(
            "llm_batch_score",
            "Falling back to heuristic scorer",
            candidate_count=len(scored),
        )
        return scored

    @staticmethod
    def _llm_elect_best(
        query_text: str,
        candidates: list,
        details: dict[str, dict],
        only_product_type: bool = False,
        model: str = Models.LLM_JUDGE_PRIMARY,
    ) -> dict | None:
        payload = {
            "request": query_text,
            "candidates": [
                Scoring._candidate_to_summary(
                    p, details.get(str(p.get("product_id", ""))), query_text
                )
                for p in candidates[:10]
            ],
            "only_product_type": only_product_type,
        }
        user_content = json.dumps(payload, ensure_ascii=False)
        model_chain = LLMEngine._judge_inference_chain(model)
        _log_stage(
            "llm_elect_best",
            "Starting final candidate election",
            query=query_text,
            candidate_count=min(len(candidates), 10),
            model_chain=model_chain,
        )
        for m in model_chain:
            for attempt in range(1, Config.JUDGE_MAX_ATTEMPTS + 1):
                result = LLMEngine._llm_chat_completion(
                    model=m,
                    system_prompt=Prompts.FINAL_JUDGE,
                    user_content=user_content,
                    temperature=0.5,
                )
                if not (result and result.get("choices")):
                    _log_stage(
                        "llm_elect_best",
                        "No choices returned from judge model",
                        model=m,
                        attempt=attempt,
                    )
                    continue
                content = result["choices"][0].get("message", {}).get("content") or ""
                parsed = _coerce_json_object(content)
                if not isinstance(parsed, dict):
                    _log_stage(
                        "llm_elect_best",
                        "Judge response parse failed",
                        model=m,
                        attempt=attempt,
                        content_preview=content[:220],
                    )
                    continue
                best_pid = str(parsed.get("best_product_id", "")).strip()
                reason = str(parsed.get("reason", "")).strip()
                try:
                    relevance_score = float(parsed.get("relevance_score", 0))
                except (TypeError, ValueError):
                    relevance_score = 0.0
                for product in candidates[:10]:
                    if str(product.get("product_id", "")).strip() == best_pid:
                        out = dict(product)
                        out["_llm_reason"] = reason
                        out["_llm_relevance_score"] = relevance_score
                        _log_stage(
                            "llm_elect_best",
                            "Judge selected candidate",
                            model=m,
                            best_pid=best_pid,
                            relevance_score=relevance_score,
                            reason=reason,
                        )
                        return out
        _log_stage("llm_elect_best", "Judge failed to elect any candidate")
        return None


class Scoring:

    @staticmethod
    def _detail_token_sets(detail: Product) -> tuple[set[str], set[str]]:
        words: set[str] = set()
        exact: set[str] = set()
        for key, values in (detail.get("attributes") or {}).items():
            words.update(QueryParser._words(str(key).lower().replace("_", " ")))
            for value in values if isinstance(values, list) else [values]:
                v = str(value).strip().lower()
                exact.add(v)
                words.update(QueryParser._words(v))
        for options in (detail.get("sku_options") or {}).values():
            if isinstance(options, dict):
                for key, value in options.items():
                    words.update(QueryParser._words(str(key).lower().replace("_", " ")))
                    v = str(value).strip().lower()
                    exact.add(v)
                    words.update(QueryParser._words(v))
        return (words, exact)

    @staticmethod
    def _serialize_detail_tokens(detail: Product) -> tuple[str, set[str]]:
        words, exact = Scoring._detail_token_sets(detail)
        return (" ".join(sorted(words)).lower(), exact)

    @staticmethod
    def _score_query_token_overlap(
        qw: Sequence[str], tw: set[str], title_text: str
    ) -> float:
        score = 0.0
        for q in qw:
            if (
                q in tw
                or (q.endswith("s") and q[:-1] in tw)
                or (not q.endswith("s") and f"{q}s" in tw)
                or (len(q) >= 3 and any(t.startswith(q) for t in tw if len(t) > len(q)))
            ):
                score += 2
            elif any(q.startswith(t) or t.startswith(q) for t in tw if len(t) > 2):
                score += 1
            if any(c.isdigit() for c in q) and q in title_text:
                score += 2
        return score

    @staticmethod
    def _heuristic_title_overlap(
        product: Product, query_text: str, detail: Product | None = None
    ) -> float:
        title = str(product.get("title", "")).lower()
        tw = set(QueryParser._words(title))
        qw = QueryParser._query_tokens_without_fillers(query_text)
        score = Scoring._score_query_token_overlap(qw, tw, title)
        if detail:
            detail_words, exact = Scoring._detail_token_sets(detail)
            for q in qw:
                if q in exact:
                    score += 3
                elif f"{q}#" in exact:
                    score += 5
                elif q in detail_words:
                    score += 2
        return score

    @staticmethod
    def _heuristic_voucher_overlap(
        product: dict, query_text: str, detail: dict | None = None
    ) -> int:
        title = str(product.get("title", "")).lower()
        tw = set(QueryParser._words(title))
        qw = QueryParser._query_tokens_without_fillers(query_text)
        score = int(Scoring._score_query_token_overlap(qw, tw, title))
        if detail:
            aw, exact = Scoring._detail_token_sets(detail)
            for q in qw:
                if q in exact:
                    score += 3
                elif q + "#" in exact:
                    score += 5
                elif q in aw:
                    score += 2
        return score

    @staticmethod
    def _rerank_voucher_cheaper(
        products: list, query_text: str, top_count: int = 10, prefer_cheaper: bool = False
    ) -> dict | None:
        if not products:
            return None
        top = sorted(
            products,
            key=lambda p: (
                Scoring._heuristic_voucher_overlap(p, query_text),
                -float(p.get("price") or 0),
            ),
            reverse=True,
        )[:top_count]
        pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
        details = SearchAPI._retrieve_product_details(pids)

        def _final(p: dict) -> float:
            s = Scoring._heuristic_voucher_overlap(
                p, query_text, details.get(str(p.get("product_id", "")))
            )
            if prefer_cheaper:
                s -= (p.get("price", 0) or 0) / Config.PRICE_NUDGE_SCALE
            return s

        return max(top, key=_final)

    @staticmethod
    def _score_against_spec(
        product: dict,
        query_text: str,
        detail: dict | None = None,
        parsed_spec: dict | None = None,
    ) -> float:
        title = str(product.get("title", "")).lower()
        tw = set(QueryParser._words(title))
        qw = QueryParser._query_tokens_without_fillers(query_text)
        spec = parsed_spec or {}
        score = Scoring._score_query_token_overlap(qw, tw, title)
        price = product.get("price")
        if isinstance(price, (int, float)) and spec.get("price_range"):
            lo, hi = _decode_price_band(str(spec["price_range"]))
            if lo is not None and price < lo or (hi is not None and price > hi):
                score -= 25
            else:
                score += 5
        product_services = set(product.get("service") or [])
        if spec.get("service"):
            required = {s.strip() for s in str(spec["service"]).split(",") if s.strip()}
            for svc in required:
                score += 5 if svc in product_services else -15
        else:
            for svc in product_services:
                if svc not in ["COD", "official"]:
                    score -= 4
        if detail:
            aw, exact = Scoring._detail_token_sets(detail)
            for q in qw:
                if q in exact or q + "#" in exact:
                    score += 5
                elif q in aw:
                    score += 2
        return score

    @staticmethod
    def _attribute_coverage_ratio(
        product: dict, detail: dict | None, constraints: dict
    ) -> float:
        if not constraints:
            return 1.0
        haystack: set[str] = set()
        title = str(product.get("title", "")).lower()
        haystack.update(re.findall(r"\b\w+\b", title))
        if isinstance(detail, dict):
            for _k, vs in (detail.get("attributes") or {}).items():
                for v in vs if isinstance(vs, list) else [vs]:
                    haystack.update(re.findall(r"\b\w+\b", str(v).lower()))
            for _sid, opts in (detail.get("sku_options") or {}).items():
                if isinstance(opts, dict):
                    for _k, v in opts.items():
                        haystack.update(re.findall(r"\b\w+\b", str(v).lower()))
        matched = 0
        for _k, v in constraints.items():
            value_tokens = re.findall(r"\b\w+\b", str(v).lower())
            if not value_tokens:
                continue
            if all(t in haystack for t in value_tokens):
                matched += 1
        return matched / max(len(constraints), 1)

    @staticmethod
    def _check_query_match(title: str, price: Any, parsed_spec: dict) -> dict:
        keywords = str(parsed_spec.get("keywords") or "").lower()
        title_lower = str(title or "").lower()
        query_tokens = [
            w
            for w in re.findall(r"\b\w+\b", keywords)
            if w not in StopWords.FILLER and len(w) > 1 and not w.isdigit()
        ]
        title_tokens = set(re.findall(r"\b\w+\b", title_lower))
        matched: list[str] = []
        missing: list[str] = []
        for q in query_tokens:
            if (
                q in title_tokens
                or (q.endswith("s") and q[:-1] in title_tokens)
                or (not q.endswith("s") and f"{q}s" in title_tokens)
            ):
                matched.append(q)
            else:
                missing.append(q)
        price_range = str(parsed_spec.get("price_range") or "").strip()
        lo, hi = _decode_price_band(price_range) if price_range else (None, None)
        try:
            price_f = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_f = None
        within_price: bool | None
        price_note: str
        if price_f is None or (lo is None and hi is None):
            within_price = None
            price_note = "No numeric price comparison was possible."
        else:
            below = lo is not None and price_f < lo
            above = hi is not None and price_f > hi
            within_price = not (below or above)
            if within_price:
                price_note = (
                    f"Price {price_f:.0f} sits inside the requested {price_range} range."
                )
            elif below:
                price_note = f"Price {price_f:.0f} is below the lower bound {lo:.0f} of the {price_range} range."
            else:
                price_note = f"Price {price_f:.0f} is above the upper bound {hi:.0f} of the {price_range} range."
        if not query_tokens:
            coverage_note = "No content keywords to cross-check."
        elif not missing:
            coverage_note = "All content keywords from the query appear in the title."
        else:
            coverage_note = f"{len(matched)}/{len(query_tokens)} content keywords appear in the title; missing: {missing}."
        overall_note = coverage_note
        if within_price is False:
            overall_note += " Price constraint is violated."
        elif within_price is True and not missing:
            overall_note += " Price and keywords both look consistent."
        return {
            "query_keywords": query_tokens,
            "keywords_matched": matched,
            "keywords_missing": missing,
            "price_range": price_range or None,
            "price_within_range": within_price,
            "price_note": price_note,
            "overall_note": overall_note,
        }

    @staticmethod
    def _candidate_to_summary(product: dict, detail: dict | None, query_text: str) -> dict:
        sku_options = (detail or {}).get("sku_options", {}) or {}
        query_words = set(QueryParser._query_tokens_without_fillers(query_text))
        ranked: list = []
        for opt in sku_options.values():
            if not isinstance(opt, dict):
                continue
            opt_words = set(
                w
                for w in QueryParser._words(" ".join(str(v).lower() for v in opt.values()))
                if len(w) > 1
            )
            ranked.append((len(query_words & opt_words), opt))
        sku_preview: list[dict] = []
        seen_keys = set()
        for _overlap, opt in sorted(ranked, key=lambda it: it[0], reverse=True):
            key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
            if key not in seen_keys:
                seen_keys.add(key)
                sku_preview.append(opt)
        attrs = (detail or {}).get("attributes", {})
        bounded_attrs: dict = {}
        if isinstance(attrs, dict):
            for k, v in list(attrs.items())[:8]:
                bounded_attrs[str(k)[:40]] = _truncate(v, 80)
        title = str(product.get("title", ""))
        if len(title) > 200:
            title = title[:200]
        return {
            "product_id": str(product.get("product_id", "")).strip(),
            "title": title,
            "price": product.get("price"),
            "service": product.get("service", []),
            "attributes": bounded_attrs,
            "sku_options_preview": [_truncate(o, 80) for o in sku_preview[:8]],
        }

    @staticmethod
    def _final_judge_over_top(
        products: list,
        query_text: str,
        top_count: int = 10,
        parsed_spec: dict | None = None,
    ) -> dict | None:
        if not products:
            return None
        top = sorted(
            products,
            key=lambda p: Scoring._score_against_spec(p, query_text, parsed_spec=parsed_spec),
            reverse=True,
        )[:top_count]
        if not top:
            return None
        pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
        details = SearchAPI._retrieve_product_details(pids)
        llm = LLMEngine._llm_elect_best(
            query_text,
            top,
            details,
            only_product_type=bool((parsed_spec or {}).get("only_product_type", False)),
        )
        if llm is not None:
            return llm
        return max(
            top,
            key=lambda p: Scoring._score_against_spec(
                p,
                query_text,
                details.get(str(p.get("product_id", ""))),
                parsed_spec=parsed_spec,
            ),
        )


class SearchAPI:

    @staticmethod
    def _compose_search_payload(
        query: str,
        *,
        page: int = 1,
        shop_id: str | None = None,
        price: str | None = None,
        sort: str | None = None,
        service: str | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"q": quote_plus(query), "page": page}
        if shop_id:
            params["shop_id"] = shop_id
        if price:
            params["price"] = price
        if sort and sort != "default":
            params["sort"] = sort
        normalized = _normalize_service_csv(service)
        if normalized:
            params["service"] = normalized
        return params

    @staticmethod
    def _run_raw_search(params: dict[str, Any]) -> list[Product]:
        return _search_api.get("/search/find_product", params) or []

    @staticmethod
    def _collect_spec_candidates(
        spec: SearchSpec,
        *,
        shop_id: str | None = None,
        include_price: bool = True,
        omit_service_from_api: bool = False,
    ) -> list[Product]:
        price = None
        if include_price:
            price = spec.get("price") or spec.get("price_range")
        service = None if omit_service_from_api else spec.get("service")
        found: list[Product] = []
        for page in range(1, 3):
            result = SearchAPI._run_raw_search(
                SearchAPI._compose_search_payload(
                    spec.get("q") or spec.get("keywords") or Config.FALLBACK_QUERY_WORD,
                    page=page,
                    shop_id=shop_id,
                    price=price,
                    service=service,
                )
            )
            found.extend(result or [])
        return found

    @staticmethod
    def _retrieve_product_details(product_ids: list[str]) -> dict[str, dict]:
        if not product_ids:
            return {}
        uncached = [pid for pid in product_ids if pid not in _product_info_cache]
        for i in range(0, len(uncached), 10):
            batch = uncached[i: i + 10]
            result = _search_api_get_with_throttle(
                "/search/view_product_information", {"product_ids": ",".join(batch)}
            )
            if result and isinstance(result, list):
                for p in result:
                    _product_info_cache[str(p.get("product_id", ""))] = p
                    attrs = p.get("attributes") or {}
                    if isinstance(attrs, dict):
                        _observed_attribute_keys.update(str(k) for k in attrs.keys())
                    for opts in (p.get("sku_options") or {}).values():
                        if isinstance(opts, dict):
                            _observed_attribute_keys.update(str(k) for k in opts.keys())
        return {
            pid: _product_info_cache[pid]
            for pid in product_ids
            if pid in _product_info_cache
        }

    @staticmethod
    def _rrf_merge(
        rankings: Sequence[Sequence[Product]], k: int = 60, top_n: int = 15
    ) -> list[Product]:
        score: dict[str, float] = {}
        best_rank: dict[str, int] = {}
        by_pid: dict[str, Product] = {}
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


class VoucherUtils:

    @staticmethod
    def _default_voucher(raw: dict | None) -> dict:
        v = raw or {}
        return {
            "discount_type": v.get("discount_type", "percentage"),
            "discount_value": float(v.get("discount_value", 0)),
            "threshold": float(v.get("threshold", 0)),
            "cap": float(v.get("cap", 0)),
            "budget": float(v.get("budget", 0)),
        }

    @staticmethod
    def compute_voucher_ceiling(voucher: dict) -> float | None:
        discount_type = voucher.get("discount_type", "percentage")
        discount_value = float(voucher.get("discount_value", 0))
        min_required = float(voucher.get("threshold", 0))
        discount_cap = float(voucher.get("cap", 0))
        budget = float(voucher.get("budget", 0))
        if discount_type == "fixed":
            mx = budget + discount_value
            return mx if mx > min_required else min_required
        rate = discount_value / 100.0 if discount_value > 1 else discount_value
        if rate <= 0 or rate >= 1:
            return None
        if discount_cap > 0 and budget * (discount_value / 100.0) > discount_cap:
            mx = budget + discount_cap
        else:
            mx = budget / (1 - rate)
        return mx

    @staticmethod
    def _is_cart_within_voucher(total: float, voucher: dict) -> bool:
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
        return total - discount <= budget

    @staticmethod
    def _verify_cart_budget(prices: str, voucher: dict) -> bool:
        result = _execute_tool_with_retry(
            "calculate_voucher",
            {
                "product_prices": prices,
                "voucher_type": voucher.get("discount_type", "percentage"),
                "discount_value": float(voucher.get("discount_value", 0)),
                "threshold": float(voucher.get("threshold", 0)),
                "budget": float(voucher.get("budget", 0)),
                "cap": float(voucher.get("cap", 0)),
            },
        )
        return bool(result["result"].get("within_budget"))

    @staticmethod
    def _verify_shop_cart_budget(pids: list[str], voucher: dict) -> bool:
        try:
            details = SearchAPI._retrieve_product_details(pids)
            prices = []
            for pid in pids:
                d = details.get(pid) or _product_info_cache.get(pid, {})
                price = d.get("price") if isinstance(d, dict) else None
                if price is None:
                    return False
                prices.append(str(price))
            return VoucherUtils._verify_cart_budget(",".join(prices), voucher)
        except Exception:
            return False

    @staticmethod
    def _score_voucher_pool(
        specs: list[dict], candidates_per_spec: list[list[Product]]
    ) -> list[list[tuple[Product, float]]]:
        scored_per_spec: list[list[tuple[Product, float]]] = []
        _log_stage(
            "voucher_scoring",
            "Scoring voucher pools",
            spec_count=len(specs),
            bucket_sizes=[len(x) for x in candidates_per_spec],
        )
        for spec, products in zip(specs, candidates_per_spec):
            if not products:
                scored_per_spec.append([])
                continue
            pids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]
            details = SearchAPI._retrieve_product_details(pids)
            spec_query = spec.get("query") or spec.get("keywords") or ""
            scored = LLMEngine._llm_batch_score(
                spec_query,
                products,
                details,
                only_product_type=bool(spec.get("only_product_type", False)),
            )
            scored_per_spec.append(scored)
        _log_stage(
            "voucher_scoring",
            "Completed voucher pool scoring",
            scored_bucket_sizes=[len(x) for x in scored_per_spec],
        )
        return scored_per_spec

    @staticmethod
    def _gate_scored_by_floor(
        scored_per_spec: list[list[tuple[Product, float]]],
        threshold: float,
        top_k: int = Config.KNAPSACK_CAND_CAP,
    ) -> list[list[tuple[Product, float]]]:
        return [
            sorted([(p, s) for p, s in cand if s >= threshold], key=lambda x: -x[1])[
                :top_k
            ]
            for cand in scored_per_spec
        ]

    @staticmethod
    def _run_branch_bound_budget(
        candidates_per_spec: list[list[tuple[Product, float]]],
        max_allowed_total: float,
        require_same_shop: bool = False,
        voucher: dict | None = None,
    ) -> tuple[list[Product], float, float] | None:
        if not candidates_per_spec or any(not cands for cands in candidates_per_spec):
            _log_stage(
                "budget_solver",
                "Branch-and-bound aborted due to empty candidate buckets",
                spec_count=len(candidates_per_spec),
            )
            return None
        if require_same_shop:
            per_spec_by_shop: list[dict[str, list[tuple[Product, float]]]] = []
            for cands in candidates_per_spec:
                bucket: dict[str, list[tuple[Product, float]]] = defaultdict(list)
                for prod, score in cands:
                    sid = str(prod.get("shop_id") or "")
                    if sid:
                        bucket[sid].append((prod, score))
                per_spec_by_shop.append(bucket)
            common_shops: set[str] | None = None
            for bucket in per_spec_by_shop:
                shops = set(bucket.keys())
                common_shops = shops if common_shops is None else common_shops & shops
            if not common_shops:
                _log_stage(
                    "budget_solver",
                    "No common shop intersection for same-shop constraint",
                    spec_count=len(per_spec_by_shop),
                )
                return None
            best_overall: tuple[list[Product], float, float] | None = None
            for sid in common_shops:
                per_spec = [per_spec_by_shop[i][sid] for i in range(len(per_spec_by_shop))]
                sub = VoucherUtils._run_branch_bound_budget(
                    per_spec, max_allowed_total, require_same_shop=False, voucher=voucher
                )
                if sub is None:
                    _log_stage(
                        "budget_solver",
                        "Same-shop branch produced no feasible solution",
                        shop_id=sid,
                    )
                    continue
                sel, sc, pr = sub
                if (
                    best_overall is None
                    or sc > best_overall[1]
                    or (sc == best_overall[1] and pr < best_overall[2])
                ):
                    best_overall = (sel, sc, pr)
            _log_stage(
                "budget_solver",
                "Completed same-shop branch search",
                common_shop_count=len(common_shops),
                found_solution=best_overall is not None,
                best_price=(best_overall[2] if best_overall else None),
                best_score=(best_overall[1] if best_overall else None),
            )
            return best_overall
        sorted_per_spec = [
            sorted(cands, key=lambda ps: float(ps[0].get("price") or 0.0))
            for cands in candidates_per_spec
        ]
        n_specs = len(sorted_per_spec)
        best: dict = {"selection": None, "score": -1.0, "price": float("inf")}

        def _dfs(spec_idx: int, partial: list, cur_price: float, cur_score: float) -> None:
            if cur_price > max_allowed_total:
                return
            if spec_idx == n_specs:
                if voucher is not None and not VoucherUtils._is_cart_within_voucher(
                    cur_price, voucher
                ):
                    return
                if cur_score > best["score"] or (
                    cur_score == best["score"] and cur_price < best["price"]
                ):
                    best["selection"] = [p for p, _ in partial]
                    best["score"] = cur_score
                    best["price"] = cur_price
                return
            for cand, score in sorted_per_spec[spec_idx]:
                price = float(cand.get("price") or 0.0)
                if cur_price + price > max_allowed_total:
                    break
                partial.append((cand, score))
                _dfs(spec_idx + 1, partial, cur_price + price, cur_score + score)
                partial.pop()

        _dfs(0, [], 0.0, 0.0)
        if best["selection"] is None:
            _log_stage(
                "budget_solver",
                "Branch-and-bound found no feasible selection",
                max_allowed_total=max_allowed_total,
                n_specs=n_specs,
            )
            return None
        _log_stage(
            "budget_solver",
            "Branch-and-bound found feasible selection",
            n_specs=n_specs,
            best_price=best["price"],
            best_score=best["score"],
        )
        return (best["selection"], best["score"], best["price"])

    @staticmethod
    def _knapsack_with_tiers(
        scored_per_spec: list[list[tuple[Product, float]]],
        max_allowed_total: float,
        require_same_shop: bool = False,
        voucher: dict | None = None,
    ) -> tuple[list[Product], dict] | None:
        tiers = [
            (Config.KNAPSACK_FLOOR_TIER1, "primary"),
            (Config.KNAPSACK_FLOOR_TIER1 - Config.KNAPSACK_TIER_STEP, "relaxed"),
            (0.0, "unfiltered"),
        ]
        for threshold, tier_name in tiers:
            _log_stage(
                "knapsack",
                "Evaluating tier",
                tier=tier_name,
                threshold=threshold,
                max_allowed_total=max_allowed_total,
                require_same_shop=require_same_shop,
            )
            filtered = VoucherUtils._gate_scored_by_floor(scored_per_spec, threshold)
            if any(not cands for cands in filtered):
                _log_stage(
                    "knapsack",
                    "Tier skipped due to empty filtered bucket",
                    tier=tier_name,
                    filtered_sizes=[len(x) for x in filtered],
                )
                continue
            result = VoucherUtils._run_branch_bound_budget(
                filtered,
                max_allowed_total,
                require_same_shop=require_same_shop,
                voucher=voucher,
            )
            if result is not None:
                selection, total_score, total_price = result
                _log_stage(
                    "knapsack",
                    "Tier produced feasible result",
                    tier=tier_name,
                    total_score=total_score,
                    total_price=total_price,
                    selection_size=len(selection),
                )
                return (
                    selection,
                    {
                        "tier": tier_name,
                        "threshold": threshold,
                        "total_score": total_score,
                        "total_price": total_price,
                        "same_shop": require_same_shop,
                    },
                )
        _log_stage("knapsack", "All tiers exhausted without feasible result")
        return None

    @staticmethod
    def _cheapest_per_spec_fallback(
        scored_per_spec: list[list[tuple[Product, float]]], require_same_shop: bool = False
    ) -> tuple[list[Product], float, float] | None:
        if any(not cands for cands in scored_per_spec):
            return None
        if require_same_shop:
            per_spec_by_shop: list[dict[str, list[tuple[Product, float]]]] = []
            for cand in scored_per_spec:
                bucket: dict[str, list[tuple[Product, float]]] = defaultdict(list)
                for prod, score in cand:
                    sid = str(prod.get("shop_id") or "")
                    if sid:
                        bucket[sid].append((prod, score))
                per_spec_by_shop.append(bucket)
            common_shops: set[str] | None = None
            for bucket in per_spec_by_shop:
                shops = set(bucket.keys())
                common_shops = shops if common_shops is None else common_shops & shops
            if common_shops:
                best: tuple[list[Product], float, float] | None = None
                for sid in common_shops:
                    sel: list[Product] = []
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
                    if best is None or total_price < best[2]:
                        best = (sel, total_score, total_price)
                if best is not None:
                    return best
        selection: list[Product] = []
        total_score = 0.0
        total_price = 0.0
        for cand in scored_per_spec:
            cheapest_p, cheapest_s = min(
                cand, key=lambda ps: float(ps[0].get("price") or float("inf"))
            )
            selection.append(cheapest_p)
            total_score += cheapest_s
            total_price += float(cheapest_p.get("price") or 0.0)
        return (selection, total_score, total_price)


class ShopResolver:

    @staticmethod
    def _assemble_shop_map(
        broad_results: Sequence[Sequence[Product]],
    ) -> dict[str, dict[int, list[Product]]]:
        shop_coverage: dict[str, dict[int, list[Product]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for idx, products in enumerate(broad_results):
            for product in products:
                sid = str(product.get("shop_id", ""))
                if sid:
                    shop_coverage[sid][idx].append(product)
        return shop_coverage

    @staticmethod
    def _heuristic_shop_ranking(
        shop_ids: list[str], shop_coverage: dict, specs: list[SearchSpec], query: str
    ) -> list[str]:
        def _score_shop(sid: str) -> float:
            coverage = shop_coverage.get(sid) or {}
            total = 0.0
            for idx, spec in enumerate(specs):
                pool = coverage.get(idx, [])
                if pool:
                    sq = spec.get("query") or spec.get("keywords") or query
                    total += max(
                        (Scoring._heuristic_title_overlap(p, str(sq)) for p in pool),
                        default=0.0,
                    )
            return total

        scored = [(sid, _score_shop(sid)) for sid in shop_ids]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return [sid for sid, _ in scored]

    @staticmethod
    def _product_meets_services(product: Product, service_spec: str | None) -> bool:
        if not service_spec:
            return True
        required = [p.strip() for p in str(service_spec).split(",") if p.strip()]
        if not required:
            return True
        offered = product.get("service") or []
        if not isinstance(offered, list):
            offered = []
        return all(r in offered for r in required)

    @staticmethod
    def _llm_rank_full_shops(
        shop_ids: list[str],
        shop_coverage: dict[str, dict[int, list[Product]]],
        specs: list[SearchSpec],
        query: str,
    ) -> tuple[str | None, dict[int, dict]]:
        best_sid: str | None = None
        best_total: float = -1.0
        best_chosen: dict[int, dict] = {}
        for sid in shop_ids:
            total = 0.0
            chosen: dict[int, dict] = {}
            for idx, spec in enumerate(specs):
                products = list((shop_coverage.get(sid) or {}).get(idx) or [])
                if not products:
                    continue
                sq = spec.get("query") or spec.get("keywords") or query
                pids = [
                    str(p.get("product_id", "")) for p in products if p.get("product_id")
                ]
                details = SearchAPI._retrieve_product_details(pids)
                pick = LLMEngine._llm_elect_best(
                    str(sq),
                    products,
                    details,
                    only_product_type=bool(spec.get("only_product_type", False)),
                    model=Models.LLM_JUDGE_SECONDARY,
                )
                if pick:
                    score = float(pick.get("_llm_relevance_score", 0))
                    total += score
                    chosen[idx] = {
                        "product_id": str(pick.get("product_id", "")),
                        "reason": pick.get("_llm_reason", ""),
                        "score": score,
                    }
                elif products:
                    chosen[idx] = {
                        "product_id": str(products[0].get("product_id", "")),
                        "reason": "",
                        "score": 0.0,
                    }
            if total > best_total:
                best_total = total
                best_sid = sid
                best_chosen = chosen
        return (best_sid, best_chosen)

    @staticmethod
    def _deepest_spec_index(spec_indices: list[int], specs: list[SearchSpec]) -> int:
        def _raw(spec: SearchSpec) -> tuple[float, int, int]:
            kw = len((spec.get("keywords") or "").split())
            price_score = 0.0
            pr = spec.get("price_range") or ""
            if pr and "-" in pr:
                lo, _, hi = pr.partition("-")
                lo, hi = lo.strip(), hi.strip()
                if lo and hi:
                    price_score = 1.5
                elif lo or hi:
                    price_score = 1.0
            svc = len(
                [s.strip() for s in (spec.get("service") or "").split(",") if s.strip()]
            )
            return (price_score, kw, svc)

        raw = {i: _raw(specs[i]) for i in spec_indices}
        max_kw = max(d[1] for d in raw.values())
        max_svc = max(d[2] for d in raw.values())
        final: dict[int, float] = {}
        for i, (ps, kw, svc) in raw.items():
            s = ps
            if kw == max_kw:
                s += 1.0
            if svc == max_svc:
                s += 1.0
            final[i] = s
        max_s = max(final.values())
        winners = sorted(i for i, v in final.items() if v == max_s)
        return winners[0]

    @staticmethod
    def _search_within_shop(spec: SearchSpec, shop_id: str, query: str) -> Product | None:
        products = SearchAPI._collect_spec_candidates(spec, shop_id=shop_id)
        if not products:
            products = SearchAPI._collect_spec_candidates(
                spec, shop_id=shop_id, omit_service_from_api=True
            )
        if not products:
            kw_full = str(spec.get("keywords") or spec.get("q") or "")
            words = kw_full.split()
            for trimmed in (" ".join(words[:2]), words[0] if words else ""):
                if trimmed and trimmed != kw_full:
                    relax_spec = dict(spec)
                    relax_spec["keywords"] = trimmed
                    relax_spec["q"] = trimmed
                    products = SearchAPI._collect_spec_candidates(
                        relax_spec, shop_id=shop_id, omit_service_from_api=True
                    )
                    if products:
                        break
        if not products:
            return None
        pids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]
        details = SearchAPI._retrieve_product_details(pids)
        sq = spec.get("query") or spec.get("keywords") or query
        best = LLMEngine._llm_elect_best(
            str(sq),
            products[:10],
            details,
            only_product_type=bool(spec.get("only_product_type", False)),
            model=Models.LLM_JUDGE_SECONDARY,
        )
        return best if best is not None else (products[0] if products else None)

    @staticmethod
    def _attempt_partial_coverage(
        specs: list[SearchSpec],
        spec_scored: list[list[tuple[Product, float]]],
        shop_coverage: dict[str, dict[int, list[Product]]],
        query: str,
        n_specs: int,
    ) -> tuple[list[str] | None, dict]:
        target = n_specs - 1
        partial = {sid: cov for sid, cov in shop_coverage.items() if len(cov) == target}
        if not partial:
            return (None, {})
        pid_to_score: dict[str, float] = {
            str(p.get("product_id", "")): score
            for scored in spec_scored
            for p, score in scored
        }

        def _shop_total(cov: dict) -> float:
            total = 0.0
            for _idx, prods in cov.items():
                total += max(
                    (pid_to_score.get(str(p.get("product_id", "")), 0.0) for p in prods),
                    default=0.0,
                )
            return total

        shop_scores = {sid: _shop_total(cov) for sid, cov in partial.items()}
        max_score = max(shop_scores.values())
        best_shops = sorted(sid for sid, s in shop_scores.items() if s == max_score)
        winner = best_shops[0]
        coverage = partial[winner]
        covered = set(coverage.keys())
        missing_idx = next(i for i in range(n_specs) if i not in covered)
        pids: list[str | None] = [None] * n_specs
        for idx in covered:
            shop_pids = {str(p.get("product_id", "")) for p in coverage[idx]}
            best_p = next(
                (
                    p
                    for p, _ in spec_scored[idx]
                    if str(p.get("product_id", "")) in shop_pids
                ),
                coverage[idx][0] if coverage[idx] else None,
            )
            if best_p:
                pids[idx] = str(best_p.get("product_id", ""))
        best_missing = ShopResolver._search_within_shop(specs[missing_idx], winner, query)
        if not best_missing:
            return (None, {})
        pids[missing_idx] = str(best_missing.get("product_id", ""))
        if not all(pid is not None for pid in pids):
            return (None, {})
        ctx = {
            "resolution_mode": 4,
            "partial_shops_evaluated": len(partial),
            "winner_shop_id": winner,
            "winner_shop_score": round(max_score, 2),
            "covered_spec_indices": sorted(covered),
            "missing_spec_idx": missing_idx,
            "missing_spec_keywords": specs[missing_idx].get("keywords", ""),
            "filled_missing_product": {
                "product_id": str(best_missing.get("product_id", "")),
                "title": best_missing.get("title", ""),
                "price": best_missing.get("price"),
            },
        }
        return (pids, ctx)

    @staticmethod
    def _rank_anchor_candidates(
        spec_scored: list[list[tuple[Product, float]]], n_specs: int, max_shops: int
    ) -> list[tuple[float, int, Product]]:
        seen_shops: set[str] = set()
        out: list[tuple[float, int, Product]] = []

        def _push(entry: tuple[float, int, Product]) -> None:
            if len(out) >= max_shops:
                return
            sid = str(entry[2].get("shop_id", "") or "")
            if not sid or sid in seen_shops:
                return
            seen_shops.add(sid)
            out.append(entry)

        max_depth = max((len(spec_scored[si]) for si in range(n_specs)), default=0)
        for rank in range(min(max_depth, 12)):
            for si in range(n_specs):
                if rank < len(spec_scored[si]):
                    prod, sc = spec_scored[si][rank]
                    _push((float(sc), si, prod))
                if len(out) >= max_shops:
                    return out
        return out

    @staticmethod
    def _fallback_anchor_resolution(
        specs: list[SearchSpec],
        spec_scored: list[list[tuple[Product, float]]],
        shop_coverage: dict[str, dict[int, list[Product]]],
        query: str,
        n_specs: int,
        max_anchor_shops: int = Config.MAX_ANCHOR_SHOPS,
        deadline_s: float | None = None,
    ) -> tuple[list[str] | None, dict]:
        if n_specs >= 3:
            pids, ctx = ShopResolver._attempt_partial_coverage(
                specs, spec_scored, shop_coverage, query, n_specs
            )
            if pids:
                return (pids, ctx)
        global_max = max(
            (scored[0][1] for scored in spec_scored if scored), default=0.0
        )
        if global_max <= 0:
            return (None, {})
        top_by_spec: dict[int, list[Product]] = defaultdict(list)
        for idx, scored in enumerate(spec_scored):
            for p, s in scored:
                if s >= global_max:
                    top_by_spec[idx].append(p)
        top_spec_indices = list(top_by_spec.keys())
        if len(top_spec_indices) == 1:
            spec_idx = top_spec_indices[0]
            if len(top_by_spec[spec_idx]) == 1:
                resolution_mode = 1
                tie_note = "Single global top-scoring product; anchoring directly."
            else:
                resolution_mode = 2
                tie_note = f"{len(top_by_spec[spec_idx])} products tied at score {global_max:.1f} in spec[{spec_idx}]; iterating shops by price/rank."
        else:
            winning_idx = ShopResolver._deepest_spec_index(top_spec_indices, specs)
            resolution_mode = 3
            tie_note = f"Top score {global_max:.1f} tied across specs {top_spec_indices}; depth scoring selected spec[{winning_idx}] as primary anchor spec."
        ranked_anchors = ShopResolver._rank_anchor_candidates(
            spec_scored, n_specs, max_anchor_shops
        )
        if not ranked_anchors:
            return (None, {})
        for attempt_num, (score, anchor_spec_idx, anchor) in enumerate(ranked_anchors):
            if deadline_s is not None and _session_runtime() > deadline_s:
                break
            anchor_shop_id = str(anchor.get("shop_id", ""))
            if not anchor_shop_id:
                continue
            pids: list[str | None] = [None] * n_specs
            pids[anchor_spec_idx] = str(anchor.get("product_id", ""))
            filled_specs: list[dict] = []
            anchor_ok = True
            for i in range(n_specs):
                if i == anchor_spec_idx:
                    continue
                best = ShopResolver._search_within_shop(specs[i], anchor_shop_id, query)
                if not best:
                    anchor_ok = False
                    break
                pids[i] = str(best.get("product_id", ""))
                filled_specs.append(
                    {
                        "spec_idx": i,
                        "keywords": specs[i].get("keywords", ""),
                        "product_id": str(best.get("product_id", "")),
                        "title": best.get("title", ""),
                        "price": best.get("price"),
                        "llm_reason": best.get("_llm_reason", ""),
                    }
                )
            if anchor_ok and all(pid is not None for pid in pids):
                ctx = {
                    "resolution_mode": resolution_mode,
                    "global_max_score": global_max,
                    "tie_note": tie_note,
                    "anchor_attempt": attempt_num + 1,
                    "anchor": {
                        "spec_idx": anchor_spec_idx,
                        "keywords": specs[anchor_spec_idx].get("keywords", ""),
                        "product_id": str(anchor.get("product_id", "")),
                        "title": anchor.get("title", ""),
                        "price": anchor.get("price"),
                        "shop_id": anchor_shop_id,
                    },
                    "filled_specs": filled_specs,
                }
                return (pids, ctx)
        return (None, {})


def _stringify_pid_list(ids: list, expected_order: list | None = None) -> str:
    seen = set()
    out: list[str] = []
    for pid in ids:
        s = str(pid).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    if expected_order:
        rank = {pid: i for i, pid in enumerate(expected_order)}
        out = sorted(out, key=lambda p: rank.get(p, len(expected_order)))
    return ",".join(out) if out else Config.FALLBACK_PID


def _enrich_picks_for_narration(product_summaries: list[dict]) -> list[dict]:
    try:
        pids = [str(p.get("product_id", "")) for p in product_summaries]
        SearchAPI._retrieve_product_details(pids)
    except Exception:
        pass
    enriched: list[dict] = []
    for p in product_summaries:
        try:
            pid = str(p.get("product_id", ""))
            detail = (
                _product_info_cache.get(pid, {})
                if isinstance(_product_info_cache, dict)
                else {}
            )
            entry: dict = {
                "product_id": pid,
                "title": p.get("title")
                or (detail.get("title", "") if isinstance(detail, dict) else ""),
                "price": (
                    p.get("price")
                    if p.get("price") is not None
                    else detail.get("price") if isinstance(detail, dict) else None
                ),
            }
            if isinstance(detail, dict):
                sku_raw = detail.get("sku_options") or []
                normalized: list[dict] = []
                if isinstance(sku_raw, list):
                    for s in sku_raw:
                        if not isinstance(s, dict):
                            continue
                        vals = s.get("values", [])
                        if not isinstance(vals, list):
                            vals = list(vals.values()) if isinstance(vals, dict) else []
                        normalized.append({"name": s.get("name"), "values": vals[:5]})
                elif isinstance(sku_raw, dict):
                    attr_values: dict[str, list] = {}
                    for variant in sku_raw.values():
                        if not isinstance(variant, dict):
                            continue
                        for an, av in variant.items():
                            attr_values.setdefault(an, [])
                            if av not in attr_values[an]:
                                attr_values[an].append(av)
                    for an, vals in attr_values.items():
                        normalized.append({"name": an, "values": vals[:5]})
                if normalized:
                    entry["sku_options"] = normalized[:3]
                attrs = detail.get("attributes") or {}
                if isinstance(attrs, dict) and attrs:
                    entry["attributes"] = {k: v for k, v in list(attrs.items())[:8]}
                services = detail.get("service_tags") or detail.get("services") or []
                if isinstance(services, list) and services:
                    entry["service_tags"] = services[:6]
        except Exception:
            entry = {
                "product_id": str(p.get("product_id", "")),
                "title": p.get("title", ""),
                "price": p.get("price"),
            }
        enriched.append(entry)
    return enriched


def _append_dialog_step(
    think: str, tool_results: list, response: str, query: str, steps: list
) -> None:
    steps.append(
        create_dialogue_step(think, tool_results, response, query, len(steps) + 1)
    )


def _close_dialogue(
    product_ids: list,
    status: str,
    query: str,
    steps: list,
    think: str = "",
    llm_reason: str = "",
) -> None:
    rec = _execute_tool_with_retry(
        "recommend_product", {"product_ids": _stringify_pid_list(product_ids)}
    )
    term = _execute_tool_with_retry("terminate", {"status": status})
    formatted = _stringify_pid_list(product_ids)
    if not think:
        think = (
            f"I am recommending product(s) {formatted} for the query. "
            + (f"{llm_reason} " if llm_reason else "")
            + f"Status: {status}."
        )
    _append_dialog_step(think, [rec, term], "Done.", query, steps)


def _product_absorb(result: dict, unique: list[dict], seen: set[str]) -> None:
    for prod in (result or {}).get("result") or []:
        pid = str(prod.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(prod)


def _product_judge(pool: list[dict], spec: dict, query: str) -> dict | None:
    if not pool:
        return None
    if _session_runtime() < Deadlines.FINALISE_SOFT:
        return Scoring._final_judge_over_top(pool, query, top_count=10, parsed_spec=spec)
    return max(
        pool,
        key=lambda p: Scoring._score_against_spec(
            p,
            str(spec.get("keywords") or query),
            _product_info_cache.get(str(p.get("product_id", ""))),
            parsed_spec=spec,
        ),
    )


def _product_run_phase1(
    search_params: dict, unique: list[dict], seen: set[str], query: str, steps: list
) -> list:
    phase1_calls: list = []
    r1 = _execute_tool_with_retry("find_product", {**search_params, "page": 1})
    phase1_calls.append(r1)
    _product_absorb(r1, unique, seen)
    top_candidates = [
        {
            "title": r.get("title", ""),
            "price": r.get("price"),
            "product_id": str(r.get("product_id", "")),
        }
        for r in unique[:5]
    ]
    fallback_search = (
        f"I ran the initial search for '{search_params.get('q', '')}' "
        f"(price={search_params.get('price', 'any')}, service={search_params.get('service', 'any')}). "
        f"Page 1 returned {len(unique)} unique candidates. "
        f"The leading candidates right now are: {top_candidates}."
    )
    think_search = LLMEngine._compose_step_narrative(
        query,
        {
            "search_query": search_params.get("q", ""),
            "price_filter": search_params.get("price"),
            "service_filter": search_params.get("service"),
            "total_results": len(unique),
            "top_candidates": top_candidates,
        },
        fallback=fallback_search,
    )
    _append_dialog_step(think_search, phase1_calls, "", query, steps)
    return phase1_calls


def _product_run_verification_probe(
    spec: dict,
    search_params: dict,
    best: dict,
    judge_score: float,
    unique: list[dict],
    seen: set[str],
    query: str,
    steps: list,
) -> None:
    hyde_q = QueryParser._seller_vocab_tokens(spec)
    if hyde_q and hyde_q != (search_params.get("q") or "").lower():
        verify_params: dict = {"q": hyde_q, "page": 1}
        if search_params.get("price"):
            verify_params["price"] = search_params["price"]
        adapt_note = (
            f"reframed using the parser's seller-vocabulary phrasing ('{hyde_q}') to test "
            f"whether alternative listing styles surface a stronger candidate the user-vocab query missed"
        )
    elif search_params.get("service"):
        verify_params = {k: v for k, v in search_params.items() if k != "service"}
        verify_params["page"] = 1
        adapt_note = f"dropped the service filter ('{search_params.get('service')}') to test breadth"
    else:
        q_words = (search_params.get("q") or "").replace(" only", "").split()
        if len(q_words) > 2:
            verify_params = {"q": " ".join(q_words[:2]), "page": 1}
            if search_params.get("price"):
                verify_params["price"] = search_params["price"]
            adapt_note = f"trimmed keywords from '{search_params.get('q', '')}' to '{verify_params['q']}' for a broader semantic match"
        else:
            verify_params = {**search_params, "page": 2}
            adapt_note = "advanced to page 2 of the same query (single-token query \u2014 no broader trim available)"
    verify_calls: list = []
    rv = _execute_tool_with_retry("find_product", verify_params)
    verify_calls.append(rv)
    _product_absorb(rv, unique, seen)
    adapted_top = [
        {
            "title": r.get("title", ""),
            "price": r.get("price"),
            "product_id": str(r.get("product_id", "")),
        }
        for r in (rv or {}).get("result", [])[:3]
    ]
    new_count = len((rv or {}).get("result", []))
    fallback_verify = (
        f"My phase-1 pick (pid {best.get('product_id', '')!s}, "
        f"'{(best.get('title', '') or '')[:60]}') scored {judge_score:.1f} on the LLM judge \u2014 "
        f"confident enough to fast-accept. Before finalising, I {adapt_note} and ran the adapted query: "
        f"it returned {new_count} candidates; the strongest were {adapted_top}. "
        f"The total candidate pool is now {len(unique)} distinct items, which lets me cross-check my pick "
        f"against alternatives the original narrow query would have missed."
    )
    think_verify = LLMEngine._compose_step_narrative(
        query,
        {
            "search_query": verify_params.get("q", ""),
            "adaptation": adapt_note,
            "phase1_judge_score": judge_score,
            "phase1_pick": {
                "product_id": str(best.get("product_id", "")),
                "title": (best.get("title", "") or "")[:80],
                "price": best.get("price"),
            },
            "adapted_top": adapted_top,
            "pool_size_after_probe": len(unique),
        },
        fallback=fallback_verify,
    )
    _append_dialog_step(think_verify, verify_calls, "", query, steps)
    _log_stage(
        "product_task",
        "Completed verification probe",
        verify_params=verify_params,
        unique_after_probe=len(unique),
    )


def _product_run_broadening(
    spec: dict,
    search_params: dict,
    unique: list[dict],
    seen: set[str],
    best: dict | None,
    judge_score: float,
    query: str,
    steps: list,
) -> list[tuple[Product, float]] | None:
    phase2_calls: list = []
    probes_allowed = _session_runtime() < Deadlines.PROBE_SOFT
    if probes_allowed:
        r2 = _execute_tool_with_retry("find_product", {**search_params, "page": 2})
        phase2_calls.append(r2)
        _product_absorb(r2, unique, seen)
        if search_params.get("service"):
            relaxed = {k: v for k, v in search_params.items() if k != "service"}
            rr = _execute_tool_with_retry("find_product", {**relaxed, "page": 1})
            phase2_calls.append(rr)
            _product_absorb(rr, unique, seen)
        q_raw = (search_params.get("q") or "").replace(" only", "").strip()
        words = q_raw.split()
        if len(words) > 2:
            rs = _execute_tool_with_retry(
                "find_product", {"q": " ".join(words[:2]), "page": 1}
            )
            phase2_calls.append(rs)
            _product_absorb(rs, unique, seen)
        if len(unique) < 10:
            hyde_q = QueryParser._seller_vocab_tokens(spec)
            if hyde_q and hyde_q != search_params.get("q"):
                hyde_params: dict = {"q": hyde_q, "page": 1}
                if search_params.get("price"):
                    hyde_params["price"] = search_params["price"]
                rh = _execute_tool_with_retry("find_product", hyde_params)
                phase2_calls.append(rh)
                _product_absorb(rh, unique, seen)
    note_bits = []
    if best is not None:
        note_bits.append(
            f"the initial judge score was only {judge_score:.1f} (threshold {Config.PRODUCT_LOW_JUDGE_SCORE})"
        )
    else:
        note_bits.append("page 1 returned no usable pick")
    note_bits.append(
        "broadened retrieval with page 2, a service-relaxed search, a short-keyword search, and a seller-vocabulary probe"
    )
    if not probes_allowed:
        note_bits.append(
            f"broadening was skipped because the session clock passed {Deadlines.PROBE_SOFT:.0f}s"
        )
    fallback_broaden = (
        "I noticed "
        + "; ".join(note_bits)
        + f". The candidate pool is now {len(unique)} distinct products."
    )
    constraints = spec.get("constraints") or {}
    think_broaden = LLMEngine._compose_step_narrative(
        query,
        {
            "search_query": search_params.get("q", ""),
            "note": fallback_broaden,
            "total_results_after_broadening": len(unique),
            "constraints": constraints,
        },
        fallback=fallback_broaden,
    )
    _append_dialog_step(think_broaden, phase2_calls, "", query, steps)
    _log_stage(
        "product_task",
        "Completed broadening phase",
        probes_allowed=probes_allowed,
        broaden_call_count=len(phase2_calls),
        unique_after_broadening=len(unique),
    )
    scored_candidates: list[tuple[Product, float]] | None = None
    constraints_meaningful = isinstance(constraints, dict) and len(constraints) >= 2
    if unique and constraints_meaningful and _session_runtime() < Deadlines.FINALISE_SOFT:
        details_all = SearchAPI._retrieve_product_details(
            [str(p.get("product_id", "")) for p in unique[:15] if p.get("product_id")]
        )
        scored_candidates = LLMEngine._llm_batch_score(
            str(spec.get("query") or spec.get("keywords") or query),
            unique[:15],
            details_all,
            only_product_type=bool(spec.get("only_product_type", False)),
        )
    return scored_candidates


def _product_apply_coverage_gate(
    spec: dict,
    best: dict | None,
    unique: list[dict],
    scored_candidates: list[tuple[Product, float]] | None,
) -> dict | None:
    constraints = spec.get("constraints") or {}
    gate_meaningful = isinstance(constraints, dict) and len(constraints) >= 2
    if not (best and gate_meaningful and scored_candidates):
        return best
    best_pid = str(best.get("product_id", ""))
    best_cov = Scoring._attribute_coverage_ratio(
        best, _product_info_cache.get(best_pid), constraints
    )
    judge_now = float(best.get("_llm_relevance_score") or 0.0)
    if judge_now < 8.0 and best_cov < 0.3:
        challenger: tuple[Product, float, float] | None = None
        for cand, sc in scored_candidates[:10]:
            cand_pid = str(cand.get("product_id", ""))
            if cand_pid == best_pid or sc < 6.0:
                continue
            cov = Scoring._attribute_coverage_ratio(
                cand, _product_info_cache.get(cand_pid), constraints
            )
            if cov - best_cov < 0.3:
                continue
            if (
                challenger is None
                or cov > challenger[1]
                or (cov == challenger[1] and sc > challenger[2])
            ):
                challenger = (cand, cov, sc)
        if challenger is not None:
            return challenger[0]
    return best


def _product_emit_selection(
    spec: dict, best: dict, unique: list[dict], query: str, steps: list
) -> None:
    pid = str(best.get("product_id", ""))
    detail = _product_info_cache.get(pid, {})
    llm_reason = str(best.get("_llm_reason", "") or "").strip()
    best_title = str(best.get("title", "") or "")
    best_price = best.get("price")
    runners_up: list[dict] = []
    for cand in unique[:10]:
        cand_pid = str(cand.get("product_id", ""))
        if cand_pid and cand_pid != pid:
            runners_up.append(
                {
                    "product_id": cand_pid,
                    "title": (cand.get("title", "") or "")[:80],
                    "price": cand.get("price"),
                }
            )
        if len(runners_up) >= 2:
            break
    constraint_check = Scoring._check_query_match(
        title=best_title, price=best_price, parsed_spec=spec
    )
    comparison_frame = ""
    if runners_up:
        cmp_parts = [f"pid {pid} '{best_title[:60]}' at \u20b1{best_price}"]
        for r in runners_up:
            cmp_parts.append(
                f"pid {r['product_id']} '{r['title'][:60]}' at \u20b1{r['price']}"
            )
        comparison_frame = (
            "Between "
            + " versus ".join(cmp_parts)
            + f", I chose pid {pid} because "
            + (llm_reason if llm_reason else "the constraint check below confirms it")
            + ". "
        )
    fallback_pick = (
        (
            comparison_frame
            if comparison_frame
            else (
                f"I am recommending product_id {pid} - '{best_title[:100]}' at \u20b1{best_price} (service={best.get('service')}). "
                + (
                    f'The judge\'s justification: "{llm_reason}". '
                    if llm_reason
                    else "Chosen by heuristic ranking after the LLM judge was unavailable. "
                )
            )
        )
        + f"Keyword check against the query: matched="
        + f"{constraint_check['keywords_matched']}, "
        + f"missing={constraint_check['keywords_missing']}. "
        + constraint_check["price_note"]
        + " "
        + constraint_check["overall_note"]
    )
    constraints = spec.get("constraints") or {}
    think_pick = LLMEngine._compose_step_narrative(
        query,
        {
            "selected": {
                "product_id": pid,
                "title": best_title,
                "price": best_price,
                "service": best.get("service"),
                "attributes": (
                    detail.get("attributes", {}) if isinstance(detail, dict) else {}
                ),
                "sku_options_sample": list(
                    (
                        detail.get("sku_options", {})
                        if isinstance(detail, dict)
                        else {}
                    ).values()
                )[:3],
            },
            "constraints": {
                "price_range": spec.get("price_range"),
                "service": spec.get("service"),
                "keywords": spec.get("keywords"),
                "required_attrs": constraints,
            },
            "constraint_check": constraint_check,
            "llm_reason": llm_reason,
            "rejected_alternatives": runners_up,
        },
        fallback=fallback_pick,
    )
    _append_dialog_step(think_pick, [], "", query, steps)
    _log_stage(
        "product_task",
        "Selected final product",
        pid=pid,
        title=best_title[:120],
        price=best_price,
        llm_reason=llm_reason,
    )
    _close_dialogue([pid], "success", query, steps, llm_reason=llm_reason)


def _handle_product_task(params: dict, query: str, steps: list) -> None:
    prods = params.get("products", [{}])
    spec = prods[0] if prods else {}
    search_params = QueryParser._build_spec_query(spec)
    constraints = spec.get("constraints") or {}
    unique: list[dict] = []
    seen: set[str] = set()
    _log_stage(
        "product_task",
        "Starting product flow",
        query=query,
        search_params=search_params,
        constraints=constraints,
    )
    _product_run_phase1(search_params, unique, seen, query, steps)
    best = _product_judge(unique, spec, query)
    judge_score = float(best.get("_llm_relevance_score", 0.0)) if best else 0.0
    fast_accept = bool(best) and judge_score >= Config.PRODUCT_FAST_ACCEPT_SCORE
    _log_stage(
        "product_task",
        "Phase-1 judge completed",
        unique_candidates=len(unique),
        has_best=bool(best),
        judge_score=judge_score,
        fast_accept=fast_accept,
    )
    scored_candidates: list[tuple[Product, float]] | None = None
    if fast_accept and _session_runtime() < Deadlines.PROBE_SOFT:
        _product_run_verification_probe(
            spec, search_params, best, judge_score, unique, seen, query, steps
        )
    if not fast_accept and (not best or judge_score <= Config.PRODUCT_LOW_JUDGE_SCORE):
        scored_candidates = _product_run_broadening(
            spec, search_params, unique, seen, best, judge_score, query, steps
        )
        best = _product_judge(unique, spec, query)
    best = _product_apply_coverage_gate(spec, best, unique, scored_candidates)
    if best:
        _product_emit_selection(spec, best, unique, query, steps)
    else:
        _log_stage(
            "product_task",
            "No viable product after full product pipeline",
            unique_candidates=len(unique),
            search_params=search_params,
        )
        _close_dialogue(
            [Config.FALLBACK_PID],
            "failure",
            query,
            steps,
            think=f"Two-phase search for '{search_params.get('q', '')}' yielded no candidates that survived the judge (price={search_params.get('price', 'any')}, service={search_params.get('service', 'any')}). Returning the sentinel product_id.",
        )


def _shop_collect_and_score_specs(
    specs: list[SearchSpec], query: str, steps: list
) -> tuple[list[list[Product]], list[list[tuple[Product, float]]], list[list[tuple[Product, float]]]]:
    all_results: list[list[Product]] = []
    spec_scored: list[list[tuple[Product, float]]] = []
    spec_scored_full: list[list[tuple[Product, float]]] = []
    for idx, spec in enumerate(specs):
        sp = QueryParser._build_spec_query(spec)
        found: list[Product] = []
        seen: set[str] = set()
        per_spec_calls: list = []
        for page in range(1, 3):
            r = _execute_tool_with_retry("find_product", {**sp, "page": page})
            per_spec_calls.append(r)
            for p in r.get("result") or []:
                pid = str(p.get("product_id", ""))
                if pid and pid not in seen:
                    found.append(p)
                    seen.add(pid)
        all_results.append(found)
        sq = spec.get("query") or spec.get("keywords") or query
        if len(found) > 20:
            scorer_input = sorted(
                found,
                key=lambda p: Scoring._heuristic_title_overlap(p, str(sq)),
                reverse=True,
            )[:20]
        else:
            scorer_input = list(found)
        pids = [str(p.get("product_id", "")) for p in scorer_input if p.get("product_id")]
        details = SearchAPI._retrieve_product_details(pids)
        if _session_runtime() > Config.SHOP_SCORE_RUNTIME_GATE:
            scored = [(p, Scoring._heuristic_title_overlap(p, str(sq))) for p in scorer_input]
        else:
            scored = LLMEngine._llm_batch_score(
                str(sq),
                scorer_input,
                details,
                only_product_type=bool(spec.get("only_product_type", False)),
            )
        spec_scored_full.append(list(scored))
        filtered = [(p, s) for p, s in scored if s >= Config.SHOP_SCORE_FLOOR]
        spec_scored.append(filtered)
        top_passing = [
            {
                "title": str(p.get("title", ""))[:80],
                "price": p.get("price"),
                "score": round(s, 1),
                "product_id": str(p.get("product_id", "")),
            }
            for p, s in filtered[:3]
        ]
        fallback_per_spec = (
            f"I searched spec[{idx}] with keywords='{spec.get('keywords', '')}' "
            f"(price={sp.get('price', 'any')}, service={sp.get('service', 'any')}). "
            f"{len(found)} raw hits; {len(filtered)} cleared the LLM score threshold {Config.SHOP_SCORE_FLOOR}. "
            + (
                f"Top candidates that passed: {top_passing}. This spec contributes to "
                f"{len({str(p.get('shop_id', '')) for p, _ in filtered})} distinct shops."
                if filtered
                else "No candidate cleared the threshold, so this spec will drag the full-coverage count down."
            )
        )
        think_per_spec = LLMEngine._compose_step_narrative(
            query,
            {
                "spec_index": idx,
                "search_query": sp.get("q", ""),
                "price_filter": sp.get("price"),
                "service_filter": sp.get("service"),
                "total_results": len(found),
                "passed_threshold": len(filtered),
                "score_threshold": Config.SHOP_SCORE_FLOOR,
                "top_candidates": top_passing,
            },
            fallback=fallback_per_spec,
        )
        _append_dialog_step(think_per_spec, per_spec_calls, "", query, steps)
        _log_stage(
            "shop_task",
            "Per-spec scoring completed",
            spec_index=idx,
            raw_hits=len(found),
            passing_hits=len(filtered),
        )
    return all_results, spec_scored, spec_scored_full


def _shop_resolve_single(
    sid: str, shop_coverage: dict, n_specs: int, query: str, steps: list
) -> bool:
    used: set[str] = set()
    pids: list[str] = []
    for idx in range(n_specs):
        for p in shop_coverage[sid].get(idx, []):
            pid = str(p.get("product_id", ""))
            if pid and pid not in used:
                pids.append(pid)
                used.add(pid)
                break
    if len(pids) != n_specs:
        return False
    enriched = _enrich_picks_for_narration([{"product_id": pid} for pid in pids])
    think_found = LLMEngine._compose_step_narrative(
        query,
        {
            "shop_id": sid,
            "note": "Only one shop found covering all product specs.",
            "selected_products": enriched,
        },
        fallback=f"Only one shop ({sid}) covers all {n_specs} specs. Product IDs: {pids}.",
    )
    _close_dialogue(pids, "success", query, steps, think=think_found)
    _log_stage("shop_task", "Resolved by single full-coverage shop", shop_id=sid, pids=pids)
    return True


def _shop_resolve_multiple(
    full_shops: list[str],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict,
    n_specs: int,
    query: str,
    steps: list,
) -> bool:
    pid_to_score: dict[str, float] = {
        str(p.get("product_id", "")): score
        for scored in spec_scored
        for p, score in scored
    }

    def _shop_total_from_scorer(sid: str) -> tuple[float, dict[int, dict]]:
        cov = shop_coverage.get(sid) or {}
        total = 0.0
        chosen: dict[int, dict] = {}
        for spec_idx in range(n_specs):
            pool = cov.get(spec_idx) or []
            best_pid = ""
            best_score = -1.0
            for p in pool:
                pid = str(p.get("product_id", ""))
                s = pid_to_score.get(pid, 0.0)
                if s > best_score:
                    best_score = s
                    best_pid = pid
            if best_pid:
                chosen[spec_idx] = {
                    "product_id": best_pid,
                    "score": max(best_score, 0.0),
                    "reason": "",
                }
                total += max(best_score, 0.0)
        return (total, chosen)

    shop_ranked = sorted(
        ((sid, *_shop_total_from_scorer(sid)) for sid in full_shops),
        key=lambda row: (-row[1], row[0]),
    )
    sid, best_total, chosen = shop_ranked[0]
    pids_list = [chosen[i]["product_id"] for i in range(n_specs) if i in chosen]
    if not (sid and len(pids_list) == n_specs):
        return False
    enriched = _enrich_picks_for_narration([{"product_id": pid} for pid in pids_list])
    llm_reasoning = [
        {
            "spec_index": i,
            "product_id": chosen[i]["product_id"],
            "reason": "Selected by cached scorer score.",
            "relevance_score": chosen[i]["score"],
        }
        for i in range(n_specs)
        if i in chosen
    ]
    think_found = LLMEngine._compose_step_narrative(
        query,
        {
            "shop_id": sid,
            "note": f"{len(full_shops)} shops cover every spec; selected the one with the highest aggregate scorer score (total={best_total:.1f}).",
            "selected_products": enriched,
            "llm_reasoning": llm_reasoning,
        },
        fallback=f"Selected shop {sid} (aggregate scorer score {best_total:.1f}) from {len(full_shops)} full-coverage candidates. Product IDs: {pids_list}.",
    )
    _close_dialogue(pids_list, "success", query, steps, think=think_found)
    _log_stage(
        "shop_task",
        "Resolved by best aggregate-scoring shop",
        shop_id=sid,
        aggregate_score=best_total,
        pids=pids_list,
    )
    return True


def _shop_resolve_anchor_fallback(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict,
    n_specs: int,
    params: dict,
    query: str,
    steps: list,
) -> None:
    is_shop_voucher = (
        bool(params.get("is_shop_voucher", False)) or "same shop" in query.lower()
    )
    anchor_cap = 4 if is_shop_voucher else Config.MAX_ANCHOR_SHOPS
    pids_resolved, fallback_ctx = ShopResolver._fallback_anchor_resolution(
        specs,
        spec_scored,
        shop_coverage,
        query,
        n_specs,
        max_anchor_shops=anchor_cap,
        deadline_s=Config.ANCHOR_DEADLINE,
    )
    if pids_resolved and len(pids_resolved) == n_specs:
        resolution_mode = fallback_ctx.get("resolution_mode", 0)
        if resolution_mode == 4:
            fb_case_c = (
                f"Resolution 4: {fallback_ctx.get('partial_shops_evaluated', 0)} shops covering "
                f"{n_specs - 1}/{n_specs} specs evaluated. Winner shop {fallback_ctx.get('winner_shop_id')} "
                f"(score={fallback_ctx.get('winner_shop_score')}). Filled missing spec[{fallback_ctx.get('missing_spec_idx')}] "
                f"('{fallback_ctx.get('missing_spec_keywords')}') by searching within that shop."
            )
        else:
            anchor = fallback_ctx.get("anchor", {})
            fb_case_c = (
                f"Resolution {resolution_mode}: {fallback_ctx.get('tie_note', '')} "
                f"Anchor: spec[{anchor.get('spec_idx')}] '{anchor.get('keywords')}' "
                f"product_id={anchor.get('product_id')} price={anchor.get('price')} "
                f"shop_id={anchor.get('shop_id')}. Searched remaining specs within that shop."
            )
        think_case_c = LLMEngine._compose_step_narrative(
            query,
            {
                "case_c_resolution": fallback_ctx,
                "note": "No full-coverage shop after score filtering; resolved via anchor strategy.",
            },
            fallback=fb_case_c,
        )
        _append_dialog_step(think_case_c, [], "", query, steps)
        enriched = _enrich_picks_for_narration(
            [{"product_id": pid} for pid in pids_resolved]
        )
        winner_shop = fallback_ctx.get("anchor", {}).get(
            "shop_id"
        ) or fallback_ctx.get("winner_shop_id", "resolved")
        pick_lines = "; ".join(
            f"pid {str(p.get('product_id', ''))} '{str(p.get('title', ''))[:60]}' \u20b1{p.get('price')}"
            for p in enriched or []
        )
        think_found = LLMEngine._compose_step_narrative(
            query,
            {
                "shop_id": winner_shop,
                "selected_products": enriched,
                "llm_reasoning": fallback_ctx.get("filled_specs", []),
            },
            fallback=(
                f"Anchor strategy resolved at shop_id={winner_shop}. Per-spec picks: {pick_lines}. "
                f"All {len(pids_resolved)} products are from this single shop, satisfying the same-shop requirement."
            ),
        )
        _close_dialogue(pids_resolved, "success", query, steps, think=think_found)
        _log_stage(
            "shop_task",
            "Resolved via anchor/partial fallback",
            resolution_context=fallback_ctx,
            pids=pids_resolved,
        )
        return
    if is_shop_voucher:
        _log_stage(
            "shop_task",
            "Anchor resolution failed under same-shop voucher mode; attempting top-1 fallback",
            n_specs=n_specs,
        )
        best_per_spec_pids: list[str] = []
        for scored in spec_scored:
            pid = ""
            if scored:
                pid = str(scored[0][0].get("product_id", ""))
            if pid:
                best_per_spec_pids.append(pid)
        if len(best_per_spec_pids) == n_specs:
            _close_dialogue(
                best_per_spec_pids,
                "success",
                query,
                steps,
                think=(
                    f"Anchor strategy could not find a single shop covering all {n_specs} specs "
                    f"(deadline guard or exhaustion). Falling back to top-1 per spec across shops: "
                    f"{best_per_spec_pids}. This may not satisfy the same-shop voucher constraint "
                    f"but preserves per-spec attribute coverage."
                ),
            )
            _log_stage(
                "shop_task",
                "Returned cross-shop top-1 fallback despite same-shop intent",
                pids=best_per_spec_pids,
            )
            return
    _close_dialogue(
        [Config.FALLBACK_PID],
        "failure",
        query,
        steps,
        think=f"Could not find a single shop carrying all {n_specs} required products.",
    )
    _log_stage(
        "shop_task",
        "Failed after exhausting full-coverage and fallback strategies",
        spec_count=n_specs,
        full_coverage_shops=0,
    )


def _handle_shop_task(params: dict, query: str, steps: list) -> None:
    specs = params.get("products", [])
    n_specs = len(specs)
    if not specs:
        _log_stage("shop_task", "No specs found; returning sentinel failure")
        _close_dialogue(
            [Config.FALLBACK_PID],
            "failure",
            query,
            steps,
            think="No product specs found in shop query.",
        )
        return
    kw_list = [s.get("keywords") or s.get("q", "") for s in specs]
    fallback_plan = (
        f"I am preparing a same-shop search for {n_specs} distinct products. "
        f"The per-item keywords are {kw_list}. "
        f"Declared price ranges: {[s.get('price_range') for s in specs]}. "
        f"Service filters: {[s.get('service') for s in specs]}. "
        f"My plan is to retrieve two pages per item, LLM-score the hits, then look for a shop "
        f"that covers every item before considering partial-coverage fallbacks."
    )
    think_analyze = LLMEngine._compose_step_narrative(
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
        fallback=fallback_plan,
    )
    _append_dialog_step(think_analyze, [], "", query, steps)
    _log_stage("shop_task", "Starting shop flow", spec_count=n_specs, keywords=kw_list)
    all_results, spec_scored, spec_scored_full = _shop_collect_and_score_specs(
        specs, query, steps
    )
    if any(len(s) == 0 for s in spec_scored):
        spec_scored = spec_scored_full
    filtered_results: list[list[Product]] = [
        [p for p, _ in scored] for scored in spec_scored
    ]
    shop_coverage = ShopResolver._assemble_shop_map(filtered_results)
    full_shops = [sid for sid, cov in shop_coverage.items() if len(cov) == n_specs]
    scoring_summary = [
        {
            "spec_idx": i,
            "keywords": specs[i].get("keywords", ""),
            "total_collected": len(all_results[i]),
            "passed_threshold": len(spec_scored[i]),
            "top_candidates": [
                {
                    "title": p.get("title", ""),
                    "price": p.get("price"),
                    "score": round(s, 1),
                }
                for p, s in spec_scored[i][:3]
            ],
        }
        for i in range(n_specs)
    ]
    fallback_scoring = (
        f"I finished LLM-scoring every page-1/2 hit for the {n_specs} items (threshold {Config.SHOP_SCORE_FLOOR}). "
        + " | ".join(
            f"spec[{i}] '{specs[i].get('keywords', '')[:40]}': {len(spec_scored[i])}/{len(all_results[i])} passed"
            for i in range(n_specs)
        )
        + f". Distinct shops covering every spec after filtering: {len(full_shops)}. "
        f"Next I decide whether to use a single shop, pick among several, or fall back to partial coverage."
    )
    think_coverage = LLMEngine._compose_step_narrative(
        query,
        {
            "scoring_summary": scoring_summary,
            "score_threshold": Config.SHOP_SCORE_FLOOR,
            "full_coverage_shops_found": len(full_shops),
        },
        fallback=fallback_scoring,
    )
    _append_dialog_step(think_coverage, [], "", query, steps)
    _log_stage(
        "shop_task",
        "Coverage matrix computed",
        full_coverage_shops=len(full_shops),
        n_specs=n_specs,
    )
    if len(full_shops) == 1:
        if _shop_resolve_single(full_shops[0], shop_coverage, n_specs, query, steps):
            return
    if len(full_shops) > 1:
        if _shop_resolve_multiple(full_shops, spec_scored, shop_coverage, n_specs, query, steps):
            return
    _shop_resolve_anchor_fallback(
        specs, spec_scored, shop_coverage, n_specs, params, query, steps
    )


def _voucher_null_scan(
    products: list[dict], allowed_total: float
) -> tuple[list[dict | None], list]:
    max_items: list[dict | None] = []
    scan_calls: list = []
    for spec in products:
        sp = QueryParser._build_spec_query(spec, include_price=False)
        sp["price"] = f"1-{allowed_total}"
        sp["sort"] = "pricedesc"
        found: list = []
        for page in range(1, 3):
            r = _execute_tool_with_retry("find_product", {**sp, "page": page})
            scan_calls.append(r)
            found.extend(r.get("result") or [])
            if found:
                break
        max_items.append(found[0] if found else None)
    return max_items, scan_calls


def _voucher_null_allocate(
    products: list[dict],
    max_items: list[dict | None],
    allowed_total: float,
    threshold: float,
    budget: float,
) -> tuple[dict[int, str], dict[int, str], float, list]:
    sorted_indices = sorted(
        range(len(max_items)), key=lambda i: float(max_items[i].get("price", 0))
    )
    prices = [float(max_items[i].get("price", 0)) for i in sorted_indices]
    above_index = (
        sorted_indices[-1] if threshold > 0 and prices[-1] >= threshold else None
    )
    pid_map: dict[int, str] = {}
    reason_map: dict[int, str] = {}
    spent = 0.0
    budget_calls: list = []
    processing_order = (
        [above_index] + [i for i in sorted_indices if i != above_index]
        if above_index is not None
        else sorted_indices
    )
    for i in processing_order:
        spec = products[i]
        remaining = allowed_total - spent
        sp = QueryParser._build_spec_query(spec, include_price=False)
        if i == above_index:
            sp["price"] = f"{budget:.0f}-{remaining:.0f}"
        else:
            sp["price"] = f"1-{remaining:.0f}"
        found: list = []
        for page in range(1, 3):
            r = _execute_tool_with_retry("find_product", {**sp, "page": page})
            budget_calls.append(r)
            found.extend(r.get("result") or [])
            if found:
                break
        if not found:
            _log_stage(
                "voucher_null_price",
                "Budget-phase query returned no candidates",
                spec_index=i,
                remaining=remaining,
                search_params=sp,
            )
            return {}, {}, spent, budget_calls
        pids_found = [str(p.get("product_id", "")) for p in found if p.get("product_id")]
        details = SearchAPI._retrieve_product_details(pids_found)
        picked = LLMEngine._llm_elect_best(
            str(spec.get("query") or spec.get("keywords") or ""),
            found,
            details,
            only_product_type=bool(spec.get("only_product_type", False)),
            model=Models.LLM_JUDGE_SECONDARY,
        )
        if picked is None:
            picked = found[0]
        pid_map[i] = str(picked["product_id"])
        reason_map[i] = picked.get("_llm_reason", "")
        spent += float(picked.get("price", 0) or 0)
        _log_stage(
            "voucher_null_price",
            "Picked product for spec during null-price budget phase",
            spec_index=i,
            picked_pid=pid_map.get(i),
            picked_price=picked.get("price"),
            running_spent=spent,
            allowed_total=allowed_total,
        )
    return pid_map, reason_map, spent, budget_calls


def _handle_voucher_null_price(
    products: list, voucher: dict, query: str, steps: list
) -> None:
    discount_type = voucher.get("discount_type", "percentage")
    discount_value = float(voucher.get("discount_value", 0))
    cap = float(voucher.get("cap", 0))
    threshold = float(voucher.get("threshold", 0))
    budget = float(voucher.get("budget", 0))
    if discount_value <= 0:
        _log_stage(
            "voucher_null_price",
            "Skipping null-price fallback due to non-positive discount",
            discount_value=discount_value,
        )
        return
    if discount_type == "fixed":
        allowed_total = budget + discount_value
    else:
        rate = discount_value / 100.0 if discount_value > 1 else discount_value
        if cap > 0 and budget * (discount_value / 100.0) > cap:
            allowed_total = budget + cap
        else:
            allowed_total = budget / (1 - rate) if rate < 1 else budget
    max_items, scan_calls = _voucher_null_scan(products, allowed_total)
    if any(item is None for item in max_items):
        _log_stage(
            "voucher_null_price",
            "Null-price pre-scan failed to find candidate for at least one spec",
            product_count=len(products),
        )
        return
    pid_map, reason_map, spent, budget_calls = _voucher_null_allocate(
        products, max_items, allowed_total, threshold, budget
    )
    pids = [pid_map[i] for i in range(len(products)) if i in pid_map]
    if len(pids) != len(products):
        return
    base = [
        {"product_id": pid_map.get(i, ""), "title": "", "price": None}
        for i in range(len(products))
    ]
    selected_info = _enrich_picks_for_narration(base)
    for i, entry in enumerate(selected_info):
        entry["llm_reason"] = reason_map.get(i, "")
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
            "top_product": (
                {
                    "product_id": str(max_items[i].get("product_id", "")) if max_items[i] else "",
                    "title": max_items[i].get("title", "") if max_items[i] else "",
                    "price": max_items[i].get("price") if max_items[i] else None,
                }
                if max_items[i]
                else None
            ),
        }
        for i in range(len(products))
    ]
    discount_desc = (
        f"fixed {discount_value}"
        if discount_type == "fixed"
        else f"{discount_value}%" + (f" capped at {cap}" if cap > 0 else "")
    )
    think_search = LLMEngine._compose_step_narrative(
        query,
        {
            "budget_constraint": voucher_constraint,
            "max_allowed_total": allowed_total,
            "candidates_per_product": top_cands,
        },
        fallback=(
            f"Voucher null-price scan: {len(products)} product(s). Budget={budget}, "
            f"discount={discount_desc}, threshold={threshold}, allowed_total={allowed_total:.2f}."
        ),
    )
    _append_dialog_step(think_search, scan_calls + budget_calls, "", query, steps)
    think_null = LLMEngine._compose_step_narrative(
        query,
        {
            "selected_products": selected_info,
            "total_spent": spent,
            "allowed_total": allowed_total,
            "budget_constraint": voucher_constraint,
        },
        fallback=(
            f"Selected {len(pids)} products within allowed total of {allowed_total:.2f} "
            f"(budget={budget}, discount={discount_desc}, threshold={threshold}). "
            f"Product IDs: {pids}, total spent: {spent:.2f}."
        ),
    )
    _close_dialogue(pids, "success", query, steps, think=think_null)
    _log_stage(
        "voucher_null_price",
        "Null-price fallback succeeded",
        pids=pids,
        total_spent=spent,
        allowed_total=allowed_total,
    )


def _voucher_build_candidate_pools(
    products: list[dict],
    voucher_ceiling: int | None,
    llm_pick_calls: list,
) -> list[list[Product]]:
    cand_products_llm: list[list[Product]] = []
    n_specs = len(products)
    if n_specs <= 2:
        sort_variants = ["pricedesc", "priceasc"]
    elif n_specs == 3:
        sort_variants = ["pricedesc"]
    else:
        sort_variants = []
    for spec in products:
        sp_broad = QueryParser._build_spec_query(spec, include_price=False)
        rare = QueryParser._rare_tokens_from_spec(spec)
        rankings: list[list[Product]] = []
        r_broad = _execute_tool_with_retry("find_product", {**sp_broad, "page": 1})
        llm_pick_calls.append(r_broad)
        rankings.append(r_broad.get("result") or [])
        if rare:
            sp_focused = dict(sp_broad)
            sp_focused["q"] = (str(sp_broad.get("q", "")) + " " + " ".join(rare)).strip()
            r_focused = _execute_tool_with_retry(
                "find_product", {**sp_focused, "page": 1}
            )
            llm_pick_calls.append(r_focused)
            rankings.append(r_focused.get("result") or [])
        else:
            r_broad_p2 = _execute_tool_with_retry(
                "find_product", {**sp_broad, "page": 2}
            )
            llm_pick_calls.append(r_broad_p2)
            rankings.append(r_broad_p2.get("result") or [])
        for sort_opt in sort_variants:
            sp_sorted = dict(sp_broad)
            if voucher_ceiling is not None and "price" not in sp_sorted:
                sp_sorted["price"] = f"0-{voucher_ceiling}"
            sp_sorted["sort"] = sort_opt
            r_sorted = _execute_tool_with_retry(
                "find_product", {**sp_sorted, "page": 1}
            )
            llm_pick_calls.append(r_sorted)
            rankings.append(r_sorted.get("result") or [])
        found_llm = SearchAPI._rrf_merge(rankings, top_n=15)
        if not found_llm and sp_broad.get("service"):
            sp_no_svc = {k: v for k, v in sp_broad.items() if k != "service"}
            r = _execute_tool_with_retry("find_product", {**sp_no_svc, "page": 1})
            llm_pick_calls.append(r)
            found_llm = SearchAPI._rrf_merge([r.get("result") or []], top_n=15)
        cand_products_llm.append(found_llm)
        _log_stage(
            "voucher_task",
            "Prepared merged candidates for spec",
            spec_keywords=spec.get("keywords"),
            rare_tokens=rare,
            ranking_count=len(rankings),
            merged_count=len(found_llm),
        )
    return cand_products_llm


def _voucher_llm_first_path(
    products: list[dict],
    cand_products_llm: list[list[Product]],
    voucher: dict,
    allowed_total: float,
    query: str,
    steps: list,
    llm_pick_calls: list,
) -> bool:
    n_specs = len(products)
    _log_stage(
        "voucher_task",
        "Built candidate pools for LLM-first selection",
        pool_sizes=[len(x) for x in cand_products_llm],
    )
    llm_picks: list[dict | None] = []
    for i, cands_i in enumerate(cand_products_llm):
        sq = products[i].get("query") or products[i].get("keywords") or query
        pids_i = [
            str(p.get("product_id", ""))
            for p in cands_i[:Config.MAX_JUDGE_CANDIDATES]
            if p.get("product_id")
        ]
        details_i = SearchAPI._retrieve_product_details(pids_i)
        pick = LLMEngine._llm_elect_best(
            str(sq),
            cands_i[:Config.MAX_JUDGE_CANDIDATES],
            details_i,
            only_product_type=bool(products[i].get("only_product_type", False)),
            model=Models.LLM_JUDGE_SECONDARY,
        )
        llm_picks.append(pick)
        _log_stage(
            "voucher_task",
            "Completed per-spec LLM election",
            spec_index=i,
            candidate_count=len(cands_i[:Config.MAX_JUDGE_CANDIDATES]),
            picked_pid=(str(pick.get("product_id", "")) if pick else None),
        )
    picks_for_feas: list[dict] = []
    for i in range(n_specs):
        resolved = llm_picks[i] if llm_picks[i] is not None else cand_products_llm[i][0]
        if resolved is not None:
            picks_for_feas.append(resolved)
    if len(picks_for_feas) != n_specs or not all(
        p.get("price") is not None for p in picks_for_feas
    ):
        return False
    top_prices = [str(p["price"]) for p in picks_for_feas]
    if VoucherUtils._verify_cart_budget(",".join(top_prices), voucher):
        pids = [str(p["product_id"]) for p in picks_for_feas]
        enriched = _enrich_picks_for_narration(
            [
                {
                    "product_id": str(p["product_id"]),
                    "title": p.get("title", ""),
                    "price": p.get("price"),
                }
                for p in picks_for_feas
            ]
        )
        pick_lines = "; ".join(
            f"spec[{i}] pid={str(p['product_id'])} '{str(p.get('title', ''))[:60]}' \u20b1{p.get('price')}"
            + (f" service={','.join(p.get('service'))}" if p.get("service") else "")
            for i, p in enumerate(picks_for_feas)
        )
        total_before = sum(float(x) for x in top_prices)
        think_ok = LLMEngine._compose_step_narrative(
            query,
            {
                "selected_products": enriched,
                "total_before_discount": total_before,
                "budget_constraint": voucher,
            },
            fallback=(
                f"After RRF retrieval (broad+focused, top-15 per spec) and the LLM batch-judge call, "
                f"the picks are: {pick_lines}. Cart total before discount: \u20b1{total_before:.2f}, "
                f"allowed_total \u20b1{allowed_total:.2f}, budget \u20b1{voucher.get('budget')}. "
                f"The combined cart fits the voucher constraint without needing a cheaper-retry."
            ),
        )
        _append_dialog_step(think_ok, llm_pick_calls, "", query, steps)
        _close_dialogue(pids, "success", query, steps)
        _log_stage(
            "voucher_task",
            "LLM-first picks satisfy voucher",
            pids=pids,
            total_before=total_before,
            allowed_total=allowed_total,
        )
        return True
    for i in range(n_specs):
        new_picks = list(picks_for_feas)
        best_cheap = Scoring._rerank_voucher_cheaper(
            cand_products_llm[i],
            str(products[i].get("query") or products[i].get("keywords") or query),
            prefer_cheaper=True,
        )
        if best_cheap:
            new_picks[i] = best_cheap
            try_prices = [
                str(p["price"]) for p in new_picks if p.get("price") is not None
            ]
            if len(try_prices) == n_specs and VoucherUtils._verify_cart_budget(
                ",".join(try_prices), voucher
            ):
                pids = [str(p["product_id"]) for p in new_picks]
                enriched_retry = _enrich_picks_for_narration(
                    [
                        {
                            "product_id": str(p.get("product_id", "")),
                            "title": p.get("title", ""),
                            "price": p.get("price"),
                        }
                        for p in new_picks
                    ]
                )
                think_retry = LLMEngine._compose_step_narrative(
                    query,
                    {
                        "selected_products": enriched_retry,
                        "total_before_discount": round(
                            sum(float(x) for x in try_prices), 2
                        ),
                        "budget_constraint": voucher,
                        "solver": "llm_first_cheaper_retry",
                        "retried_spec_index": i,
                    },
                    fallback=(
                        f"After the initial LLM picks exceeded the budget, I re-ranked candidates "
                        f"for spec {i} ('{products[i].get('keywords', '')}') with a cheaper-preference. "
                        f"New picks: "
                        + "; ".join(
                            f"spec[{j}] pid={str(p.get('product_id', ''))} '{str(p.get('title', ''))[:60]}' \u20b1{p.get('price')}"
                            for j, p in enumerate(new_picks)
                        )
                        + f". Cart total \u20b1{sum(float(x) for x in try_prices):.2f} now fits allowed_total \u20b1{allowed_total:.2f}."
                    ),
                )
                _append_dialog_step(think_retry, llm_pick_calls, "", query, steps)
                _close_dialogue(pids, "success", query, steps)
                _log_stage(
                    "voucher_task",
                    "Cheaper-retry achieved feasible cart",
                    retried_spec=i,
                    pids=pids,
                )
                return True
    kept_prices = [
        float(p.get("price") or 0.0) for p in picks_for_feas if p.get("price") is not None
    ]
    think_fallthrough = LLMEngine._compose_step_narrative(
        query,
        {
            "plan": "llm_first picks or cheaper-retry picks did not fit the voucher; falling through to the knapsack solver",
            "llm_first_total": round(sum(kept_prices), 2) if kept_prices else None,
            "allowed_total": round(allowed_total, 2),
            "budget_constraint": voucher,
            "n_specs": n_specs,
        },
        fallback=(
            f"LLM-first picks summed to {sum(kept_prices):.2f} with allowed_total={allowed_total:.2f}; "
            f"neither the direct combination nor the per-spec cheaper retry satisfied the voucher, "
            f"so I am handing off to the knapsack solver over the same candidate pool."
        ),
    )
    _append_dialog_step(think_fallthrough, llm_pick_calls, "", query, steps)
    _log_stage(
        "voucher_task",
        "Falling through to knapsack solver",
        allowed_total=allowed_total,
        llm_first_total=sum(kept_prices) if kept_prices else 0.0,
    )
    return False


def _voucher_knapsack_path(
    products: list[dict],
    cand_products_llm: list[list[Product]],
    voucher: dict,
    allowed_total: float,
    params: dict,
    query: str,
    steps: list,
) -> bool:
    _log_stage(
        "voucher_task",
        "Entering tiered knapsack stage",
        runtime=_session_runtime(),
        voucher_soft_deadline=Deadlines.VOUCHER_SOFT,
    )
    try:
        scored_per_spec = VoucherUtils._score_voucher_pool(products, cand_products_llm)
    except Exception:
        scored_per_spec = []
        _log_stage(
            "voucher_task",
            "Scoring pool failed before knapsack; proceeding to allocator fallback",
        )
    if not (scored_per_spec and all(scored_per_spec)):
        return False
    require_same_shop = bool(params.get("is_shop_voucher", False))
    ladder_result = VoucherUtils._knapsack_with_tiers(
        scored_per_spec,
        max_allowed_total=allowed_total,
        require_same_shop=require_same_shop,
        voucher=voucher,
    )
    if ladder_result is None:
        return False
    selection, ctx = ladder_result
    pids = [str(p.get("product_id", "")) for p in selection]
    total_price_k = float(ctx.get("total_price", 0.0))
    enriched = _enrich_picks_for_narration(
        [
            {
                "product_id": str(p.get("product_id", "")),
                "title": p.get("title", ""),
                "price": p.get("price"),
            }
            for p in selection
        ]
    )
    k_pick_lines = "; ".join(
        f"spec[{j}] pid={str(p.get('product_id', ''))} '{str(p.get('title', ''))[:60]}' \u20b1{p.get('price')}"
        for j, p in enumerate(selection)
    )
    think_k = LLMEngine._compose_step_narrative(
        query,
        {
            "selected_products": enriched,
            "total_before_discount": round(total_price_k, 2),
            "budget_constraint": voucher,
            "solver": "knapsack",
            "tier": ctx.get("tier"),
        },
        fallback=(
            f"The LLM-first picks did not fit; I ran the knapsack solver (tier='{ctx.get('tier')}') "
            f"over the RRF candidate pools and selected {len(selection)} product(s): {k_pick_lines}. "
            f"Cart total \u20b1{total_price_k:.2f}, allowed_total \u20b1{allowed_total:.2f}, voucher fits."
        ),
    )
    _append_dialog_step(think_k, [], "", query, steps)
    _close_dialogue(pids, "success", query, steps)
    _log_stage(
        "voucher_task",
        "Knapsack stage produced feasible cart",
        pids=pids,
        knapsack_context=ctx,
    )
    return True


def _voucher_allocator_path(
    products: list[dict],
    scored_per_spec: list[list[tuple[Product, float]]],
    max_prices: list[float],
    voucher: dict,
    allowed_total: float,
    params: dict,
    query: str,
    steps: list,
) -> bool:
    n_specs = len(products)
    remaining_order: list[int] = sorted(
        range(n_specs), key=lambda i: max_prices[i], reverse=True
    )
    picked_products: list[Product] = []
    picked_orig_idx: list[int] = []
    budget_tool_calls: list = []
    while remaining_order:
        position = len(picked_products)
        is_anchor = position == 0
        spent = sum(float(p.get("price", 0)) for p in picked_products)
        found_valid = False
        for candidate_i in list(remaining_order):
            others = [j for j in remaining_order if j != candidate_i]
            reserved = sum(
                lo or 0.0
                for j in others
                for lo, _ in [
                    _decode_price_band(str(products[j].get("price_range") or ""))
                ]
            )
            ceiling = allowed_total - spent - reserved
            if n_specs > 1:
                floor = allowed_total / n_specs if is_anchor else 1.0
            else:
                floor = 1.0
            orig_lo, orig_hi = _decode_price_band(
                str(products[candidate_i].get("price_range") or "")
            )
            final_lo = max(orig_lo if orig_lo is not None else 0.0, floor)
            final_hi = min(orig_hi if orig_hi is not None else float("inf"), ceiling)
            if final_lo > final_hi:
                continue
            sp = QueryParser._build_spec_query(products[candidate_i], include_price=False)
            sp["price"] = f"{final_lo:.0f}-{final_hi:.0f}"
            cands: list[Product] = []
            seen_pids: set[str] = set()
            for page in range(1, 3):
                r = _execute_tool_with_retry("find_product", {**sp, "page": page})
                budget_tool_calls.append(r)
                for p in r.get("result") or []:
                    pid = str(p.get("product_id", ""))
                    if pid and pid not in seen_pids:
                        cands.append(p)
                        seen_pids.add(pid)
            if not cands:
                continue
            spec_q = (
                products[candidate_i].get("query")
                or products[candidate_i].get("keywords")
                or query
            )
            top_cands = sorted(
                cands,
                key=lambda p: Scoring._heuristic_title_overlap(p, str(spec_q)),
                reverse=True,
            )[:10]
            pids_cands = [
                str(p.get("product_id", "")) for p in top_cands if p.get("product_id")
            ]
            details = SearchAPI._retrieve_product_details(pids_cands)
            chosen = LLMEngine._llm_elect_best(
                str(spec_q),
                top_cands,
                details,
                only_product_type=bool(
                    products[candidate_i].get("only_product_type", False)
                ),
                model=Models.LLM_JUDGE_SECONDARY,
            )
            if chosen is None:
                chosen = top_cands[0] if top_cands else cands[0]
            picked_products.append(chosen)
            picked_orig_idx.append(candidate_i)
            remaining_order = [j for j in remaining_order if j != candidate_i]
            found_valid = True
            break
        if not found_valid:
            think_fail = LLMEngine._compose_step_narrative(
                query,
                {
                    "position": position,
                    "allowed_total": round(allowed_total, 2),
                    "spent_so_far": round(spent, 2),
                    "note": f"No product found for spec at processing position {position} that fits within the remaining voucher budget.",
                },
                fallback=(
                    f"Could not find a suitable product for spec at position {position} "
                    f"within the remaining budget (spent={spent:.2f}, allowed_total={allowed_total:.2f})."
                ),
            )
            _append_dialog_step(think_fail, budget_tool_calls, "", query, steps)
            if scored_per_spec and all(scored_per_spec):
                fallback = VoucherUtils._cheapest_per_spec_fallback(
                    scored_per_spec,
                    require_same_shop=bool(params.get("is_shop_voucher", False)),
                )
                if fallback is not None:
                    selection_fb, _, total_fb = fallback
                    pids_fb = [str(p.get("product_id", "")) for p in selection_fb]
                    enriched_fb = _enrich_picks_for_narration(
                        [
                            {
                                "product_id": str(p.get("product_id", "")),
                                "title": p.get("title", ""),
                                "price": p.get("price"),
                            }
                            for p in selection_fb
                        ]
                    )
                    think_fb = LLMEngine._compose_step_narrative(
                        query,
                        {
                            "selected_products": enriched_fb,
                            "total_before_discount": round(total_fb, 2),
                            "budget_constraint": voucher,
                            "solver": "best_effort_cheapest",
                        },
                        fallback=(
                            f"Best-effort fallback: picked the cheapest product per spec (total {total_fb:.2f}) "
                            f"after the knapsack and allocator both failed to fit allowed_total={allowed_total:.2f}."
                        ),
                    )
                    _append_dialog_step(think_fb, [], "", query, steps)
                    _close_dialogue(pids_fb, "success", query, steps)
                    return True
            _close_dialogue(
                [Config.FALLBACK_PID],
                "failure",
                query,
                steps,
                think=(
                    f"Could not find a product for spec at position {position} within the voucher "
                    f"budget constraints (allowed_total={allowed_total:.2f})."
                ),
            )
            _log_stage(
                "voucher_task",
                "Allocator fallback exhausted and failed",
                allowed_total=allowed_total,
                picked_count=len(picked_products),
                remaining_count=len(remaining_order),
            )
            return False
    pid_map = {
        orig_idx: str(picked_products[k].get("product_id", ""))
        for k, orig_idx in enumerate(picked_orig_idx)
    }
    price_map = {
        orig_idx: float(picked_products[k].get("price", 0) or 0)
        for k, orig_idx in enumerate(picked_orig_idx)
    }
    pids = [pid_map[i] for i in range(n_specs)]
    total_price = sum(price_map.values())
    enriched = _enrich_picks_for_narration(
        [
            {
                "product_id": pid_map[i],
                "title": picked_products[picked_orig_idx.index(i)].get("title", ""),
                "price": picked_products[picked_orig_idx.index(i)].get("price"),
            }
            for i in range(n_specs)
        ]
    )
    think_done = LLMEngine._compose_step_narrative(
        query,
        {
            "selected_products": enriched,
            "total_before_discount": round(total_price, 2),
            "budget_constraint": voucher,
        },
        fallback=(
            f"Voucher search complete via reservation-window allocation. "
            f"Total before discount: {total_price:.2f}, allowed_total={allowed_total:.2f}, "
            f"budget={voucher.get('budget')}. Product IDs: {pids}."
        ),
    )
    _append_dialog_step(think_done, budget_tool_calls, "", query, steps)
    _close_dialogue(pids, "success", query, steps)
    return True


def _handle_voucher_task(params: dict, query: str, steps: list) -> None:
    is_shop = bool(params.get("is_shop_voucher", False)) or "same shop" in query.lower()
    products = params.get("products", [])
    voucher = VoucherUtils._default_voucher(params.get("voucher"))
    _log_stage(
        "voucher_task",
        "Starting voucher flow",
        is_shop=is_shop,
        product_count=len(products),
        voucher=voucher,
    )
    if is_shop and len(products) > 1:
        _log_stage(
            "voucher_task",
            "Same-shop voucher detected; invoking shop solver pre-pass",
            product_count=len(products),
        )
        pre_step_count = len(steps)
        _handle_shop_task(params, query, steps)
        if len(steps) > pre_step_count:
            last = steps[-1]
            try:
                msg = last.get("completion", {}).get("message", {})
                tool_calls = msg.get("tool_call") or []
                if tool_calls:
                    rec = next(
                        (tc for tc in tool_calls if tc.get("name") == "recommend_product"),
                        None,
                    )
                    if rec:
                        rec_pids_str = rec.get("parameters", {}).get("product_ids", "")
                        rec_pids = [
                            p.strip() for p in str(rec_pids_str).split(",") if p.strip()
                        ]
                        if rec_pids and rec_pids != [Config.FALLBACK_PID]:
                            if VoucherUtils._verify_shop_cart_budget(rec_pids, voucher):
                                _log_stage(
                                    "voucher_task",
                                    "Shop solver result passed voucher verification",
                                    pids=rec_pids,
                                )
                                return
            except Exception:
                pass
            _log_stage(
                "voucher_task",
                "Shop solver pre-pass completed without voucher-feasible result",
                pre_step_count=pre_step_count,
                post_step_count=len(steps),
            )
            return
    all_null = all(
        not p.get("price_range") and not p.get("service") for p in products
    )
    if all_null and len(steps) <= 1:
        _log_stage(
            "voucher_task",
            "Detected null-price/null-service case; delegating to fallback handler",
            product_count=len(products),
        )
        _handle_voucher_null_price(products, voucher, query, steps)
        if len(steps) > 1:
            return
    allowed_total = VoucherUtils.compute_voucher_ceiling(voucher)
    if not allowed_total or allowed_total <= 0:
        _log_stage(
            "voucher_task",
            "Failed to compute allowed total from voucher",
            voucher=voucher,
            allowed_total=allowed_total,
        )
        _close_dialogue(
            [Config.FALLBACK_PID],
            "failure",
            query,
            steps,
            think="Could not calculate allowed total from voucher parameters.",
        )
        return
    n_specs = len(products)
    if n_specs == 0:
        _close_dialogue(
            [Config.FALLBACK_PID],
            "failure",
            query,
            steps,
            think="No product specs found in voucher query.",
        )
        return
    kw_list = [p.get("keywords", "") for p in products]
    think_analyze = LLMEngine._compose_step_narrative(
        query,
        {
            "product_count": n_specs,
            "budget": voucher.get("budget"),
            "allowed_total": round(allowed_total, 2),
            "products": [
                {"keywords": p.get("keywords"), "price_range": p.get("price_range")}
                for p in products
            ],
        },
        fallback=(
            f"Voucher task: {n_specs} product(s). "
            f"Budget={voucher.get('budget')}, allowed_total={allowed_total:.2f}. "
            f"Keywords: {kw_list}."
        ),
    )
    scan_tool_calls: list = []
    max_prices: list[float] = []
    for spec in products:
        sp = QueryParser._build_spec_query(spec, include_price=False)
        sp["price"] = f"1-{allowed_total:.0f}"
        sp["sort"] = "pricedesc"
        r = _execute_tool_with_retry("find_product", sp)
        scan_tool_calls.append(r)
        found = r.get("result") or []
        max_prices.append(float(found[0].get("price", 0)) if found else 0.0)
    _append_dialog_step(think_analyze, scan_tool_calls, "", query, steps)
    _log_stage(
        "voucher_task",
        "Initial voucher scan complete",
        allowed_total=allowed_total,
        scan_call_count=len(scan_tool_calls),
        max_prices=max_prices,
    )
    voucher_ceiling = int(allowed_total) if allowed_total > 0 else None
    llm_pick_calls: list = []
    cand_products_llm = _voucher_build_candidate_pools(
        products, voucher_ceiling, llm_pick_calls
    )
    scored_per_spec: list[list[tuple[Product, float]]] = []
    if len(cand_products_llm) == n_specs and all(cand_products_llm):
        llm_first_success = _voucher_llm_first_path(
            products, cand_products_llm, voucher, allowed_total, query, steps, llm_pick_calls
        )
        if llm_first_success:
            return
        if _session_runtime() < Deadlines.VOUCHER_SOFT:
            knapsack_success = _voucher_knapsack_path(
                products, cand_products_llm, voucher, allowed_total, params, query, steps
            )
            if knapsack_success:
                return
            try:
                scored_per_spec = VoucherUtils._score_voucher_pool(
                    products, cand_products_llm
                )
            except Exception:
                scored_per_spec = []
    _voucher_allocator_path(
        products, scored_per_spec, max_prices, voucher, allowed_total, params, query, steps
    )


def agent_main(problem_data: dict) -> list[dict]:
    global _session_started_at
    _session_started_at = time.monotonic()
    _product_info_cache.clear()
    _narrator_state.remaining = Config.NARRATOR_BUDGET_PER_PROBLEM
    steps: list = []
    query: str = str(problem_data.get("query", ""))
    _log_stage(
        "agent_main",
        "Starting agent execution",
        query=query,
        problem_keys=sorted(list(problem_data.keys())),
    )
    try:
        task_type = QueryParser._infer_task_kind(query)
        _narrator_state.remaining = Config.NARRATOR_BUDGET_BY_TASK.get(
            task_type, Config.NARRATOR_BUDGET_PER_PROBLEM
        )
        params = QueryParser._analyze_user_query(query, task_type)
        _log_stage(
            "agent_main",
            "Parsed query successfully",
            task_type=task_type,
            params_preview=params,
        )
        products_info = params.get("products", []) or []
        keywords_list = [
            p.get("keywords") or p.get("q", "")
            for p in products_info
            if isinstance(p, dict)
        ]
        price_list = [
            p.get("price_range") for p in products_info if isinstance(p, dict)
        ]
        service_list = [p.get("service") for p in products_info if isinstance(p, dict)]
        fallback_init = (
            f"Query: '{query[:300]}'. Search keywords: {keywords_list}. "
            f"Price constraints: {price_list}. Service filters: {service_list}."
        )
        ctx_init: dict = {
            "keywords": keywords_list,
            "price_constraints": price_list,
            "service_filters": service_list,
        }
        if (
            products_info
            and isinstance(products_info[0], dict)
            and bool(products_info[0].get("only_product_type"))
        ):
            ctx_init["only_product_type"] = True
            ctx_init["only_product_type_reason"] = (
                "The query refers to the product type alone with no additional qualifiers "
                "(no brand, color, material, or numeric spec). Appending 'only' to the search "
                "query narrows results to this exact product type and avoids unrelated products "
                "that merely contain this term."
            )
        if params.get("voucher") and isinstance(params.get("voucher"), dict):
            v = params["voucher"]
            ctx_init["budget_constraint"] = {
                "discount_type": v.get("discount_type"),
                "discount_value": v.get("discount_value"),
                "threshold": v.get("threshold"),
                "cap": v.get("cap"),
                "budget": v.get("budget"),
            }
        think_init = LLMEngine._compose_step_narrative(
            query, ctx_init, fallback=fallback_init
        )
        _append_dialog_step(think_init, [], "", query, steps)
        if task_type == "shop":
            _handle_shop_task(params, query, steps)
        elif task_type == "voucher":
            _handle_voucher_task(params, query, steps)
        else:
            _handle_product_task(params, query, steps)
        _log_stage(
            "agent_main",
            "Task handler completed",
            task_type=task_type,
            step_count=len(steps),
        )
    except Exception as exc:
        exc_type = type(exc).__name__
        exc_msg = str(exc)[:200]
        try:
            _close_dialogue(
                [Config.FALLBACK_PID],
                "failure",
                query,
                steps,
                think=(
                    f"Encountered {exc_type} during agent execution: '{exc_msg}'. "
                    f"I inspected the query '{query[:200]}' and was unable to recover, "
                    f"so I am returning the fallback product id."
                ),
            )
        except Exception:
            steps.append(
                create_dialogue_step(
                    f"Encountered {exc_type} during agent execution and also failed to finish cleanly.",
                    [],
                    "Done.",
                    query,
                    len(steps) + 1,
                )
            )
        _log_stage(
            "agent_main",
            "Unhandled exception in agent_main",
            exception_type=exc_type,
            exception_message=exc_msg,
        )
    if not steps:
        steps.append(
            create_dialogue_step(
                f"No steps were generated for query '{query[:200]}'; emitting a sentinel step.",
                [],
                "Done.",
                query,
                1,
            )
        )
    _log_stage("agent_main", "Agent execution finished", final_step_count=len(steps))
    return steps
