"""
application/orchestration/chat_flow.py
The chat workflow (Rule 3): order the intercepts, then run the normal RAG turn.

Produces a single async generator of NDJSON lines (meta / token / done) so the
route stays a one-liner. Intercept order is owned by command_detector; the LLM
provider branch is owned by llm_client; the wire format by stream_formatter.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

from application.commands.command_detector import detect
from business_logic.meal_prep.shopify_meal_builder import add_meal
from config.model_settings import (
    ACTION_PLAN_MAX_TOKENS,
    CHAT_MAX_TOKENS,
    MODEL_DEEP,
    MODEL_FAST,
    SCRAPE_REVIEW_MAX_TOKENS,
)
from integrations.sheets.menu_context import get_menu_context
from model_runtime.clients import llm_client
from model_runtime.prompt_building.system_prompt_builder import build_system_prompt
from model_runtime.streaming.stream_formatter import done_line, meta_line, token_line
from web.scraping.page_fetcher import scrape_and_review
from web.search.ddgs_search import do_web_search
from web.search.search_decision import needs_web_search

_PERSONAL_SIGNALS = ["ptpreps", "my ", "i am", "i'm", "we ", "our ", "alex"]
_USER_CORRECTIONS = [
    "we already", "actually", "that's wrong", "not quite",
    "to clarify", "we don't", "we do", "i am", "i'm not",
]


def _plain_reply(conversation_id: str, reply: str, auto_ingested: bool = False):
    """Yield a complete single-reply NDJSON stream (meta + token + done)."""
    yield meta_line(conversation_id, web_search_used=False)
    yield token_line(reply)


async def stream_chat(req) -> AsyncIterator[str]:
    """Top-level chat generator: dispatch an intercept, else run the normal turn."""
    conversation_id = req.conversation_id or str(uuid.uuid4())
    kind, payload = detect(req.message, conversation_id)

    # ── Shopify intercept ─────────────────────────────────────────────────────
    if kind == "shopify":
        error, result = await add_meal(payload)
        reply = error if error else (
            f"Done! Created 4 products for {result['matched']} on Shopify:\n"
            + "".join(f"  ✓ {p['title']}\n" for p in result["created"])
        )
        for line in _plain_reply(conversation_id, reply):
            yield line
        return


    # ── Scrape intercept ──────────────────────────────────────────────────────
    if kind == "scrape":
        scrape_url = payload
        review_prompt = await scrape_and_review(scrape_url)
        if review_prompt.startswith("Couldn't") or review_prompt.startswith("Failed"):
            for line in _plain_reply(conversation_id, review_prompt):
                yield line
            return
        messages = [
            {"role": "system", "content": build_system_prompt(None, "", "")},
            {"role": "user", "content": review_prompt},
        ]
        yield meta_line(conversation_id, web_search_used=False)
        full = []
        try:
            async for token in llm_client.stream_tokens(
                messages, model=MODEL_FAST, max_tokens=SCRAPE_REVIEW_MAX_TOKENS
            ):
                full.append(token)
                yield token_line(token)
        except Exception as e:
            yield token_line(f"Review error: {e}")
        reply = "".join(full)
        yield done_line(auto_ingested=True)
        return

    # ── Normal chat flow ──────────────────────────────────────────────────────
    did_search = needs_web_search(req.message)
    web = await do_web_search(req.message) if did_search else ""
    has_personal = any(s in req.message.lower() for s in _PERSONAL_SIGNALS)

    sheets = get_menu_context(req.message)

    messages = [{"role": "system", "content": build_system_prompt(web, sheets)}]
    messages += req.history
    messages.append({"role": "user", "content": req.message})


    yield meta_line(conversation_id, web_search_used=did_search)

    full = []
    model = MODEL_DEEP if req.deep else MODEL_FAST
    try:
        async for token in llm_client.stream_tokens(
            messages, model=model, max_tokens=CHAT_MAX_TOKENS
        ):
            full.append(token)
            yield token_line(token)
    except Exception as e:
        print(f"stream_response error: {e}")
        yield token_line("\n\n[Response error — please try again]")

    reply = "".join(full)
