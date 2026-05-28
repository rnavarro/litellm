"""
Background poller for Z.AI quota.

Z.AI does not return quota information in chat-completion response headers (unlike
Synthetic's x-synthetic-quotas or NeuralWatt's x-energy-* headers). It only exposes
quota via a separate monitor endpoint. This module polls that endpoint on an interval
(registered as an APScheduler job in proxy_server.py) and caches the result in-process.

The cached value is read by spend_tracking_utils.get_logging_payload(), which stamps it
onto Z.AI spend-log rows as a synthetic "x-zai-quota" entry in provider_response_headers,
so Z.AI quota lands in the same place and shape as the other providers' quota headers.
"""
import os
import time
from typing import Any, Dict

from litellm._logging import verbose_proxy_logger

ZAI_QUOTA_API_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

# Per-process cache. Each uvicorn worker maintains its own; the consumer reads it
# in-process, so no cross-worker sharing is needed.
_ZAI_QUOTA_CACHE: Dict[str, Any] = {"data": None, "fetched_at": 0.0, "error": None}


def get_cached_zai_quota() -> Dict[str, Any]:
    return _ZAI_QUOTA_CACHE


async def poll_zai_quota() -> None:
    """Fetch Z.AI quota and update the in-process cache. Keeps last-known data on error."""
    import aiohttp

    api_key = os.environ.get("ZAI_API_KEY")
    if not api_key:
        return
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                ZAI_QUOTA_API_URL,
                # Pi's zai-usage extension found undici fails to decompress gzip here;
                # request identity encoding to keep the JSON body parseable.
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Accept-Encoding": "identity",
                },
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    _ZAI_QUOTA_CACHE["error"] = f"http{resp.status}"
                    return
                body = await resp.json()
    except Exception as e:  # noqa: BLE001 - network/parse errors keep last-known data
        _ZAI_QUOTA_CACHE["error"] = str(e)
        verbose_proxy_logger.debug("zai_quota_poll error: %s", e)
        return

    # Z.AI can return HTTP 200 with an error body, e.g.
    # {"code":401,"msg":"token expired or incorrect","success":false}
    if isinstance(body, dict) and body.get("success") is False:
        _ZAI_QUOTA_CACHE["error"] = body.get("msg") or "api_error"
        return

    limits = (body or {}).get("data", {}).get("limits")
    if limits is None:
        _ZAI_QUOTA_CACHE["error"] = "no_limits_in_response"
        return

    _ZAI_QUOTA_CACHE.update(
        data={"limits": limits, "level": body["data"].get("level")},
        fetched_at=time.time(),
        error=None,
    )
