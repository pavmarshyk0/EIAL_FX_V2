"""Optional Ollama proposer. Model output is UNTRUSTED: only allowlisted JSON recipes run."""
from __future__ import annotations
import json
from urllib.request import Request, urlopen
from urllib.parse import urlsplit

PROMPT = """You are a skeptical quant researcher. Propose at most {n} DISTINCT testable FX daily-reference-rate factor hypotheses.
All data are known by the daily ECB fixing at time t; prediction horizon is next fixing.
Return ONLY one JSON object with a single key 'candidates' holding an array of objects.
Allowed candidate specs only:
{{"family":"momentum" or "zscore" or "vol_ratio" or "acceleration" or "cross_difference","source":"eurusd" or "eurpln" or "usdpln","lookback":3 or 5 or 10 or 20 or 40,"short":5 or 10,"long":20 or 40,"other":"eurusd" or "eurpln" or "usdpln","thesis":"short economic hypothesis"}}
For momentum/zscore/acceleration use source and lookback. For vol_ratio use source, short and long (short < long). For cross_difference use source, other and lookback (distinct pairs).
Do not propose future labels, bank private data, nonexistent feeds, arbitrary Python, or unmeasurable AI-latency/CBDC proxies.
Consider lead-lag, volatility, risk-on/off and microstructure limitations.
"""


def propose_ollama(model: str, n: int = 8, endpoint: str = "http://127.0.0.1:11434", timeout: int = 75) -> list[dict]:
    """No remote URLs accepted; avoids sending strategy context to unknown servers."""
    host = urlsplit(endpoint)
    if host.scheme != "http" or host.hostname not in {"127.0.0.1", "localhost", "::1"} or host.username or host.password:
        raise ValueError("Ollama endpoint must be local HTTP (127.0.0.1/localhost)")
    req = Request(endpoint.rstrip("/") + "/api/generate",
                  data=json.dumps({"model": model, "prompt": PROMPT.format(n=n),
                                   "stream": False, "format": "json", "options": {"temperature": 0.5}}).encode("utf-8"),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=timeout) as response:
        raw = json.load(response)
    body = raw.get("response", "")
    parsed = json.loads(body)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
        raise ValueError("LLM response must be {candidates: [...]} JSON")
    return parsed["candidates"][:n]
