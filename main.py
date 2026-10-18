import asyncio
import httpx
import json
import os
import calendar
import re
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

BASE = "https://exam.sanand.workers.dev/questionData?email=24f1000557%40ds.study.iitm.ac.in&quizSign=QtpAeIpTcVdmiQ4m17WU%2FIpTHNCcVjotS%2Ffl55lMMBefVprzNabSYYaogssxu4hrQeNakZu5dFRFqU58nr8kDNGZHWKhK3KIOKpWar6i35d0bdgispniVQYLn22Pap1NwuhGLjFNDNtLGyd%2FVU74Qqoo4a1rBaMxNfGPfbm%2Fm%2FSTGLT3sOBeJTW8nAkMQ7WSPAMWjN72rbHYNF7nW2npMSIjlE%2FifSeEkbKFge%2FYdayiHD2bXYrqMihhJiFo59mqvmoswu6lms2wD0upMupnTJMdcHO4f6OZ2lfsYEvjp38WG3U2rSqJigNlH6hhaCWVF3VsOET%2FDIA%2FvGMMjSoF6Q%3D%3D&questionId=q-ledger-agent-server"

IST = timezone(timedelta(hours=5, minutes=30))
app = FastAPI()

ORDERS = []
RATES = {}

# Real names from your /orders data — see check_data.py output.
REGIONS = ["Central", "East", "North", "South", "West"]
PRODUCTS = [
    "Air Fryer", "Blender", "Grinder", "Juicer",
    "Kettle", "Mixer", "Rice Cooker", "Toaster",
]

# Refund synonyms so paraphrases like "how much money was returned" work.
REFUND_WORDS = ["refund", "returned", "money back", "reversal", "reversed"]

USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
OLLAMA_BASE = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b-it-qat")

client = OpenAI(
    api_key="ollama",
    base_url=f"{OLLAMA_BASE}/v1",
    timeout=5.0,
    max_retries=0,
)


@app.on_event("startup")
async def load_data():
    global ORDERS, RATES
    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.get(BASE + "&path=%2Frates")
        RATES = r.json()["usd_per_unit"]

        raw = []
        url = BASE + "&path=%2Forders&page=1"
        while url:
            r = await http.get(url)
            data = r.json()
            raw.extend(data["orders"])
            url = data.get("next")

        # Deduplicate: keep the row with the latest updated_at per order id.
        latest = {}
        for o in raw:
            oid = o["id"]
            if oid not in latest or o["updated_at"] > latest[oid]["updated_at"]:
                latest[oid] = o
        ORDERS = list(latest.values())

        # Pre-compute USD amount and IST date once per order.
        for o in ORDERS:
            rate = RATES.get(o["currency"], 1.0)
            o["_usd_amount"] = o["amount"] * rate
            dt = datetime.fromisoformat(o["created_at"].replace("Z", "+00:00"))
            o["_ist_date"] = dt.astimezone(IST).strftime("%Y-%m-%d")

    print(f"Loaded {len(raw)} raw rows -> {len(ORDERS)} unique orders", flush=True)


SYSTEM_PROMPT = """
You convert a plain-English question about an order ledger into a JSON query plan.
Revenue = sum of status "paid". Refunds = sum of status "refunded".
Dates are in Asia/Kolkata timezone.

Return ONLY this JSON:
{"metric":"revenue"|"refund","region":string|null,"product":string|null,
 "start_date":"YYYY-MM-DD"|null,"end_date":"YYYY-MM-DD"|null}
/no_think
"""


class Question(BaseModel):
    question: str


def compute(filters: dict) -> float:
    metric = filters.get("metric", "revenue")
    target_status = "paid" if metric == "revenue" else "refunded"
    region = filters.get("region")
    product = filters.get("product")
    start = filters.get("start_date")
    end = filters.get("end_date")

    total = 0.0
    for o in ORDERS:
        if o["status"] != target_status:
            continue
        if region and o["region"].lower() != region.lower():
            continue
        if product and o["product"].lower() != product.lower():
            continue
        if start and o["_ist_date"] < start:
            continue
        if end and o["_ist_date"] > end:
            continue
        total += o["_usd_amount"]
    return round(total, 2)


def parse_question_regex(q: str) -> dict:
    ql = q.lower()

    # Metric: refund if any synonym is present, otherwise revenue.
    metric = "refund" if any(w in ql for w in REFUND_WORDS) else "revenue"

    # Region: longest-name-first so multi-word names win.
    region = None
    for r in sorted(REGIONS, key=len, reverse=True):
        if r.lower() in ql:
            region = r
            break

    # Product: longest-name-first so "Rice Cooker" is matched before any
    # accidental substring, and "Air Fryer" is preserved as two words.
    product = None
    for p in sorted(PRODUCTS, key=len, reverse=True):
        if p.lower() in ql:
            product = p
            break

    # Month + Year
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    start = end = None
    m = re.search(
        r"(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)\s+(\d{4})",
        ql,
    )
    if m:
        mo, yr = months[m.group(1)], int(m.group(2))
        start = f"{yr}-{mo:02d}-01"
        end = f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    # Quarter
    qm = re.search(r"q([1-4])\s+(\d{4})", ql)
    if qm:
        qtr, yr = int(qm.group(1)), int(qm.group(2))
        first = (qtr - 1) * 3 + 1
        last = first + 2
        start = f"{yr}-{first:02d}-01"
        end = f"{yr}-{last:02d}-{calendar.monthrange(yr, last)[1]}"

    return {"metric": metric, "region": region, "product": product,
            "start_date": start, "end_date": end}


async def parse_with_llm(question: str, hard_timeout: float = 5.0):
    """Call Ollama with a true hard timeout. Returns None on any failure."""
    def _sync_call():
        return client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

    try:
        resp = await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=hard_timeout)
        return json.loads(resp.choices[0].message.content)
    except (asyncio.TimeoutError, Exception) as exc:
        print(f"LLM unavailable/timed out ({type(exc).__name__}: {exc}); using regex", flush=True)
        return None


@app.post("/answer")
async def answer(q: Question):
    filters = None
    if USE_OLLAMA:
        filters = await parse_with_llm(q.question)
    if filters is None:
        filters = parse_question_regex(q.question)
    return {"answer": compute(filters)}


@app.get("/health")
async def health():
    return {"status": "ok"}
