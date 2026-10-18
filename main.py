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

REGIONS = ["Central", "East", "North", "South", "West"]
PRODUCTS = [
    "Air Fryer", "Blender", "Grinder", "Juicer",
    "Kettle", "Mixer", "Rice Cooker", "Toaster",
]
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

        latest = {}
        for o in raw:
            oid = o["id"]
            if oid not in latest or o["updated_at"] > latest[oid]["updated_at"]:
                latest[oid] = o
        ORDERS = list(latest.values())

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
{"action":"sum"|"avg"|"count"|"count_customers"|"top_product", "metric":"revenue"|"refund",
 "region":string|null, "product":string|null,
 "start_date":"YYYY-MM-DD"|null, "end_date":"YYYY-MM-DD"|null}

Rules:
- "top-selling", "best-selling" -> action: "top_product"
- "average", "mean", "how much is a ... worth" -> action: "avg"
- "how many distinct customers", "unique customers" -> action: "count_customers"
- "count", "how many orders", "how many" -> action: "count"
- default -> action: "sum"
- "refunded" -> metric: "refund", else "revenue"
/no_think
"""


class Question(BaseModel):
    question: str


def compute(filters: dict):
    action = filters.get("action", "sum")
    metric = filters.get("metric", "revenue")
    target_status = "paid" if metric == "revenue" else "refunded"
    region = filters.get("region")
    product = filters.get("product")
    start = filters.get("start_date")
    end = filters.get("end_date")

    filtered = []
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
        filtered.append(o)

    # 1. Count distinct customers
    if action == "count_customers":
        return len(set(o["customer"] for o in filtered))

    # 2. Standard count of orders
    if action == "count":
        return len(filtered)

    if not filtered:
        return "None" if action == "top_product" else 0.0

    # 3. Average logic
    if action == "avg":
        total_usd = sum(o["_usd_amount"] for o in filtered)
        return round(total_usd / len(filtered), 2)

    # 4. Top Product logic
    if action == "top_product":
        product_sums = {}
        for o in filtered:
            p = o["product"]
            product_sums[p] = product_sums.get(p, 0.0) + o["_usd_amount"]
        best_product = max(product_sums.items(), key=lambda x: x[1])
        return best_product[0]

    # 5. Default: Sum logic
    total_usd = sum(o["_usd_amount"] for o in filtered)
    return round(total_usd, 2)


def parse_question_regex(q: str) -> dict:
    ql = q.lower()

    # Action detection
    action = "sum"
    if "top-selling" in ql or "top selling" in ql or "best-selling" in ql:
        action = "top_product"
    elif "average" in ql or "mean" in ql or "avg" in ql or "how much is a" in ql or "worth" in ql:
        action = "avg"
    elif "distinct customer" in ql or "unique customer" in ql:
        action = "count_customers"
    elif "count" in ql or "how many" in ql:
        action = "count"

    # Metric detection
    metric = "refund" if any(w in ql for w in REFUND_WORDS) else "revenue"

    # Region detection
    region = None
    for r in sorted(REGIONS, key=len, reverse=True):
        if r.lower() in ql:
            region = r
            break

    # Product detection
    product = None
    for p in sorted(PRODUCTS, key=len, reverse=True):
        if p.lower() in ql:
            product = p
            break

    # Date detection
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

    qm = re.search(r"q([1-4])\s+(\d{4})", ql)
    if qm:
        qtr, yr = int(qm.group(1)), int(qm.group(2))
        first = (qtr - 1) * 3 + 1
        last = first + 2
        start = f"{yr}-{first:02d}-01"
        end = f"{yr}-{last:02d}-{calendar.monthrange(yr, last)[1]}"

    return {"action": action, "metric": metric, "region": region, "product": product,
            "start_date": start, "end_date": end}


async def parse_with_llm(question: str, hard_timeout: float = 5.0):
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
    except Exception as exc:
        print(f"LLM unavailable ({exc}); using regex", flush=True)
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
