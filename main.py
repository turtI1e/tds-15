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
REFUND_WORDS = ["refund", "returned", "money back", "reversal", "reversed", "cancelled"]

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

        # Deduplicate
        latest = {}
        for o in raw:
            oid = o["id"]
            if oid not in latest or o["updated_at"] > latest[oid]["updated_at"]:
                latest[oid] = o
        ORDERS = list(latest.values())

        # Precompute USD amount and IST date
        for o in ORDERS:
            rate = RATES.get(o["currency"], 1.0)
            o["_usd_amount"] = o["amount"] * rate
            dt = datetime.fromisoformat(o["created_at"].replace("Z", "+00:00"))
            o["_ist_date"] = dt.astimezone(IST).strftime("%Y-%m-%d")

    print(f"Loaded {len(raw)} raw rows -> {len(ORDERS)} unique orders", flush=True)


class Question(BaseModel):
    question: str


def compute(filters: dict):
    action = filters.get("action", "sum")
    metric = filters.get("metric", "revenue")
    target_status = "paid" if metric == "revenue" else "refunded"
    region = filters.get("region")
    product = filters.get("product")
    customer = filters.get("customer")  # NEW: Customer filter
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
        if customer and o["customer"].lower() != customer.lower(): # NEW
            continue
        if start and o["_ist_date"] < start:
            continue
        if end and o["_ist_date"] > end:
            continue
        filtered.append(o)

    if not filtered:
        return "None" if action == "top_product" else 0.0

    if action == "count_customers":
        return len(set(o["customer"] for o in filtered))

    if action == "count":
        return len(filtered)

    if action == "avg":
        total_usd = sum(o["_usd_amount"] for o in filtered)
        return round(total_usd / len(filtered), 2)

    if action == "top_product":
        product_sums = {}
        for o in filtered:
            p = o["product"]
            product_sums[p] = product_sums.get(p, 0.0) + o["_usd_amount"]
        return max(product_sums.items(), key=lambda x: x[1])[0]

    total_usd = sum(o["_usd_amount"] for o in filtered)
    return round(total_usd, 2)


def parse_question_regex(q: str) -> dict:
    ql = q.lower()

    # 1. ACTION DETECTION (Using Regex word boundaries to prevent "account" -> "count")
    action = "sum"
    if re.search(r'\b(top|best|highest|most|maximum)[- ]?(selling|product|revenue|sold|money)\b', ql) or "which product" in ql:
        action = "top_product"
    elif re.search(r'\b(average|avg|mean|typical)\b', ql) or "how much is a" in ql or "worth" in ql:
        action = "avg"
    elif re.search(r'\b(distinct|unique|different)\b', ql) or re.search(r'\bcustomers\b', ql):
        action = "count_customers"
    elif re.search(r'\b(count|number|how many|total)\b', ql):
        action = "count"

    # 2. METRIC DETECTION
    metric = "refund" if any(re.search(rf'\b{w}\b', ql) for w in REFUND_WORDS) else "revenue"

    # 3. REGION DETECTION
    region = None
    for r in sorted(REGIONS, key=len, reverse=True):
        if re.search(rf'\b{r.lower()}\b', ql):
            region = r
            break

    # 4. PRODUCT DETECTION
    product = None
    for p in sorted(PRODUCTS, key=len, reverse=True):
        if re.search(rf'\b{p.lower()}\b', ql):
            product = p
            break

    # 5. CUSTOMER DETECTION (Look for C followed by digits, e.g., C1131)
    customer = None
    cm = re.search(r'\b(c\d+)\b', ql)
    if cm:
        customer = cm.group(1).upper()

    # 6. DATE DETECTION
    months = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
    start = end = None
    
    # Explicit Month + Year (e.g., "March 2026")
    m = re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})", ql)
    if m:
        mo, yr = months[m.group(1)], int(m.group(2))
        start = f"{yr}-{mo:02d}-01"
        end = f"{yr}-{mo:02d}-{calendar.monthrange(yr, mo)[1]}"

    # Quarter (e.g., "Q1 2026")
    qm = re.search(r"q([1-4])\s+(\d{4})", ql)
    if qm:
        qtr, yr = int(qm.group(1)), int(qm.group(2))
        first = (qtr - 1) * 3 + 1
        last = first + 2
        start = f"{yr}-{first:02d}-01"
        end = f"{yr}-{last:02d}-{calendar.monthrange(yr, last)[1]}"

    # Just a Year (e.g., "in 2026") -> Only trigger if start/end are still None
    if not start and not end:
        ym = re.search(r'\b(202\d)\b', ql)
        if ym:
            yr = ym.group(1)
            start = f"{yr}-01-01"
            end = f"{yr}-12-31"

    return {
        "action": action, "metric": metric, "region": region, 
        "product": product, "customer": customer,
        "start_date": start, "end_date": end
    }


@app.post("/answer")
async def answer(q: Question):
    filters = parse_question_regex(q.question)
    return {"answer": compute(filters)}

@app.get("/health")
async def health():
    return {"status": "ok"}
