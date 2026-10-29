""" Beaver's Choice Paper Company - complete single-file submission.

Five PydanticAI agents: Orchestrator, Inventory, Quoting, Sales, Business Advisor.
Requirements: pandas, numpy, sqlalchemy, python-dotenv, pydantic-ai-slim[openai].
The supplied 2025 CSV samples are historical simulations, not live customer orders.

Run: python project_starter.py --simulate-acceptance
Optional: --csv quote_requests_sample.csv --auto-reorder --skip-worker-narration
Do not commit .env, API credentials, or munder_difflin.db to public GitHub.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Literal, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sqlalchemy import Engine, create_engine, text
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
DB_FILE = ROOT / 'munder_difflin.db'
db_engine = create_engine('sqlite:///' + str(DB_FILE))

# Original course catalog: exact item names are transaction identifiers.
paper_supplies = [
    {'item_name': 'A4 paper', 'category': 'paper', 'unit_price': .05},
    {'item_name': 'Letter-sized paper', 'category': 'paper', 'unit_price': .06},
    {'item_name': 'Cardstock', 'category': 'paper', 'unit_price': .15},
    {'item_name': 'Colored paper', 'category': 'paper', 'unit_price': .10},
    {'item_name': 'Glossy paper', 'category': 'paper', 'unit_price': .20},
    {'item_name': 'Matte paper', 'category': 'paper', 'unit_price': .18},
    {'item_name': 'Recycled paper', 'category': 'paper', 'unit_price': .08},
    {'item_name': 'Eco-friendly paper', 'category': 'paper', 'unit_price': .12},
    {'item_name': 'Poster paper', 'category': 'paper', 'unit_price': .25},
    {'item_name': 'Banner paper', 'category': 'paper', 'unit_price': .30},
    {'item_name': 'Kraft paper', 'category': 'paper', 'unit_price': .10},
    {'item_name': 'Construction paper', 'category': 'paper', 'unit_price': .07},
    {'item_name': 'Wrapping paper', 'category': 'paper', 'unit_price': .15},
    {'item_name': 'Glitter paper', 'category': 'paper', 'unit_price': .22},
    {'item_name': 'Decorative paper', 'category': 'paper', 'unit_price': .18},
    {'item_name': 'Letterhead paper', 'category': 'paper', 'unit_price': .12},
    {'item_name': 'Legal-size paper', 'category': 'paper', 'unit_price': .08},
    {'item_name': 'Crepe paper', 'category': 'paper', 'unit_price': .05},
    {'item_name': 'Photo paper', 'category': 'paper', 'unit_price': .25},
    {'item_name': 'Uncoated paper', 'category': 'paper', 'unit_price': .06},
    {'item_name': 'Butcher paper', 'category': 'paper', 'unit_price': .10},
    {'item_name': 'Heavyweight paper', 'category': 'paper', 'unit_price': .20},
    {'item_name': 'Standard copy paper', 'category': 'paper', 'unit_price': .04},
    {'item_name': 'Bright-colored paper', 'category': 'paper', 'unit_price': .12},
    {'item_name': 'Patterned paper', 'category': 'paper', 'unit_price': .15},
    {'item_name': 'Paper plates', 'category': 'product', 'unit_price': .10},
    {'item_name': 'Paper cups', 'category': 'product', 'unit_price': .08},
    {'item_name': 'Paper napkins', 'category': 'product', 'unit_price': .02},
    {'item_name': 'Disposable cups', 'category': 'product', 'unit_price': .10},
    {'item_name': 'Table covers', 'category': 'product', 'unit_price': 1.50},
    {'item_name': 'Envelopes', 'category': 'product', 'unit_price': .05},
    {'item_name': 'Sticky notes', 'category': 'product', 'unit_price': .03},
    {'item_name': 'Notepads', 'category': 'product', 'unit_price': 2.00},
    {'item_name': 'Invitation cards', 'category': 'product', 'unit_price': .50},
    {'item_name': 'Flyers', 'category': 'product', 'unit_price': .15},
    {'item_name': 'Party streamers', 'category': 'product', 'unit_price': .05},
    {'item_name': 'Decorative adhesive tape (washi tape)', 'category': 'product', 'unit_price': .20},
    {'item_name': 'Paper party bags', 'category': 'product', 'unit_price': .25},
    {'item_name': 'Name tags with lanyards', 'category': 'product', 'unit_price': .75},
    {'item_name': 'Presentation folders', 'category': 'product', 'unit_price': .50},
    {'item_name': 'Large poster paper (24x36 inches)', 'category': 'large_format', 'unit_price': 1.00},
    {'item_name': 'Rolls of banner paper (36-inch width)', 'category': 'large_format', 'unit_price': 2.50},
    {'item_name': '100 lb cover stock', 'category': 'specialty', 'unit_price': .50},
    {'item_name': '80 lb text paper', 'category': 'specialty', 'unit_price': .40},
    {'item_name': '250 gsm cardstock', 'category': 'specialty', 'unit_price': .30},
    {'item_name': '220 gsm poster paper', 'category': 'specialty', 'unit_price': .35},
]


def sql_df(query: str, params: dict | None = None) -> pd.DataFrame:
    """SQLAlchemy 2 compatible DataFrame query helper."""
    with db_engine.connect() as conn:
        return pd.read_sql(text(query), conn, params=params or {})


def generate_sample_inventory(paper_supplies: list, coverage: float = .4, seed: int = 137) -> pd.DataFrame:
    """Original starter behavior: seeded 40% catalog selection, stock 200-799."""
    np.random.seed(seed)
    selected = np.random.choice(range(len(paper_supplies)), size=int(len(paper_supplies) * coverage), replace=False)
    result = []
    for i in selected:
        item = paper_supplies[int(i)]
        result.append(dict(item, current_stock=int(np.random.randint(200, 800)),
                           min_stock_level=int(np.random.randint(50, 150))))
    return pd.DataFrame(result)


def init_database(db_engine: Engine, seed: int = 137) -> Engine:
    """Initialize the starter's transactions, historical requests/quotes, and stock.

    WARNING: intentionally resets all tables. Use on a disposable test DB only.
    """
    requests = pd.read_csv(ROOT / 'quote_requests.csv')
    requests['id'] = range(1, len(requests) + 1)
    quotes = pd.read_csv(ROOT / 'quotes.csv')
    if len(quotes) != len(requests):
        raise ValueError('Historical quote and request files must have matching row counts')
    quotes['request_id'] = range(1, len(quotes) + 1)
    quotes['order_date'] = '2025-01-01'
    def metadata(raw):
        try:
            obj = ast.literal_eval(raw) if isinstance(raw, str) else raw
            return obj if isinstance(obj, dict) else {}
        except (ValueError, SyntaxError):
            return {}
    meta = quotes['request_metadata'].map(metadata)
    for dest, key in [('job_type', 'job_type'), ('order_size', 'order_size'), ('event_type', 'event_type')]:
        quotes[dest] = meta.map(lambda d: d.get(key, ''))
    quotes = quotes[['request_id','total_amount','quote_explanation','order_date','job_type','order_size','event_type']]
    stock = generate_sample_inventory(paper_supplies, seed=seed)
    # Initial $50K cash is represented as the same seed credit as in Udacity starter.
    transactions = [{'item_name': None, 'transaction_type': 'sales', 'units': None,
                     'price': 50000., 'transaction_date': '2025-01-01'}]
    for row in stock.itertuples(index=False):
        transactions.append({'item_name': row.item_name, 'transaction_type': 'stock_orders',
                             'units': int(row.current_stock), 'price': float(row.current_stock * row.unit_price),
                             'transaction_date': '2025-01-01'})
    with db_engine.begin() as conn:
        requests.to_sql('quote_requests', conn, if_exists='replace', index=False)
        quotes.to_sql('quotes', conn, if_exists='replace', index=False)
        stock.to_sql('inventory', conn, if_exists='replace', index=False)
        pd.DataFrame(transactions).to_sql('transactions', conn, if_exists='replace', index=False)
    return db_engine


def create_transaction(item_name: str, transaction_type: str, quantity: int,
                       price: float, date: Union[str, datetime]) -> int:
    """Starter transaction API. Purchase increases stock, sale decreases stock."""
    if transaction_type not in {'stock_orders', 'sales'}:
        raise ValueError("Transaction type must be 'stock_orders' or 'sales'")
    if item_name not in {x['item_name'] for x in paper_supplies} or quantity <= 0 or price < 0:
        raise ValueError('Invalid catalog item, quantity, or transaction amount')
    when = date.isoformat() if isinstance(date, datetime) else str(date)
    with db_engine.begin() as conn:
        result = conn.execute(text('''INSERT INTO transactions
            (item_name, transaction_type, units, price, transaction_date)
            VALUES (:item, :type, :units, :price, :when)'''),
            {'item': item_name, 'type': transaction_type, 'units': int(quantity),
             'price': float(price), 'when': when})
        return int(result.lastrowid)


def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """Starter inventory snapshot, positive stock as of date."""
    result = sql_df('''SELECT item_name, SUM(CASE WHEN transaction_type='stock_orders' THEN units
        WHEN transaction_type='sales' THEN -units ELSE 0 END) AS stock
        FROM transactions WHERE item_name IS NOT NULL AND transaction_date <= :d
        GROUP BY item_name HAVING stock > 0''', {'d': str(as_of_date)})
    return {str(r.item_name): int(r.stock) for r in result.itertuples(index=False)}


def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """Starter single-item stock snapshot, DataFrame with current_stock field."""
    day = as_of_date.isoformat() if isinstance(as_of_date, datetime) else str(as_of_date)
    data = sql_df('''SELECT COALESCE(SUM(CASE WHEN transaction_type='stock_orders' THEN units
        WHEN transaction_type='sales' THEN -units ELSE 0 END), 0) AS current_stock
        FROM transactions WHERE item_name=:item AND transaction_date<=:d''',
        {'item': item_name, 'd': day})
    data.insert(0, 'item_name', item_name)
    return data


def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """Starter supplier estimate: <=10:0d, <=100:1d, <=1000:4d, else:7d."""
    day = datetime.fromisoformat(str(input_date_str)[:10]).date()
    days = 0 if quantity <= 10 else 1 if quantity <= 100 else 4 if quantity <= 1000 else 7
    return (day + timedelta(days=days)).isoformat()


def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """Starter cash calculation: revenue minus stock purchases as of date."""
    day = as_of_date.isoformat() if isinstance(as_of_date, datetime) else str(as_of_date)
    data = sql_df('''SELECT COALESCE(SUM(CASE WHEN transaction_type='sales' THEN price
        WHEN transaction_type='stock_orders' THEN -price ELSE 0 END),0) AS cash
        FROM transactions WHERE transaction_date <= :d''', {'d': day})
    return float(data.iloc[0]['cash'])


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """Starter financial snapshot: cash, inventory, asset value and best sellers."""
    day = as_of_date.isoformat() if isinstance(as_of_date, datetime) else str(as_of_date)
    inv = sql_df('SELECT item_name, unit_price FROM inventory')
    breakdown = []
    for row in inv.itertuples(index=False):
        units = int(get_stock_level(row.item_name, day).iloc[0]['current_stock'])
        breakdown.append({'item_name': row.item_name, 'stock': units,
                          'unit_price': float(row.unit_price), 'value': round(units * row.unit_price, 2)})
    stock_value = round(sum(r['value'] for r in breakdown), 2)
    cash = get_cash_balance(day)
    leaders = sql_df('''SELECT item_name, SUM(units) AS total_units, SUM(price) AS total_revenue
        FROM transactions WHERE transaction_type='sales' AND item_name IS NOT NULL
        AND transaction_date<=:d GROUP BY item_name ORDER BY total_revenue DESC LIMIT 5''', {'d': day})
    return {'as_of_date': day, 'cash_balance': cash, 'inventory_value': stock_value,
            'total_assets': round(cash + stock_value, 2), 'inventory_summary': breakdown,
            'top_selling_products': leaders.to_dict(orient='records')}


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Starter historical quote lookup, same fields and AND-term matching convention."""
    terms = [str(t).strip().lower() for t in search_terms if str(t).strip()][:5]
    predicates = []
    params: dict = {'n': max(1, int(limit))}
    for i, term in enumerate(terms):
        predicates.append(f'(LOWER(r.response) LIKE :t{i} OR LOWER(q.quote_explanation) LIKE :t{i})')
        params[f't{i}'] = f'%{term}%'
    where = ' AND '.join(predicates) or '1=1'
    result = sql_df(f'''SELECT r.response AS original_request, q.total_amount, q.quote_explanation,
        q.job_type, q.order_size, q.event_type, q.order_date FROM quotes q
        JOIN quote_requests r ON q.request_id=r.id WHERE {where}
        ORDER BY q.order_date DESC LIMIT :n''', params)
    return result.to_dict(orient='records')

# ============================================================================
# YOUR MULTI AGENT STARTS HERE: 1 orchestrator + 4 specialist agents = 5.
# ============================================================================

PROXY_URL = os.getenv("UDACITY_OPENAI_BASE_URL", "https://openai.vocareum.com/v1")
MODEL_NAME = os.getenv("UDACITY_MODEL", "gpt-4o-mini")  # Change if proxy model list differs.
API_KEY = os.getenv("UDACITY_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# Illustrative customer-facing sales policy. Update after reviewing client policy.
BASE_MARKUP = Decimal("1.25")
BULK_TIERS = [(1000, Decimal("0.15")), (500, Decimal("0.10")), (100, Decimal("0.05"))]
MIN_MARKUP = Decimal("1.05")
REPORT_EVERY = 5
WORKER_NARRATION = True  # Disable only for faster local diagnostics.
_WRITE_ALLOWED = ContextVar("allow_sales_writes", default=False)


class RequestedItem(BaseModel):
    """One paper product requested, with a positive requested quantity."""
    description: str
    quantity: int = Field(ge=1)


class RoutedRequest(BaseModel):
    """Structured message interpreted by the orchestration agent."""
    intent: Literal["inventory", "quote", "order", "report", "unknown"]
    items: list[RequestedItem] = Field(default_factory=list)
    deadline: str | None = None  # Explicit YYYY-MM-DD customer latest acceptable date.
    explanation: str = ""


@dataclass
class PlannedLine:
    item_name: str
    quantity: int
    unit_price: Decimal
    total: Decimal
    discount: Decimal
    in_stock: int
    supplier_eta: str | None
    status: str  # ready | shortage | deadline_failed | unknown
    cost: Decimal


def iso_date(raw: str | None, default: str | None = None) -> str:
    """Parse only well-defined ISO-like dates; do not silently invent a deadline."""
    if raw is None or str(raw).strip() == "":
        if default is None:
            raise ValueError("A request date is required")
        return default
    return date.fromisoformat(str(raw).strip().split("T")[0]).isoformat()


def money(x: Decimal | float | int | str) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def catalog() -> dict[str, dict]:
    return {x["item_name"]: x for x in paper_supplies}


def resolve_item(description: str) -> str | None:
    """Map customer wording to a catalog SKU; do not invent unavailable SKUs.

    The starter catalog does not encode paper size for most specialty products.
    A phrase such as 'A4 glossy paper' therefore resolves to 'Glossy paper'
    (stock and price are based on that generic SKU); it does NOT guarantee A4.
    """
    raw = re.sub(r"[^a-z0-9]+", " ", description.lower()).strip()
    if not raw:
        return None
    names = list(catalog())
    for name in names:
        if re.sub(r"[^a-z0-9]+", " ", name.lower()).strip() == raw:
            return name

    # High-confidence product aliases (descriptors/packing may be present).
    # Priority matters: specific modifiers must precede generic 'paper'.
    aliases = [
        (r"washi|decorative adhesive tape", "Decorative adhesive tape (washi tape)"),
        (r"streamer", "Party streamers"),
        (r"balloon|ribbon|glue|sticker", None),
        (r"lanyard|name tag", "Name tags with lanyards"),
        (r"presentation folder", "Presentation folders"),
        (r"poster board|24\s*x\s*36", "Large poster paper (24x36 inches)"),
        (r"250\s*gsm", "250 gsm cardstock"),
        (r"220\s*gsm", "220 gsm poster paper"),
        (r"100\s*lb", "100 lb cover stock"),
        (r"80\s*lb", "80 lb text paper"),
        (r"banner.*roll|roll.*banner", "Rolls of banner paper (36-inch width)"),
        (r"bright.*color", "Bright-colored paper"),
        (r"construction", "Construction paper"),
        (r"cardstock|card stock", "Cardstock"),
        (r"glossy", "Glossy paper"),
        (r"matte", "Matte paper"),
        (r"recycled", "Recycled paper"),
        (r"eco.friendly", "Eco-friendly paper"),
        (r"poster", "Poster paper"),
        (r"kraft", "Kraft paper"),
        (r"patterned", "Patterned paper"),
        (r"colored|colour|colorful|coloured", "Colored paper"),
        (r"photo", "Photo paper"),
        (r"envelope", "Envelopes"),
        (r"sticky notes?", "Sticky notes"),
        (r"notepad", "Notepads"),
        (r"invitation card", "Invitation cards"),
        (r"fl(y|ie)ers?", "Flyers"),
        (r"napkin", "Paper napkins"),
        (r"disposable cup", "Disposable cups"),
        (r"paper cup|cups?", "Paper cups"),
        (r"paper plate|plates?", "Paper plates"),
        (r"table cover", "Table covers"),
        (r"party bag", "Paper party bags"),
        (r"standard.*(printer|copy)|white printer|printer paper|copy paper", "Standard copy paper"),
        (r"letterhead", "Letterhead paper"),
        (r"legal", "Legal-size paper"),
        (r"letter.sized|8\s*5\s*x\s*11", "Letter-sized paper"),
        (r"\ba4\b", "A4 paper"),
        (r"\ba3\b|\ba5\b", None),
    ]
    for pattern, item_name in aliases:
        if re.search(pattern, raw):
            return item_name

    # Conservative fuzzy fallback for misspellings of well-defined SKUs.
    scored = sorted(((SequenceMatcher(None, raw, name.lower()).ratio(), name) for name in names), reverse=True)
    if scored and scored[0][0] >= .89 and (len(scored) == 1 or scored[0][0] - scored[1][0] > .1):
        return scored[0][1]
    return None


def on_hand(name: str, as_of: str) -> int:
    frame = get_stock_level(name, as_of)
    if getattr(frame, "empty", True):
        return 0
    return max(0, int(frame.iloc[0]["current_stock"]))


def discount_for(quantity: int) -> Decimal:
    return next((discount for threshold, discount in BULK_TIERS if quantity >= threshold), Decimal("0"))


def price_line(name: str, quantity: int) -> tuple[Decimal, Decimal, Decimal]:
    """Calculate policy-based quote, using historical precedent for promotions.

    Past quotes inform a small additional discount when a product has multiple
    comparable historical offers. They do not override verified catalog prices.
    """
    supplier_unit = Decimal(str(catalog()[name]["unit_price"]))
    discount = discount_for(quantity)
    historical_terms = [w for w in re.findall(r"[a-z0-9]+", name.lower()) if w != 'paper']
    precedents = search_quote_history(historical_terms[:1] or [name], limit=5)
    if quantity >= 500 and len(precedents) >= 3:
        discount += Decimal('0.01')
    # Quote as unit price so the customer-facing multiplication is exact.
    # This is an educational policy, not a supplier-contract margin guarantee.
    floor = supplier_unit * MIN_MARKUP
    sales_unit = max(floor, supplier_unit * BASE_MARKUP * (1 - discount))
    unit_price = money(sales_unit)
    return unit_price, money(unit_price * quantity), discount


# -------- Actual model-backed agents: exactly 5 --------------------------------
def build_agents() -> dict[str, Agent]:
    if not API_KEY:
        raise RuntimeError("Set UDACITY_OPENAI_API_KEY in .env before running")
    model = OpenAIChatModel(MODEL_NAME, provider=OpenAIProvider(base_url=PROXY_URL, api_key=API_KEY))
    orchestrator = Agent(
        model, name="orchestrator", output_type=RoutedRequest,
        instructions=(
            "Route customer text accurately. Extract EVERY distinct ordered item (never omit an item). Output: intent, paper "
            "items with integer quantities, an explicit latest acceptable delivery deadline "
            "when given, and a brief rationale. Quote inquiries are 'quote', not 'order'. "
            "Use 'order' only if the customer clearly confirms a purchase. Do not infer "
            "nonexistent quantities or dates. If uncertain, output unknown or empty items. "
            "Interpret 'need by X' as deadline. Preserve user quantities exactly. "
            "A ream contains 500 sheets; if customer specifies reams, multiply by 500 for paper products. "
            "A quote inquiry is not a confirmed purchase. Do not expose internal financial data."
        ),
    )
    inventory = Agent(model, name="inventory", instructions=(
        "You are the inventory analyst. Use your tools to verify stock and supplier lead times. "
        "Never guess stock levels. Provide a concise readiness assessment."
    ))
    quoting = Agent(model, name="quoting", instructions=(
        "You are the quotation specialist. Call the historical quote tool and quote-price "
        "tool when asked. Explain visible bulk discounts and totals; never reveal costs."
    ))
    sales = Agent(model, name="sales", instructions=(
        "You are the order fulfillment analyst. Use stock/ETA tools for a read-only "
        "recommendation. Only the Python controller can authorize recording sales or purchases. "
        "Do not claim an unconfirmed order has been placed."
    ))
    advisor = Agent(model, name="business_advisor", instructions=(
        "You are an internal operations advisor. Use financial tools for actionable stock and "
        "cash insights. These findings must never appear in customer-facing responses."
    ))

    @inventory.tool_plain
    def snapshot_inventory(as_of_date: str) -> dict[str, int]:
        """Retrieve all currently stocked item quantities as of an ISO date."""
        return get_all_inventory(iso_date(as_of_date))

    @inventory.tool_plain
    def inspect_inventory(item_name: str, as_of_date: str) -> dict:
        """Inspect one item's stock using the starter get_stock_level helper."""
        name = resolve_item(item_name)
        return {"item": name, "stock": on_hand(name, iso_date(as_of_date)) if name else None}

    @inventory.tool_plain
    def supplier_delivery(start_date: str, quantity: int) -> str:
        """Calculate incoming supplier goods arrival, not final customer delivery."""
        return get_supplier_delivery_date(iso_date(start_date), quantity)

    @quoting.tool_plain
    def historical_quotes(item_search: str) -> list[dict]:
        """Retrieve comparable historical customer quotes for pricing context."""
        terms = [w for w in re.findall(r"[a-zA-Z0-9]+", item_search) if len(w) > 3][:3]
        return search_quote_history(terms or [item_search], limit=5)

    @quoting.tool_plain
    def calculate_customer_quote(item_name: str, quantity: int) -> dict:
        """Compute total, unit price and bulk savings from the catalog policy."""
        name = resolve_item(item_name)
        if not name or quantity <= 0:
            return {"error": "Unrecognized product or nonpositive quantity"}
        unit_price, total, discount = price_line(name, quantity)
        return {"item": name, "quantity": quantity, "unit_price": str(unit_price),
                "total": str(total), "bulk_discount_pct": str(discount * 100)}

    @sales.tool_plain
    def check_order_feasibility(item_name: str, quantity: int, as_of_date: str,
                                deadline: str = "") -> dict:
        """Check stock and supplier timing before approving a customer order."""
        name = resolve_item(item_name)
        if not name or quantity <= 0:
            return {"feasible": False, "reason": "Unrecognized product or quantity"}
        stock = on_hand(name, iso_date(as_of_date))
        shortage = max(0, quantity - stock)
        eta = get_supplier_delivery_date(iso_date(as_of_date), shortage) if shortage else None
        return {"item": name, "stock": stock, "shortage": shortage,
                "supplier_eta": eta, "deadline_feasible": not deadline or not eta or eta <= deadline}

    @sales.tool_plain
    def record_transaction(item_name: str, transaction_type: str, quantity: int,
                           total_price: float, transaction_date: str) -> dict:
        """Record a validated sale or stock purchase in the starter transaction ledger.

        IMPORTANT: The controller invokes this only after deterministic checks;
        no model is given authority to invoke it during ordinary agent runs.
        """
        if not _WRITE_ALLOWED.get():
            raise PermissionError("Only validated Python controller may post ledger transactions")
        if transaction_type not in {"sales", "stock_orders"} or quantity <= 0:
            raise ValueError("Invalid transaction type or quantity")
        if resolve_item(item_name) != item_name:
            raise ValueError("Exact catalog item name required")
        transaction_id = create_transaction(item_name, transaction_type, quantity,
                                            total_price, iso_date(transaction_date))
        return {"transaction_id": transaction_id, "status": "recorded"}

    @advisor.tool_plain
    def read_cash_balance(as_of_date: str) -> float:
        """Read internal cash balance from the starter's ledger helper."""
        return float(get_cash_balance(iso_date(as_of_date)))

    @advisor.tool_plain
    def financial_overview(as_of_date: str) -> dict:
        """Generate internal financial and inventory performance report."""
        return generate_financial_report(iso_date(as_of_date))

    return {"orchestrator": orchestrator, "inventory": inventory,
            "quoting": quoting, "sales": sales, "advisor": advisor,
            "validated_ledger_tool": record_transaction}


def one_agent_response(agent: Agent, prompt: str) -> str:
    """Non-critical LLM narration; validation never depends on narrative text."""
    if not WORKER_NARRATION:
        return ""
    try:
        return str(agent.run_sync(prompt).output)
    except Exception as exc:
        print(f"[internal] Optional agent narration unavailable: {type(exc).__name__}")
        return ""


def build_line(item: RequestedItem, request_date: str, deadline: str | None) -> PlannedLine | str:
    name = resolve_item(item.description)
    if not name:
        return f"Product '{item.description}' is not an unambiguous match in our catalog."
    if item.quantity <= 0:
        return "Requested quantity must be a positive integer."
    unit_price, total, discount = price_line(name, item.quantity)
    stock = on_hand(name, request_date)
    shortage = max(0, item.quantity - stock)
    eta = get_supplier_delivery_date(request_date, shortage) if shortage else None
    status = "ready" if not shortage else "shortage"
    if deadline and eta and eta > deadline:
        status = "deadline_failed"
    return PlannedLine(name, item.quantity, unit_price, total, discount, stock,
                       eta, status, money(catalog()[name]["unit_price"] * shortage))


def execute_order(lines: list[PlannedLine], request_date: str,
                  deadline: str | None, accepted: bool, ledger_tool) -> tuple[str, str, str]:
    """Return (status, reason, fulfillment_date) with fail-closed checks."""
    if not accepted:
        return "quoted", "A quote was provided; customer purchase acceptance is required.", ""
    if deadline and deadline < request_date:
        return "rejected", "Requested deadline is before the inquiry date.", ""
    for line in lines:
        if line.status == "deadline_failed":
            return "rejected", (f"{line.item_name}: supplier arrival {line.supplier_eta} "
                                f"misses the requested deadline {deadline}."), ""
    # No speculative future-sales entries: reorders are separate from fulfillment.
    for line in lines:
        if line.status == "shortage":
            return "unfulfilled", (f"{line.item_name}: {line.quantity} requested, "
                                   f"{line.in_stock} currently available. Supplier arrival is "
                                   f"estimated for {line.supplier_eta}; contact sales to arrange "
                                   "a backorder after availability is confirmed."), ""
    # Single-threaded test guard: re-read right before posting any sale.
    for line in lines:
        if on_hand(line.item_name, request_date) < line.quantity:
            return "unfulfilled", f"{line.item_name}: insufficient stock at final validation.", ""
    # All lines have passed checks; updates are for this educational sequential harness.
    # Production: use one atomic SQL transaction plus uniqueness/idempotency keys.
    try:
        token = _WRITE_ALLOWED.set(True)
        try:
            for line in lines:
                ledger_tool(line.item_name, "sales", line.quantity,
                            float(line.total), request_date)
        finally:
            _WRITE_ALLOWED.reset(token)
    except Exception as exc:
        print(f"[internal] Transaction failure: {type(exc).__name__}")
        return "error", "We could not confirm this order. Please contact sales.", ""
    return "fulfilled", "The requested items were available and the order was recorded.", request_date


def cash_safe_restock(as_of_date: str, ledger_tool, min_reorder: int = 25) -> list[dict]:
    """Replenish items under threshold if cash supports the purchase.

    Designed for explicit invocation by the advisor under --auto-reorder.
    Supplier arrival is a future stock-order date; no goods are considered on hand early.
    """
    import pandas as pd
    inventory = pd.read_sql("SELECT item_name, min_stock_level, unit_price FROM inventory", db_engine)
    actions = []
    for row in inventory.itertuples(index=False):
        name = str(row.item_name)
        stock = on_hand(name, as_of_date)
        target = max(int(row.min_stock_level) * 2, int(row.min_stock_level) + min_reorder)
        if stock >= int(row.min_stock_level):
            continue
        qty = target - stock
        expense = money(qty * Decimal(str(row.unit_price)))
        if get_cash_balance(as_of_date) < float(expense):
            actions.append({"item": name, "status": "recommended_not_ordered", "reason": "Insufficient funds"})
            continue
        eta = get_supplier_delivery_date(as_of_date, qty)
        # Guard against duplicate order for the same item/arrival date.
        existing = pd.read_sql(
            "SELECT COUNT(*) AS n FROM transactions WHERE item_name = :item AND "
            "transaction_type = 'stock_orders' AND transaction_date >= :start "
            "AND transaction_date <= :arrival",
            db_engine, params={"item": name, "start": as_of_date, "arrival": eta})
        if int(existing.iloc[0]["n"]) > 0:
            actions.append({"item": name, "status": "already_pending"})
            continue
        try:
            token = _WRITE_ALLOWED.set(True)
            try:
                ledger_tool(name, "stock_orders", qty, float(expense), eta)
            finally:
                _WRITE_ALLOWED.reset(token)
            actions.append({"item": name, "status": "ordered", "quantity": qty,
                            "supplier_arrival": eta})
        except Exception as exc:
            actions.append({"item": name, "status": "not_ordered", "reason": type(exc).__name__})
    return actions


def process_request(agents: dict[str, Agent], customer_text: str, request_date: str,
                    customer_accepts_quote: bool = False) -> dict:
    """Five-agent handoff: route -> inventory -> quoting -> sales -> advisor periodically."""
    request_date = iso_date(request_date)
    cash_before = float(get_cash_balance(request_date))
    record = {"request_date": request_date, "request": customer_text,
              "status": "unfulfilled", "reason": "", "quote_total": "0.00",
              "cash_before": cash_before, "cash_after": cash_before,
              "cash_change": 0.0, "fulfillment_date": "", "customer_response": ""}
    try:
        route = agents["orchestrator"].run_sync(
            f"Customer request date: {request_date}. Catalog product names: "
            + ", ".join(catalog()) + f". Customer message: {customer_text}").output
    except Exception as exc:
        record.update(status="error", reason="We need a representative to clarify your request.")
        record["customer_response"] = record["reason"]
        print(f"[internal] Routing failed: {type(exc).__name__}")
        return record
    record["intent"] = route.intent
    try:
        deadline = iso_date(route.deadline) if route.deadline else None
    except ValueError:
        deadline = None  # Invalid deadline means do not confirm a timed order.
        record.update(status="clarification", reason="Please confirm your requested delivery date.")
        record["customer_response"] = record["reason"]
        return record

    if route.intent == "report":
        # Internal only: no financial details are included in the customer response.
        record.update(status="internal_only", reason="Internal report access is restricted.")
        record["customer_response"] = record["reason"]
        return record
    if not route.items or route.intent == "unknown":
        record.update(status="clarification", reason="Please provide a product and quantity.")
        record["customer_response"] = record["reason"]
        return record
    planned: list[PlannedLine] = []
    for item in route.items:
        outcome = build_line(item, request_date, deadline)
        if isinstance(outcome, str):
            record.update(status="clarification", reason=outcome)
            record["customer_response"] = outcome
            return record
        planned.append(outcome)

    # Merge repeated catalog SKUs to prevent double allocation and apply tiered pricing.
    grouped = {}
    for line in planned:
        grouped[line.item_name] = grouped.get(line.item_name, 0) + line.quantity
    if len(grouped) != len(planned):
        planned = [build_line(RequestedItem(description=name, quantity=qty), request_date, deadline)
                   for name, qty in grouped.items()]

    # Each worker is actually invoked; their advice cannot bypass Python checks.
    facts = [{"item": p.item_name, "quantity": p.quantity, "in_stock": p.in_stock,
              "supplier_eta": p.supplier_eta} for p in planned]
    one_agent_response(agents["inventory"],
                       "Check current stock using your tools for these facts: " +
                       json.dumps({"date": request_date, "items": facts}))
    historical = {}
    for line in planned:
        try:
            historical[line.item_name] = search_quote_history([line.item_name], limit=3)
        except Exception:
            historical[line.item_name] = []
    one_agent_response(agents["quoting"],
                       "Use historical quote tool to explain the externally visible pricing. " +
                       json.dumps({"items": [p.item_name for p in planned],
                                   "historical_examples": {k: len(v) for k, v in historical.items()}}))

    total = money(sum((line.total for line in planned), Decimal("0")))
    record["quote_total"] = str(total)
    details = "; ".join(f"{p.quantity} x {p.item_name} @ ${p.unit_price} = ${p.total} "
                        f"({int(p.discount * 100)}% volume/history-based discount policy)" for p in planned)
    if route.intent == "inventory":
        status = "answered"
        reason = "; ".join(f"{p.item_name}: {p.in_stock} units currently available" for p in planned)
        fulfillment_date = ""
    elif route.intent == "quote" and not customer_accepts_quote:
        status, reason, fulfillment_date = execute_order(planned, request_date, deadline, False, agents["validated_ledger_tool"])
    else:
        # A sample harness may deliberately simulate quote acceptance for evaluation.
        confirmed = route.intent == "order" or customer_accepts_quote
        status, reason, fulfillment_date = execute_order(planned, request_date, deadline, confirmed, agents["validated_ledger_tool"])
    one_agent_response(agents["sales"],
                       "Review this read-only fulfillment result; do not call write tools: " +
                       json.dumps({"status": status, "reason": reason, "date": request_date}))
    response = f"{details}. Quoted total: ${total}. {reason}"
    if deadline:
        response += f" Requested deadline: {deadline}."
    record.update(status=status, reason=reason, fulfillment_date=fulfillment_date,
                  customer_response=response, cash_after=float(get_cash_balance(request_date)))
    record["cash_change"] = round(record["cash_after"] - cash_before, 2)
    return record


def run_test_scenarios(agents: dict[str, Agent], sample_path: str = "quote_requests_sample.csv",
                       simulate_acceptance: bool = False, auto_reorder: bool = False) -> list[dict]:
    """Run every CSV row, preserve original row identity, and export true test results."""
    with open(sample_path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No sample requests supplied")
    # Process in chronological order to keep ledger-as-of-date comparisons coherent.
    ordered = sorted(enumerate(rows), key=lambda pair: (iso_date(pair[1]["request_date"]), pair[0]))
    results = []
    for index, row in ordered:
        print(f"\n[{len(results)+1}/{len(rows)}] Orchestrator -> Inventory -> Quoting -> Sales")
        case = process_request(agents, str(row.get("request", "")), row["request_date"],
                               customer_accepts_quote=simulate_acceptance)
        case.update({"original_row": index + 1, "job": row.get("job", ""),
                     "need_size": row.get("need_size", ""), "event": row.get("event", "")})
        results.append(case)
        print(f"  {case['status'].upper()}: {case['reason']}")
        if auto_reorder and (len(results) % REPORT_EVERY == 0):
            restocks = cash_safe_restock(case["request_date"], agents["validated_ledger_tool"])
            print(f"  [internal advisor] Reorder review: {len(restocks)} actions")
        if len(results) % REPORT_EVERY == 0:
            one_agent_response(agents["advisor"],
                               f"Run your private cash and financial tools as of {case['request_date']}; "
                               "offer 2 stock recommendations, internal only.")
    fields = ["original_row", "job", "need_size", "event", "request_date", "request",
              "intent", "status", "reason", "quote_total", "cash_before",
              "cash_after", "cash_change", "fulfillment_date", "customer_response"]
    # Stable output order matches input data, even if processing was date-sorted.
    with open("test_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(results, key=lambda r: r["original_row"]))
    successes = sum(r["status"] == "fulfilled" for r in results)
    movements = sum(abs(r["cash_change"]) > 0 for r in results)
    declines = sum(r["status"] in {"unfulfilled", "rejected", "clarification"} for r in results)
    print(f"\nSummary: {len(results)} rows | fulfilled={successes} | "
          f"cash movements={movements} | not fulfilled/rejected={declines}")
    print("Rubric: at least 3 fulfilled, at least 3 cash movements, at least 1 rejected/unfulfilled")
    if successes < 3 or movements < 3 or declines < 1:
        print("WARNING: Rubric thresholds not achieved. Inspect generated test_results.csv.")
    print("Saved test_results.csv")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Munder Difflin agentic sales evaluation")
    parser.add_argument("--csv", default="quote_requests_sample.csv")
    parser.add_argument("--simulate-acceptance", action="store_true",
                        help="TEST ONLY: simulate customer accepting quoted prices")
    parser.add_argument("--auto-reorder", action="store_true",
                        help="TEST ONLY: permit finance-checked replenishment")
    parser.add_argument("--skip-worker-narration", action="store_true",
                        help="Speed up local diagnostic runs; full submission evaluation should invoke workers")
    args = parser.parse_args()
    WORKER_NARRATION = not args.skip_worker_narration
    # The starter's init_database resets seed data. The harness is reproducible.
    # Run only against a disposable or project database; it is NOT production-safe.
    init_database(db_engine)
    team = build_agents()
    run_test_scenarios(team, args.csv, args.simulate_acceptance, args.auto_reorder)
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        final_date = max(r["request_date"] for r in csv.DictReader(f))
    print("Final internal financial report:")
    print(json.dumps(generate_financial_report(final_date), default=str, indent=2))
