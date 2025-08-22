# ai/ai_assist.py (improved robust version)
import os
import sys
import json
import re
import joblib
import traceback
from dotenv import load_dotenv
from datetime import datetime
from google import genai
from google.genai import types

# If running from ai/ subfolder directly, ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import your model helper (adjust path/name if you renamed it)
try:
    from model.train_model import predict_from_dict
except Exception:
    # fallback: try different import name if you have train_model.py
    try:
        from model.train_model import predict_from_dict
    except Exception as e:
        # we'll still continue, but predictions will use a local wrapper if needed
        predict_from_dict = None

# -----------------------
# Load env and initialize
# -----------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")

if not GOOGLE_API_KEY:
    raise RuntimeError("Please set GOOGLE_API_KEY in your .env (project root).")

client = genai.Client(api_key=GOOGLE_API_KEY)

# -----------------------
# Load model artifacts
# -----------------------
MODEL_DIR = os.path.join(ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "flight_price_model.pkl")
ENC_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
ROUTE_MED_PATH = os.path.join(MODEL_DIR, "route_medians.json")
ROUTE_AVG_PATH = os.path.join(MODEL_DIR, "route_avg_prices.json")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENC_PATH) if os.path.exists(ENC_PATH) else {}

route_medians = json.load(open(ROUTE_MED_PATH)) if os.path.exists(ROUTE_MED_PATH) else {}
route_avg_prices = json.load(open(ROUTE_AVG_PATH)) if os.path.exists(ROUTE_AVG_PATH) else {}

# -----------------------
# Helpers: parsing & normalize
# -----------------------
def extract_text_from_genai_response(resp):
    """
    Safely extract concatenated text parts from genai response.candidates[0].content.parts
    while ignoring non-text parts.
    """
    try:
        candidate = resp.candidates[0]
        parts = getattr(candidate, "content", None)
        # candidate.content may be a list of parts; iterate and take 'text' fields
        if parts and isinstance(parts, list):
            texts = []
            for p in parts:
                if hasattr(p, "text") and p.text:
                    texts.append(p.text)
                else:
                    # some SDK returns dict-like parts
                    try:
                        txt = p.get("text")
                        if txt:
                            texts.append(txt)
                    except Exception:
                        pass
            return "\n".join(texts).strip()
        # fallback older property
        return getattr(candidate, "output_text", getattr(candidate, "text", str(candidate)))
    except Exception:
        # very fallback
        try:
            return getattr(resp, "text", str(resp))
        except Exception:
            return ""

def extract_json_from_text(s):
    """
    Try to find a JSON object in a text string and parse it.
    Returns dict or None.
    """
    if not s or not isinstance(s, str):
        return None
    # try direct parse
    s_strip = s.strip()
    try:
        return json.loads(s_strip)
    except Exception:
        pass
    # try to locate first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            # sometimes model outputs trailing commas: try to fix common issues
            cleaned = re.sub(r",\s*}", "}", candidate)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            try:
                return json.loads(cleaned)
            except Exception:
                return None
    return None

def simple_fallback_extractor(text):
    """
    Very small heuristic extractor: find 'from X to Y', times, duration, airline keywords.
    Returns a dict with keys possibly present.
    """
    out = {"airline": None, "from": None, "to": None, "class": None, "stops": None,
           "dep_block": None, "duration": None}
    if not text:
        return out
    txt = text.lower()
    m = re.search(r'from\s+([a-z ]+?)\s+(?:to|->)\s+([a-z ]+)', txt)
    if m:
        out["from"] = m.group(1).strip().title()
        out["to"] = m.group(2).strip().title()
    # airline simple
    for a in ["indigo","vistara","spicejet","air india","goair","air asia","airasia","go first","air india express"]:
        if a in txt:
            out["airline"] = a.title()
            break
    # duration minutes
    m2 = re.search(r'(\d+)\s*(?:min|minutes)', txt)
    if m2:
        out["duration"] = int(m2.group(1))
    else:
        m3 = re.search(r'(\d+)\s*h(?:ours)?', txt)
        if m3:
            out["duration"] = int(m3.group(1))*60
    # dep/arr blocks
    for blk in ["morning","afternoon","evening","night"]:
        if blk in txt and not out["dep_block"]:
            out["dep_block"] = blk.title()
    if "non" in txt:
        out["stops"] = "non-stop"
    return out

def normalize_city_name(s):
    if not s:
        return None
    return re.sub(r'\s+', ' ', str(s).strip()).title()

def make_route_key(origin, dest):
    if not origin or not dest:
        return None
    return f"{origin.strip().lower()}->{dest.strip().lower()}"

# -----------------------
# Derived features & defaults
# -----------------------
def estimate_arrival_block(dep_block, duration_min):
    block_hours = {"Morning": 8, "Afternoon": 14, "Evening": 19, "Night": 23}
    hour = block_hours.get(dep_block, 8)
    arrival_time = (hour * 60 + (duration_min or 0)) % (24 * 60)
    arr_hour = arrival_time // 60
    if 5 <= arr_hour < 12:
        return "Morning"
    if 12 <= arr_hour < 17:
        return "Afternoon"
    if 17 <= arr_hour < 21:
        return "Evening"
    return "Night"

def get_default_duration(origin, dest):
    key = make_route_key(origin, dest)
    if key and key in route_medians:
        try:
            return int(route_medians[key]["median"])
        except Exception:
            pass
    return 120

def auto_fill_features(parsed):
    # parsed may have keys or be None
    parsed = parsed or {}
    p = {k: parsed.get(k) for k in ["airline","from","to","class","stops","dep_block","arr_block","duration"]}
    # normalize city names
    p["from"] = normalize_city_name(p.get("from"))
    p["to"] = normalize_city_name(p.get("to"))

    # default duration from route medians if missing
    if not p.get("duration"):
        p["duration"] = get_default_duration(p.get("from"), p.get("to"))

    # arrival block from dep_block + duration if arr_block missing
    if not p.get("arr_block"):
        p["arr_block"] = estimate_arrival_block(p.get("dep_block") or "Morning", int(p["duration"] or 120))

    # day_of_week / is_weekend default to today
    dow = datetime.today().weekday()
    p["day_of_week"] = dow
    p["is_weekend"] = 1 if dow in (5,6) else 0

    # categorical defaults
    for c in ["airline","class","stops","dep_block"]:
        if not p.get(c):
            p[c] = "__missing__"

    return p

# -----------------------
# Model prediction wrapper
# -----------------------
def safe_predict(pfeatures):
    """
    pfeatures: dict with fields expected by predict_from_dict
    returns: float predicted price or None
    """
    try:
        # if your predict_from_dict expects (model, encoders, input_dict)
        if predict_from_dict:
            return float(predict_from_dict(model, label_encoders, pfeatures))
        # fallback: try to run model directly with same encoding logic
        # (simple implementation assuming label_encoders are LabelEncoder objects)
        row = {}
        for c, le in label_encoders.items():
            val = pfeatures.get(c, "__missing__")
            try:
                code = int(le.transform([str(val)])[0])
            except Exception:
                try:
                    code = int(le.transform(["__missing__"])[0])
                except Exception:
                    code = 0
            row[c] = code
        # numeric
        row['duration'] = int(pfeatures.get('duration', 0))
        row['day_of_week'] = int(pfeatures.get('day_of_week', 0))
        row['is_weekend'] = int(pfeatures.get('is_weekend', 0))
        feature_order = ['airline','from','to','class','stops','dep_block','arr_block','duration','day_of_week','is_weekend']
        X = [row.get(f, 0) for f in feature_order]
        pred = float(model.predict([X])[0])
        return pred
    except Exception:
        print("Prediction failed:", traceback.format_exc())
        return None

def booking_advice(predicted_price, origin, dest):
    key = make_route_key(origin, dest)
    avg = route_avg_prices.get(key) if route_avg_prices else None
    if avg:
        diff = (predicted_price - avg) / avg * 100
        if diff <= -10:
            return "BUY — price well below historical average."
        if diff >= 10:
            return "WAIT — price above average, consider monitoring."
        return "NEUTRAL — price near historical average."
    return "No historical average for this route; consider your budget and flexibility."

# -----------------------
# Main loop
# -----------------------
print("✈️ Personalized Flight Planner — type 'quit' to exit")
while True:
    try:
        user = input("\nYou: ").strip()
        if user.lower() in ("quit","exit"):
            break

        # Ask GenAI to extract JSON (minimal prompt). We'll robustly pull text parts.
        extraction_prompt = (
            "Extract these fields as JSON: airline, from, to, class, stops, dep_block, arr_block, duration (minutes). "
            "If any value is unknown, use null. Return only a JSON object."
            f"\nUser: {user}"
        )
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512
        )

        resp = client.models.generate_content(
            model=LLM_MODEL,
            contents=extraction_prompt,
            config=config
        )

        text = extract_text_from_genai_response(resp)
        parsed = extract_json_from_text(text)
        if parsed is None:
            # fallback to heuristic extractor if JSON parse fails
            parsed = simple_fallback_extractor(user)

        filled = auto_fill_features(parsed)
        pred = safe_predict(filled)
        if pred is None:
            print("⚠️ Could not produce a prediction.")
            continue

        advice = booking_advice(pred, filled.get("from"), filled.get("to"))

        # Print results and assumptions
        print("\n--- Flight Plan & Assumptions ---")
        for k in ("from","to","airline","class","stops","dep_block","arr_block","duration"):
            print(f"{k}: {filled.get(k)}")
        print(f"\n💰 Predicted price: ₹{pred:,.0f}")
        print(f"Recommendation: {advice}")
        # show route averages if available
        rk = make_route_key(filled.get("from"), filled.get("to"))
        if rk and route_avg_prices.get(rk):
            print(f"(Historical avg: ₹{route_avg_prices[rk]:,.0f})")
        print("-------------------------------")

    except KeyboardInterrupt:
        print("\nExiting.")
        break
    except Exception:
        print("Unhandled error:", traceback.format_exc())
        break
