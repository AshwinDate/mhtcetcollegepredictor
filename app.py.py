from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Flask is working!"

if __name__ == "__main__":
    app.run(debug=True)



# app.py
from flask import Flask, render_template, request
import pandas as pd
import math

app = Flask(__name__)

# === CONFIG ===
DATA_CSV = "colleges.csv"      # update path if you place data elsewhere
TOP_N = 10                     # how many results to show

# === Helper functions ===

def load_data():
    """Load dataset from CSV into pandas DataFrame."""
    df = pd.read_csv(DATA_CSV)
    # ensure columns exist
    required = {"college_id","college_name","branch","category","cutoff_percentile_2024"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset missing required columns: {required - set(df.columns)}")
    # fill missing package years with 0
    for y in ["avg_package_2024","avg_package_2023","avg_package_2022","avg_package_2021","avg_package_2020"]:
        if y not in df.columns:
            df[y] = 0.0
    return df

def is_eligible(row, user_percentile, user_category, user_branch):
    """Simple eligibility: user_percentile >= cutoff_percentile_2024 and branch & category match."""
    try:
        if user_branch.strip().lower() not in row['branch'].strip().lower():
            return False
    except Exception:
        return False
    # category check: row category could be 'GENERAL' or 'ALL' etc.
    row_cat = str(row.get('category','ALL')).upper()
    if row_cat not in ("ALL", user_category.upper(), "GENERAL"):
        # allow GENERAL as open; also allow ALL
        # If real data has category-specific cutoffs, store multiple rows per category.
        pass
    # numeric check
    cutoff = float(row.get('cutoff_percentile_2024', 0.0))
    return user_percentile >= cutoff

def score_college(row):
    """
    Score a college based on:
     - historical average package (5-year average)
     - recent trend (slope) across years
    Returns a numeric score (higher = better).
    """
    years = ["avg_package_2024","avg_package_2023","avg_package_2022","avg_package_2021","avg_package_2020"]
    values = []
    for y in years:
        try:
            values.append(float(row.get(y,0.0)))
        except:
            values.append(0.0)
    # 1) average package
    avg = sum(values) / (len(values) if len(values) else 1)
    # 2) trend: simple linear slope estimate (least-squares fit)
    # x = 0..4, y = values
    n = len(values)
    x = list(range(n))
    x_mean = sum(x)/n
    y_mean = sum(values)/n
    num = sum((x[i]-x_mean)*(values[i]-y_mean) for i in range(n))
    den = sum((x[i]-x_mean)**2 for i in range(n))
    slope = num/den if den != 0 else 0.0
    # final score: weighted (avg * 0.8) + (slope * 2) to give trending colleges some boost
    score = avg * 0.8 + slope * 2.0
    return score, avg, slope

@app.route("/", methods=["GET", "POST"])
def index():
    df = load_data()
    results = []
    form = {"percentile": "", "category": "", "branch": ""}
    if request.method == "POST":
        # read form
        try:
            percentile = float(request.form.get("percentile","0"))
        except:
            percentile = 0.0
        category = request.form.get("category","GENERAL").upper()
        branch = request.form.get("branch","").strip()
        form = {"percentile": percentile, "category": category, "branch": branch}
        # filter and score
        eligible = []
        for _, row in df.iterrows():
            if is_eligible(row, percentile, category, branch):
                sc, avg, slope = score_college(row)
                eligible.append({
                    "college_id": row.get("college_id"),
                    "college_name": row.get("college_name"),
                    "branch": row.get("branch"),
                    "cutoff_percentile_2024": row.get("cutoff_percentile_2024"),
                    "avg_5yr": round(avg,2),
                    "trend_slope": round(slope,3),
                    "score": round(sc,3)
                })
        # rank by score and then avg_5yr and then cutoff (lower cutoff preferred)
        eligible_sorted = sorted(eligible, key=lambda r: (-r['score'], -r['avg_5yr'], r['cutoff_percentile_2024']))
        results = eligible_sorted[:TOP_N]
    return render_template("index.html", results=results, form=form)

if __name__ == "__main__":
    app.run(debug=True)
