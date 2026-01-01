from flask import Flask, render_template, request
import csv
from collections import defaultdict

app = Flask(__name__)

# =========================
# COLLEGE PRIORITY (LOWER = BETTER)
# =========================
COLLEGE_RANKING = {
    "COEP Tech Pune": 1,
    "VJTI Mumbai": 2,
    "Walchand COE Sangli": 3,
    "SPCE Mumbai": 4,
    "PICT Pune": 5,
    "VIT Pune": 6,
    "PCCOE Pune": 7,
    "DJ Sanghvi": 8,
    "PVG COET Pune": 9,
    "MIT Pune": 10,
    "AISSMS COE Pune": 11
}

# =========================
# MHT-CET RANK → PERCENTILE (APPROX)
# =========================
def rank_to_percentile(rank):
    if rank <= 100: return 99.95
    elif rank <= 300: return 99.85
    elif rank <= 500: return 99.75
    elif rank <= 1000: return 99.50
    elif rank <= 2000: return 99.00
    elif rank <= 4000: return 98.00
    elif rank <= 6000: return 97.00
    elif rank <= 8000: return 95.50
    elif rank <= 12000: return 93.00
    elif rank <= 20000: return 90.00
    elif rank <= 30000: return 87.00
    else: return 82.00

# =========================
# LOAD CSV DATA
# =========================
def load_colleges():
    data = []
    with open("colleges.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["cutoff"] = float(row["cutoff"])
                data.append(row)
            except:
                continue
    return data

# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    summary = ""

    if request.method == "POST":
        rank = int(request.form["rank"])
        category = request.form.get("category") or "GENERAL"
        seat_type = request.form.get("seat_type", "HU")
        ladies = request.form.get("ladies", "NO")  # YES / NO

        your_percentile = rank_to_percentile(rank)
        rows = load_colleges()

        grouped = defaultdict(lambda: {
            "college": "",
            "safe": [],
            "moderate": [],
            "ambitious": []
        })

        for r in rows:
            # Exam filter
            if r["exam"] != "MHTCET":
                continue

            # Category filter
            if r["category"] != category:
                continue

            # University / Ladies quota logic
            if ladies == "YES":
                if r["quota"] != "LADIES":
                    continue
            else:
                if r["quota"] != seat_type:
                    continue

            margin = round(your_percentile - r["cutoff"], 2)

            # CHANCE LOGIC
            if margin >= 1.0:
                chance = "safe"
            elif -4.0 <= margin < 1.0:
                chance = "moderate"
            else:
                chance = "ambitious"

            college = r["college"]
            grouped[college]["college"] = college
            grouped[college][chance].append(r["branch"])

        # Convert to list & sort by college ranking
        results = sorted(
            grouped.values(),
            key=lambda x: COLLEGE_RANKING.get(x["college"], 999)
        )

        summary = (
            f"Rank {rank} ≈ {your_percentile}% | "
            f"{len(results)} colleges matched"
        )

    return render_template("index.html", results=results, summary=summary)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
