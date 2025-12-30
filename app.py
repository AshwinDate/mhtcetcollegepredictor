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
# RANK → PERCENTILE
# =========================
def rank_to_percentile(rank):
    if rank <= 100: return 99.95
    elif rank <= 200: return 99.90
    elif rank <= 300: return 99.85
    elif rank <= 500: return 99.75
    elif rank <= 700: return 99.65
    elif rank <= 1000: return 99.50
    elif rank <= 2000: return 98.80
    elif rank <= 3000: return 98.00
    elif rank <= 5000: return 97.00
    elif rank <= 8000: return 95.50
    elif rank <= 12000: return 93.00
    elif rank <= 20000: return 90.00
    else: return 85.00

# =========================
# LOAD CSV
# =========================
def load_colleges():
    data = []
    with open("colleges.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            try:
                r["cutoff"] = float(r["cutoff"])
                data.append(r)
            except:
                continue
    return data

# =========================
# MAIN ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    results = []
    summary = ""

    if request.method == "POST":
        rank = int(request.form["rank"])
        category = request.form.get("category") or "GENERAL"
        seat_type = request.form.get("seat_type", "HU")
        sort_by = request.form["sort_by"]

        your_percentile = rank_to_percentile(rank)
        rows = load_colleges()

        grouped = defaultdict(list)

        for c in rows:
            if c["category"] != category:
                continue
            if c["seat_type"] != seat_type:
                continue
            if your_percentile >= c["cutoff"]:
                margin = round(your_percentile - c["cutoff"], 2)

                if margin >= 5:
                    chance = "Safe"
                elif margin >= 1:
                    chance = "Moderate"
                else:
                    chance = "Ambitious"

                grouped[c["college"]].append({
                    "branch": c["branch"],
                    "margin": margin,
                    "chance": chance
                })

        # ===== Convert grouped → list
        for college, branches in grouped.items():
            best_margin = max(b["margin"] for b in branches)
            best_chance = sorted(
                [b["chance"] for b in branches],
                key=lambda x: {"Safe": 1, "Moderate": 2, "Ambitious": 3}[x]
            )[0]

            results.append({
                "college": college,
                "branches": sorted(set(b["branch"] for b in branches)),
                "chance": best_chance,
                "best_margin": best_margin,
                "rank": COLLEGE_RANKING.get(college, 999)
            })

        # ===== SORT
        if sort_by == "college":
            results.sort(key=lambda x: (x["rank"], -x["best_margin"]))
        elif sort_by == "chance":
            results.sort(key=lambda x: (
                {"Safe": 1, "Moderate": 2, "Ambitious": 3}[x["chance"]],
                x["rank"]
            ))

        summary = f"{len(results)} colleges recommended (Rank {rank} ≈ {your_percentile}%)"

    return render_template("index.html", results=results, summary=summary)


if __name__ == "__main__":
    app.run(debug=True)
