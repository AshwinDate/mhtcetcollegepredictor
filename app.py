from flask import Flask, render_template, request
import csv
from collections import defaultdict

app = Flask(__name__)

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
    "AISSMS COE Pune": 11,
}

def rank_to_percentile(rank):
    if rank <= 100: return 99.95
    elif rank <= 500: return 99.75
    elif rank <= 1000: return 99.50
    elif rank <= 3000: return 98.00
    elif rank <= 5000: return 97.00
    elif rank <= 10000: return 95.00
    else: return 90.00

def load_colleges():
    data = []
    with open("colleges.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["cutoff"] = float(r["cutoff"])
            data.append(r)
    return data

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    summary = ""

    if request.method == "POST":
        rank = int(request.form["rank"])
        category = request.form.get("category") or "GENERAL"
        seat_type = request.form.get("seat_type", "HU")

        your_percentile = rank_to_percentile(rank)
        rows = load_colleges()

        grouped = defaultdict(lambda: {
            "college": "",
            "safe": [],
            "moderate": [],
            "ambitious": []
        })

        for r in rows:
            if r["category"] != category or r["seat_type"] != seat_type:
                continue

            if your_percentile >= r["cutoff"]:
                margin = round(your_percentile - r["cutoff"], 2)

                if margin >= 1.0:
                    chance = "safe"
                elif -4.0 <= margin < 1.0:
                    chance = "moderate"
                else:
                    chance = "ambitious"

                college = r["college"]
                grouped[college]["college"] = college
                grouped[college][chance].append(r["branch"])

        results = sorted(
            grouped.values(),
            key=lambda x: COLLEGE_RANKING.get(x["college"], 999)
        )

        summary = f"{len(results)} top colleges recommended (Rank {rank} ≈ {your_percentile}%)"

    return render_template("index.html", results=results, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
