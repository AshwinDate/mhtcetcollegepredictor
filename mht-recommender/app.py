from flask import Flask, render_template, request
import csv

app = Flask(__name__)

# =========================
# COLLEGE PRIORITY (USER DEFINED)
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
    "AISSMS COE Pune": 11,
    "KJ Somaiya": 12
}

# =========================
# FIXED BRANCH ORDER
# =========================
BRANCH_ORDER = {
    "Computer Science": 1,
    "Information Technology": 2,
    "Electronics & Telecommunication": 3,
    "Mechanical Engineering": 4,
    "Instrumentation Engineering": 5,
    "Civil Engineering": 6
}

# =========================
# RANK → PERCENTILE (APPROX)
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
    elif rank <= 30000: return 87.00
    else: return 82.00

# =========================
# LOAD CSV
# =========================
def load_colleges():
    colleges = []
    with open("colleges.csv", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                row["cutoff"] = float(row["cutoff"])
                colleges.append(row)
            except:
                continue
    return colleges

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
        colleges = load_colleges()

        for c in colleges:
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

                results.append({
                    "college": c["college"],
                    "branch": c["branch"],
                    "category": c["category"],
                    "seat_type": c["seat_type"],
                    "percentile": your_percentile,
                    "margin": margin,
                    "chance": chance
                })

        # =========================
        # SORTING
        # =========================
        if sort_by == "college":
            results.sort(
                key=lambda x: (
                    COLLEGE_RANKING.get(x["college"], 999),
                    -x["margin"]
                )
            )

        elif sort_by == "branch":
            results.sort(
                key=lambda x: (
                    BRANCH_ORDER.get(x["branch"], 999),
                    COLLEGE_RANKING.get(x["college"], 999),
                    -x["margin"]
                )
            )

        elif sort_by == "chance":
            chance_priority = {"Safe": 1, "Moderate": 2, "Ambitious": 3}
            results.sort(
                key=lambda x: (
                    chance_priority[x["chance"]],
                    COLLEGE_RANKING.get(x["college"], 999),
                    -x["margin"]
                )
            )

        summary = f"{len(results)} colleges recommended (Rank {rank} ≈ {your_percentile}%)"

    return render_template("index.html", results=results, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
