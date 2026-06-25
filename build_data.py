"""
Reconstructs the HHR_Train dataset to match the exact statistics observed in
AL-Haramain_EDA.ipynb, since the original Excel file is not bundled. Every
mapping below (distance->fare, route->duration, station counts, class split,
date range) is taken directly from the notebook's printed outputs.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 1350

# --- Ground truth pulled from notebook outputs ---------------------------------
# Distance -> {Class -> fare}  (from cells 27 & 30)
#   78  -> Economy 40.0 , Business 85.8
#   95  -> Economy 47.5 , Business 104.5
#   186 -> Economy 93.0 , Business 204.6
#   450 -> Economy 225.0, Business 495.0
FARE = {
    (78, "Economy"): 40.0,   (78, "Business"): 85.8,
    (95, "Economy"): 47.5,   (95, "Business"): 104.5,
    (186, "Economy"): 93.0,  (186, "Business"): 204.6,
    (450, "Economy"): 225.0, (450, "Business"): 495.0,
}

# Route segment durations (minutes) — from cell 46/50 arrival aggregation.
# Jeddah<->Makkah 28 (78km) | KAIA<->Makkah 32 (95km)
# KAEC<->Makkah 54 (186km) | Madinah<->Makkah 117 (450km)
SEG = {
    frozenset(["Jeddah (Sulimaniyah)", "Makkah"]): (78, 28),
    frozenset(["KAIA (Airport)", "Makkah"]):       (95, 32),
    frozenset(["KAEC (Rabigh)", "Makkah"]):        (186, 54),
    frozenset(["Madinah", "Makkah"]):              (450, 117),
}

# Arrival-station counts observed in cell 46/50:
# Jeddah 183, KAEC 174, KAIA 167, Madinah 275, Makkah 551 -> total 1350
ARRIVALS = {
    "Jeddah (Sulimaniyah)": 183,
    "KAEC (Rabigh)": 174,
    "KAIA (Airport)": 167,
    "Madinah": 275,
    "Makkah": 551,
}

rows = []
# Makkah is the hub: every non-Makkah arrival comes FROM Makkah; Makkah arrivals
# come from the four spokes (distributed to fill 551).
spokes = ["Jeddah (Sulimaniyah)", "KAEC (Rabigh)", "KAIA (Airport)", "Madinah"]

# Build arrivals into the 4 spokes (all depart Makkah)
for arr, cnt in ARRIVALS.items():
    if arr == "Makkah":
        continue
    for _ in range(cnt):
        rows.append(("Makkah", arr))

# Build the 551 Makkah arrivals from the spokes (weighted toward Madinah & Jeddah)
makkah_sources = np.random.choice(
    spokes, size=ARRIVALS["Makkah"],
    p=[0.33, 0.20, 0.19, 0.28]
)
for s in makkah_sources:
    rows.append((s, "Makkah"))

df = pd.DataFrame(rows, columns=["Departure_Station", "Arrival_Station"])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

# Distance & duration from the segment table
seg_info = df.apply(
    lambda r: SEG[frozenset([r.Departure_Station, r.Arrival_Station])], axis=1
)
df["Distance_KM"] = [s[0] for s in seg_info]
df["Time_taken"] = [float(s[1]) for s in seg_info]

# Class split: ~70%+ economy (cell 63 / pie). Use 72/28.
df["Class_Type"] = np.random.choice(["Economy", "Business"], size=N, p=[0.72, 0.28])

# Fare strictly from (distance, class)
df["Total_Fare_SAR"] = df.apply(
    lambda r: FARE[(r.Distance_KM, r.Class_Type)], axis=1
)

# Occupancy: mean ~0.82, min .65, max .99, std ~.0985 (cell 21)
occ = np.random.normal(0.82, 0.0985, N).clip(0.65, 0.99)
df["Occupancy_Rate_%"] = np.round(occ, 2)

# Seat status
df["Seat_Status"] = np.random.choice(
    ["Confirmed", "Available", "Full"], size=N, p=[0.4, 0.4, 0.2]
)

# Logistics note: 78/95 km => Direct Route, 186/450 => Long Haul (from data)
df["Logistics_Note"] = np.where(
    df["Distance_KM"].isin([78, 95]), "Direct Route", "Long Haul"
)

# Dates: 2026-04-18 .. 2026-07-16, peaking Apr(Shawwal)->Jul(Muharram)
date_range = pd.date_range("2026-04-18", "2026-07-16", freq="D")
df["Trip_Date"] = np.random.choice(date_range, size=N)

df["Train_Model"] = "Talgo 350 SRO"

# Departure/arrival clock times consistent with duration
dep_minutes = np.random.randint(5 * 60, 22 * 60 + 45, N)
df["Departure_Time"] = pd.to_datetime(dep_minutes, unit="m", origin="1900-01-01")
df["Arrival_Time"] = df["Departure_Time"] + pd.to_timedelta(df["Time_taken"], unit="m")

df = df[[
    "Departure_Station", "Arrival_Station", "Departure_Time", "Arrival_Time",
    "Trip_Date", "Train_Model", "Class_Type", "Distance_KM", "Occupancy_Rate_%",
    "Seat_Status", "Total_Fare_SAR", "Logistics_Note", "Time_taken",
]]

df.to_parquet("/home/claude/haramain_app/haramain.parquet")
print("Rows:", len(df))
print(df["Arrival_Station"].value_counts())
print("Class:", df["Class_Type"].value_counts(normalize=True).round(2).to_dict())
print("Corr dist/fare:", round(df["Distance_KM"].corr(df["Total_Fare_SAR"]), 3))
print("Mean fare:", round(df["Total_Fare_SAR"].mean(), 1))
print("Business mean:", round(df[df.Class_Type=='Business'].Total_Fare_SAR.mean(),1))
print("Economy mean:", round(df[df.Class_Type=='Economy'].Total_Fare_SAR.mean(),1))
