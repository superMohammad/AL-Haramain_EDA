"""
Al-Haramain High-Speed Railway — Exploratory Data Analysis Dashboard
=====================================================================
A Streamlit presentation of the EDA originally written in AL-Haramain_EDA.ipynb.
All insights, questions, and conclusions are the author's own — preserved exactly.
Charts re-themed to a consistent light-green palette.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG + LIGHT-GREEN DESIGN SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Al-Haramain Railway · EDA",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light-green palette — used consistently across every chart and UI element.
GREEN = {
    "darkest": "#1B4332",   # headers, strong text
    "dark":    "#2D6A4F",   # primary accent
    "mid":     "#40916C",   # secondary
    "base":    "#52B788",   # main bars
    "light":   "#74C69D",   # tertiary
    "lighter": "#95D5B2",   # highlights
    "lightest":"#D8F3DC",   # backgrounds / fills
    "leaf":    "#B7E4C7",   # surfaces
}
# Ordered sequence for categorical charts (all green family)
GREEN_SEQ = ["#2D6A4F", "#52B788", "#95D5B2", "#74C69D", "#40916C", "#B7E4C7"]
# Continuous green scale
GREEN_SCALE = [[0, "#D8F3DC"], [0.5, "#52B788"], [1, "#1B4332"]]

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

    html, body, [class*="css"]  {{ font-family: 'Plus Jakarta Sans', sans-serif; }}

    .stApp {{ background: linear-gradient(180deg, #F4FBF6 0%, #EAF7EE 100%); }}

    /* Headings */
    h1, h2, h3 {{ color: {GREEN['darkest']}; font-weight: 800; letter-spacing:-0.5px; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {GREEN['darkest']};
    }}
    section[data-testid="stSidebar"] * {{ color: {GREEN['lightest']} !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ color: {GREEN['lightest']} !important; }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {GREEN['leaf']};
        border-left: 5px solid {GREEN['base']};
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 10px rgba(45,106,79,0.06);
    }}
    div[data-testid="stMetric"] label {{ color: {GREEN['mid']} !important; font-weight:600; }}
    div[data-testid="stMetricValue"] {{ color: {GREEN['darkest']} !important; }}

    /* Insight callout box */
    .insight {{
        background: #FFFFFF;
        border-left: 5px solid {GREEN['dark']};
        border-radius: 0 12px 12px 0;
        padding: 16px 22px;
        margin: 12px 0 24px 0;
        box-shadow: 0 2px 12px rgba(45,106,79,0.07);
        font-size: 1.02rem;
        line-height: 1.65;
        color: #25422F;
    }}
    .insight b {{ color: {GREEN['dark']}; }}
    .insight .tag {{
        display:inline-block; background:{GREEN['lightest']}; color:{GREEN['dark']};
        font-size:0.72rem; font-weight:700; letter-spacing:0.5px; text-transform:uppercase;
        padding:3px 10px; border-radius:20px; margin-bottom:8px;
    }}

    /* Explainer (neutral) box */
    .explain {{
        background: {GREEN['lightest']};
        border-radius: 12px;
        padding: 14px 20px;
        margin: 8px 0 20px 0;
        font-size: 0.97rem; line-height:1.6; color:#2C4A39;
    }}

    .step-pill {{
        display:inline-block; background:{GREEN['dark']}; color:#fff;
        border-radius:20px; padding:4px 14px; font-size:0.8rem; font-weight:700;
        margin-bottom:10px;
    }}
    hr {{ border-color:{GREEN['leaf']}; }}
    .stDataFrame {{ border-radius:12px; overflow:hidden; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    font=dict(family="Plus Jakarta Sans", color=GREEN["darkest"], size=13),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.55)",
    margin=dict(t=60, l=20, r=20, b=20),
    title=dict(font=dict(size=18, color=GREEN["darkest"])),
)

def style_fig(fig, ytitle=None, xtitle=None):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, title=xtitle, color=GREEN["darkest"])
    fig.update_yaxes(showgrid=True, gridcolor=GREEN["leaf"], title=ytitle, color=GREEN["darkest"])
    return fig

# ──────────────────────────────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(Path(__file__).parent / "haramain.parquet")
    df["Trip_Date"] = pd.to_datetime(df["Trip_Date"])
    return df

df_all = load_data()

MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']

def insight(text, tag=None):
    st.markdown(f'<div class="insight">💡 &nbsp;{text}</div>',
                unsafe_allow_html=True)

def explain(text):
    st.markdown(f'<div class="explain">{text}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SIDEBAR NAV
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚄 Al-Haramain EDA")
    st.markdown("High-Speed Railway · Saudi Arabia")
    st.markdown("---")
    page = st.radio(
        "Explore the analysis",
        [
            "🏠 Overview",
            "🧹 Data Preparation",
            "🎫 Class & Fares",
            "🔗 Distance ↔ Price",
            "🗺️ Stations & Routes",
            "📅 Seasonality",
            "⏱️ Trip Duration",
            "💺 Occupancy & Seats",
            "📌 Key Takeaways",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # ── Interactive filters ──────────────────────────────────────────────
    st.markdown("### 🎛️ Filters")
    st.caption("Slice the data — every chart updates live.")

    f_class = st.multiselect(
        "Travel class",
        options=sorted(df_all["Class_Type"].unique()),
        default=sorted(df_all["Class_Type"].unique()),
    )
    f_dep = st.multiselect(
        "Departure station",
        options=sorted(df_all["Departure_Station"].unique()),
        default=sorted(df_all["Departure_Station"].unique()),
    )
    f_arr = st.multiselect(
        "Arrival station",
        options=sorted(df_all["Arrival_Station"].unique()),
        default=sorted(df_all["Arrival_Station"].unique()),
    )
    f_logi = st.multiselect(
        "Trip type",
        options=sorted(df_all["Logistics_Note"].unique()),
        default=sorted(df_all["Logistics_Note"].unique()),
    )

    dmin, dmax = df_all["Trip_Date"].min().date(), df_all["Trip_Date"].max().date()
    f_dates = st.slider(
        "Trip date range",
        min_value=dmin, max_value=dmax, value=(dmin, dmax), format="MMM DD",
    )

    omin, omax = float(df_all["Occupancy_Rate_%"].min()), float(df_all["Occupancy_Rate_%"].max())
    f_occ = st.slider(
        "Occupancy rate", min_value=round(omin, 2), max_value=round(omax, 2),
        value=(round(omin, 2), round(omax, 2)), step=0.01,
    )

    if st.button("↺ Reset filters", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption(f"Source · {len(df_all):,} trips · 5 stations · Apr–Jul 2026")
    st.caption("Train model: Talgo 350 SRO")

# Apply filters → df is the live, filtered view used by all analytical pages.
mask = (
    df_all["Class_Type"].isin(f_class or df_all["Class_Type"].unique())
    & df_all["Departure_Station"].isin(f_dep or df_all["Departure_Station"].unique())
    & df_all["Arrival_Station"].isin(f_arr or df_all["Arrival_Station"].unique())
    & df_all["Logistics_Note"].isin(f_logi or df_all["Logistics_Note"].unique())
    & (df_all["Trip_Date"].dt.date >= f_dates[0])
    & (df_all["Trip_Date"].dt.date <= f_dates[1])
    & (df_all["Occupancy_Rate_%"] >= f_occ[0])
    & (df_all["Occupancy_Rate_%"] <= f_occ[1])
)
df = df_all[mask].copy()

# Guard: if filters exclude everything, warn and fall back to full data.
_filtered_empty = df.empty
if _filtered_empty:
    df = df_all.copy()

# Live filter-status banner (shown on every analytical page, not Data Prep).
def filter_banner():
    n, total = len(df), len(df_all)
    if _filtered_empty:
        st.warning("No trips match the current filters — showing the full dataset. "
                   "Loosen a filter in the sidebar to narrow it down.")
    elif n < total:
        pct = n / total * 100
        st.markdown(
            f'<div class="explain" style="margin-top:0;">'
            f'🎛️ <b>Live view:</b> showing <b>{n:,}</b> of {total:,} trips '
            f'({pct:.0f}%) based on your sidebar filters.</div>',
            unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("# Al-Haramain High-Speed Railway")
    filter_banner()
    st.markdown("### An exploratory data analysis of 1,350 train trips across the holy-city corridor")

    explain(
        "The <b>Al-Haramain High-Speed Railway (HHR)</b> connects Makkah and Madinah through "
        "Jeddah, King Abdullah Economic City (KAEC / Rabigh), and King Abdulaziz International "
        "Airport (KAIA). This dashboard walks through the full exploratory analysis — from raw "
        "data cleaning to the questions asked and the conclusions reached. Use the sidebar to "
        "move through each stage of the story."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trips", f"{len(df):,}")
    c2.metric("Stations Served", df["Departure_Station"].nunique())
    c3.metric("Avg Fare", f"{df['Total_Fare_SAR'].mean():.0f} SAR")
    c4.metric("Avg Occupancy", f"{df['Occupancy_Rate_%'].mean()*100:.0f}%")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date Range", "Apr 18 – Jul 16")
    c2.metric("Distance Range", "78 – 450 km")
    c3.metric("Economy Share", f"{(df['Class_Type']=='Economy').mean()*100:.0f}%")
    c4.metric("Busiest Station", "Makkah")

    st.markdown("---")
    st.markdown("### What this analysis answers")
    explain(
        "Rather than wandering through the data, the analysis was driven by a clear set of "
        "questions. Each section of this dashboard answers one of them:"
        "<ul>"
        "<li>What is the most frequent station, and what do the routes mean?</li>"
        "<li>Which route is used most — and are trips mostly direct or long-haul?</li>"
        "<li>How are fares related to distance and travel class?</li>"
        "<li>When do people travel, and why those months?</li>"
        "<li>How long does each route take, and are there real outliers?</li>"
        "<li>How full are the trains, and how are seats distributed?</li>"
        "</ul>"
    )

    # A small route map preview
    st.markdown("### The network at a glance")
    route_counts = (df.groupby(["Departure_Station","Arrival_Station"])
                      .size().reset_index(name="Trips")
                      .sort_values("Trips", ascending=True))
    route_counts["Route"] = route_counts["Departure_Station"] + " → " + route_counts["Arrival_Station"]
    fig = px.bar(route_counts.tail(12), x="Trips", y="Route", orientation="h",
                 color="Trips", color_continuous_scale=GREEN_SCALE)
    fig.update_layout(coloraxis_showscale=False, height=460,
                      title="Busiest origin → destination routes")
    style_fig(fig, xtitle="Number of Trips")
    st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🧹 Data Preparation":
    st.markdown("# 🧹 Data Preparation")
    explain(
        "Before any analysis, the raw data had to be cleaned and reshaped. Good decisions here "
        "make every later chart trustworthy. Here is exactly what was done and why."
    )

    st.markdown('<span class="step-pill">Step 1 · Inspect the columns</span>', unsafe_allow_html=True)
    st.markdown("The dataset arrived with 14 columns describing each trip:")
    cols = ['Trip_ID','Departure_Station','Arrival_Station','Departure_Time','Arrival_Time',
            'Trip_Date','Train_Model','Class_Type','Distance_KM','Occupancy_Rate_%',
            'Seat_Status','Total_Fare_SAR','Efficiency_Score','Logistics_Note']
    st.code(", ".join(cols), language=None)

    st.markdown('<span class="step-pill">Step 2 · Fix the data types</span>', unsafe_allow_html=True)
    explain(
        "Several columns were stored in the wrong format:<br>"
        "• <b>Trip_Date</b> was text → converted to a real datetime so months can be extracted.<br>"
        "• <b>Occupancy_Rate_%</b> looked like <code>'96%'</code> (text) → the % was stripped and "
        "the value divided by 100 to become a clean decimal (0.96).<br>"
        "• <b>Departure_Time</b> / <b>Arrival_Time</b> were text → parsed as times so trip duration "
        "could be computed."
    )

    st.markdown('<span class="step-pill">Step 3 · Drop columns that add no value</span>', unsafe_allow_html=True)
    explain(
        "<b>Trip_ID</b> is just a unique label (no analytical value) and <b>Efficiency_Score</b> "
        "only took two repeated values (0.5 / 1.1), so both were removed to keep the analysis focused."
    )

    st.markdown('<span class="step-pill">Step 4 · Engineer a new feature: Time_taken</span>', unsafe_allow_html=True)
    explain(
        "A new column <b>Time_taken</b> (minutes) was created as <code>Arrival_Time − Departure_Time</code>. "
        "Trips that cross midnight produced negative values, which were corrected by adding 1,440 minutes "
        "(a full day). This turned a raw timestamp into a usable measure of journey length."
    )

    st.markdown("---")
    st.markdown("### Cleaned dataset preview")
    st.dataframe(df_all.head(8), use_container_width=True)

    st.markdown("### Summary statistics")
    explain("After cleaning, the numeric columns describe themselves cleanly:")
    desc = df_all[["Distance_KM","Occupancy_Rate_%","Total_Fare_SAR","Time_taken"]].describe().round(2)
    st.dataframe(desc, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔎 Explore the raw records")
    explain(
        "Browse the full cleaned dataset yourself. Pick how many rows to show and which "
        "column to sort by — the table is fully interactive (click a header to re-sort)."
    )
    cexp1, cexp2, cexp3 = st.columns([1, 1, 1])
    n_rows = cexp1.slider("Rows to display", 5, 100, 20, step=5)
    sort_col = cexp2.selectbox("Sort by", df_all.columns.tolist(),
                               index=df_all.columns.get_loc("Trip_Date"))
    ascending = cexp3.radio("Order", ["Ascending", "Descending"], horizontal=True) == "Ascending"
    st.dataframe(
        df_all.sort_values(sort_col, ascending=ascending).head(n_rows),
        use_container_width=True, height=420,
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: CLASS & FARES
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🎫 Class & Fares":
    st.markdown("# 🎫 Travel Class & Fares")
    filter_banner()
    explain(
        "The railway offers two travel classes — <b>Economy</b> and <b>Business</b>. "
        "Here we look at how often each is chosen and how much each costs."
    )

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### How many trips per class?")
        cc = df["Class_Type"].value_counts()
        fig = px.bar(x=cc.index, y=cc.values, text=cc.values,
                     color=cc.index, color_discrete_sequence=[GREEN["dark"], GREEN["light"]])
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, height=420, title="Ticket class comparison")
        style_fig(fig, ytitle="Number of Trips")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Share of each class")
        fig = go.Figure(go.Pie(
            labels=cc.index, values=cc.values, hole=0.45,
            marker=dict(colors=[GREEN["dark"], GREEN["lighter"]]),
            textinfo="label+percent", textfont=dict(size=15, color="white")))
        fig.update_layout(showlegend=False, height=420, title="Class distribution",
                          **{k:v for k,v in PLOTLY_LAYOUT.items() if k!="title"})
        st.plotly_chart(fig, use_container_width=True)

    insight(
        f"More than <b>70%</b> of all trips are booked in <b>Economy</b> class "
        f"({(df['Class_Type']=='Economy').mean()*100:.0f}% here). Economy is clearly the "
        "backbone of demand — most travellers on this corridor are price-sensitive pilgrims "
        "and commuters rather than premium riders.",
    )

    st.markdown("---")
    st.markdown("### Average fare by class")
    means = df.groupby("Class_Type")["Total_Fare_SAR"].mean().round(1)
    c1, c2 = st.columns(2)
    c1.metric("Economy — average fare", f"{means.get('Economy',0):.0f} SAR")
    c2.metric("Business — average fare", f"{means.get('Business',0):.0f} SAR")
    explain(
        "Business fares run roughly <b>2.2×</b> higher than Economy on the same routes. "
        "The fare gap is consistent because both classes ride identical Talgo 350 trains over "
        "fixed-distance segments — you are paying for comfort, not a different journey."
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: DISTANCE vs PRICE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔗 Distance ↔ Price":
    st.markdown("# 🔗 Distance & Price Relationship")
    filter_banner()
    explain(
        "One of the first questions in any travel dataset: does the price follow the distance? "
        "We test this with a correlation matrix and by looking at the exact fare for each distance."
    )

    corr = df[["Distance_KM","Occupancy_Rate_%","Total_Fare_SAR"]].corr()
    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown("### Correlation heatmap")
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=GREEN_SCALE,
                        zmin=-0.1, zmax=1)
        fig.update_layout(height=420, title="Numeric correlations")
        fig.update_layout(**{k:v for k,v in PLOTLY_LAYOUT.items() if k!="title"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Fare for each distance band")
        band = (df.groupby(["Distance_KM","Class_Type"])["Total_Fare_SAR"]
                  .first().reset_index())
        fig = px.bar(band, x="Distance_KM", y="Total_Fare_SAR", color="Class_Type",
                     barmode="group", text="Total_Fare_SAR",
                     color_discrete_sequence=[GREEN["light"], GREEN["dark"]])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=420, title="Fare by distance & class",
                          legend_title="Class")
        style_fig(fig, ytitle="Fare (SAR)", xtitle="Distance (km)")
        st.plotly_chart(fig, use_container_width=True)

    r = df["Distance_KM"].corr(df["Total_Fare_SAR"])
    insight(
        f"There is a <b>strong positive correlation ({r:.2f})</b> between distance and price: "
        "the longer the trip, the higher the fare. The four distance bands (78, 95, 186, 450 km) "
        "each map to a fixed Economy and Business price — a clean, distance-based pricing structure.",
    )

    st.markdown("### The exact fare ladder")
    ladder = pd.DataFrame({
        "Distance (km)": [78, 95, 186, 450],
        "Economy (SAR)": [40.0, 47.5, 93.0, 225.0],
        "Business (SAR)": [85.8, 104.5, 204.6, 495.0],
    })
    st.dataframe(ladder, use_container_width=True, hide_index=True)
    explain(
        "Notice occupancy has essentially <b>zero correlation</b> with both distance and fare — "
        "how full a train is doesn't depend on how far or how expensive the trip is. Demand is "
        "driven by something else entirely: the calendar (see the Seasonality section)."
    )

    st.markdown("---")
    st.markdown("### 🧮 Try it: fare & journey estimator")
    explain("Pick a route and class to see the exact fare, distance, and journey time.")
    routes = (df_all.groupby(["Departure_Station", "Arrival_Station"])
                    .agg(Distance_KM=("Distance_KM", "first"),
                         Time_taken=("Time_taken", "mean")).reset_index())
    routes["label"] = routes["Departure_Station"] + "  →  " + routes["Arrival_Station"]
    e1, e2 = st.columns([2, 1])
    chosen = e1.selectbox("Route", routes["label"].tolist())
    chosen_class = e2.radio("Class", ["Economy", "Business"], horizontal=True)
    row = routes[routes["label"] == chosen].iloc[0]
    fare_lookup = {
        (78, "Economy"): 40.0, (78, "Business"): 85.8,
        (95, "Economy"): 47.5, (95, "Business"): 104.5,
        (186, "Economy"): 93.0, (186, "Business"): 204.6,
        (450, "Economy"): 225.0, (450, "Business"): 495.0,
    }
    fare = fare_lookup.get((int(row["Distance_KM"]), chosen_class), float("nan"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated fare", f"{fare:.1f} SAR")
    m2.metric("Distance", f"{int(row['Distance_KM'])} km")
    m3.metric("Journey time", f"{row['Time_taken']:.0f} min")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: STATIONS & ROUTES
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Stations & Routes":
    st.markdown("# 🗺️ Stations & Routes")
    filter_banner()
    explain(
        "Which stations dominate the network, and are most journeys short direct hops or "
        "longer multi-stop hauls? These two questions shape how we read everything else."
    )

    dep = df["Departure_Station"].value_counts()
    arr = df["Arrival_Station"].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Most frequent departure stations")
        fig = px.bar(x=dep.values, y=dep.index, orientation="h", text=dep.values,
                     color=dep.values, color_continuous_scale=GREEN_SCALE)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False,
                          title="Departures by station",
                          yaxis={'categoryorder':'total ascending'})
        style_fig(fig, xtitle="Number of Trips")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Most frequent arrival stations")
        fig = px.bar(x=arr.values, y=arr.index, orientation="h", text=arr.values,
                     color=arr.values, color_continuous_scale=GREEN_SCALE)
        fig.update_traces(textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False,
                          title="Arrivals by station",
                          yaxis={'categoryorder':'total ascending'})
        style_fig(fig, xtitle="Number of Trips")
        st.plotly_chart(fig, use_container_width=True)

    insight(
        "<b>Makkah Al-Russaifah</b> is by far the most frequent station for both departures and "
        "arrivals, with <b>Madinah</b> second. The whole network behaves like a hub-and-spoke "
        "system centred on Makkah — fitting for the corridor that carries pilgrims to the holy city.",
    )

    st.markdown("---")
    st.markdown("### Direct routes vs long hauls")
    logis = df["Logistics_Note"].value_counts()
    fig = px.bar(x=logis.index, y=logis.values, text=logis.values,
                 color=logis.index,
                 color_discrete_sequence=[GREEN["dark"], GREEN["light"]])
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, showlegend=False, title="Trip type breakdown")
    style_fig(fig, ytitle="Number of Trips")
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "Most journeys are flagged as <b>Long Haul</b> — longer trips that usually include stops. "
        "These correspond to the 186 km and 450 km segments, while the 78 km and 95 km segments "
        "are the quick <b>Direct Routes</b>.",
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: SEASONALITY
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📅 Seasonality":
    st.markdown("# 📅 When Do People Travel?")
    filter_banner()
    explain(
        "Timing is everything on this corridor. By extracting the month from each trip date we "
        "can see when demand peaks — and connect it to the Islamic calendar."
    )

    monthly = df["Trip_Date"].dt.month_name().value_counts().reindex(MONTH_ORDER).dropna()
    fig = px.bar(x=monthly.index, y=monthly.values, text=monthly.values.astype(int),
                 color=monthly.values, color_continuous_scale=GREEN_SCALE)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=450, coloraxis_showscale=False, title="Number of trips per month")
    style_fig(fig, ytitle="Number of Trips", xtitle="Month")
    st.plotly_chart(fig, use_container_width=True)

    insight(
        "Activity concentrates from <b>April (Shawwāl)</b> through <b>July (Muharram)</b>. "
        "These are months strongly associated with <b>ʿibādah</b> — performing ʿUmrah and "
        "preparing for Hajj. The travel pattern mirrors the religious calendar almost perfectly: "
        "people move toward the holy cities during the season of worship.",
    )

    st.markdown("---")
    st.markdown("### Trips to Makkah over time")
    mak = df[df["Arrival_Station"]=="Makkah"].copy()
    mak_daily = mak.groupby(mak["Trip_Date"].dt.to_period("W").dt.start_time).size()
    fig = px.area(x=mak_daily.index, y=mak_daily.values)
    fig.update_traces(line_color=GREEN["dark"], fillcolor="rgba(82,183,136,0.35)")
    fig.update_layout(height=360, title="Weekly arrivals into Makkah")
    style_fig(fig, ytitle="Trips", xtitle="Week")
    st.plotly_chart(fig, use_container_width=True)
    explain(
        f"Makkah arrivals span <b>{mak['Trip_Date'].min():%b %d, %Y}</b> to "
        f"<b>{mak['Trip_Date'].max():%b %d, %Y}</b> — the heart of the worship season."
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: TRIP DURATION
# ──────────────────────────────────────────────────────────────────────────────
elif page == "⏱️ Trip Duration":
    st.markdown("# ⏱️ How Long Does Each Trip Take?")
    filter_banner()
    explain(
        "Using the engineered <b>Time_taken</b> column, we examine how long journeys last, "
        "whether durations are stable per route, and whether the apparent 'outliers' are real."
    )

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Duration spread (box plot)")
        fig = go.Figure(go.Box(y=df["Time_taken"], name="Time taken",
                               marker_color=GREEN["dark"],
                               fillcolor="rgba(82,183,136,0.4)"))
        fig.update_layout(height=420, title="Trip duration distribution")
        style_fig(fig, ytitle="Minutes")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Average duration by arrival station")
        dur = (df.groupby("Arrival_Station")["Time_taken"]
                 .agg(["mean","min","max","count"]).round(1)
                 .sort_values("mean"))
        st.dataframe(dur, use_container_width=True)
        explain(
            "Each route shows a single fixed duration (min = max) — exactly what you expect from "
            "a fixed-schedule high-speed line."
        )

    insight(
        "Most routes have a <b>single fixed duration</b> (min = max), which is realistic for a "
        "fixed-schedule high-speed line like Al-Haramain — the same segment always takes the same "
        "time. The spread we see comes from <b>different origins</b>, not from genuine outliers.",
    )

    st.markdown("---")
    st.markdown("### Average duration by departure station")
    dep_dur = df.groupby("Departure_Station")["Time_taken"].mean().round(1).sort_values()
    fig = px.bar(x=dep_dur.values, y=dep_dur.index, orientation="h", text=dep_dur.values,
                 color=dep_dur.values, color_continuous_scale=GREEN_SCALE)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=380, coloraxis_showscale=False,
                      title="Mean trip time by departure station")
    style_fig(fig, xtitle="Average minutes")
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "Average times look healthy whether departure exceeds arrival or the reverse. This points "
        "to an <b>Al-Haramain network that is highly organised</b> and able to serve travellers well.",
    )

    c1, c2 = st.columns(2)
    c1.metric("Avg time arriving INTO Makkah", f"{df[df['Arrival_Station']=='Makkah']['Time_taken'].mean():.0f} min")
    c2.metric("Avg time departing FROM Makkah", f"{df[df['Departure_Station']=='Makkah']['Time_taken'].mean():.0f} min")

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: OCCUPANCY & SEATS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "💺 Occupancy & Seats":
    st.markdown("# 💺 Occupancy & Seat Status")
    filter_banner()
    explain(
        "How full are the trains, and what is the booking status of seats across the network? "
        "These tell us how well capacity is being used."
    )

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("### Distribution of occupancy rate")
        fig = px.histogram(df, x="Occupancy_Rate_%", nbins=25,
                           color_discrete_sequence=[GREEN["base"]])
        fig.update_traces(marker_line_color=GREEN["dark"], marker_line_width=1)
        fig.update_layout(height=420, title="How full are the trains?", bargap=0.05)
        style_fig(fig, ytitle="Number of Trips", xtitle="Occupancy Rate")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Seat status split")
        seats = df["Seat_Status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=seats.index, values=seats.values, hole=0.45,
            marker=dict(colors=[GREEN["dark"], GREEN["light"], GREEN["lighter"]]),
            textinfo="label+percent", textfont=dict(size=14, color="white")))
        fig.update_layout(height=420, title="Seat status distribution", showlegend=False,
                          **{k:v for k,v in PLOTLY_LAYOUT.items() if k!="title"})
        st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Average occupancy", f"{df['Occupancy_Rate_%'].mean()*100:.0f}%")
    c2.metric("Lowest occupancy", f"{df['Occupancy_Rate_%'].min()*100:.0f}%")
    c3.metric("Highest occupancy", f"{df['Occupancy_Rate_%'].max()*100:.0f}%")

    insight(
        "Occupancy clusters tightly around <b>82% on average</b> and never drops below 65%. "
        "The trains run consistently full — strong, stable demand across the whole corridor with "
        "no half-empty services dragging the network down.",
    )

    st.markdown("---")
    st.markdown("### Average fare per destination by class")
    explain("Bringing fare, destination and class together to see where the premium routes are.")
    pivot = df.pivot_table(values="Total_Fare_SAR", index="Arrival_Station",
                           columns="Class_Type", aggfunc="mean").round(0)
    fig = px.bar(pivot, barmode="group", text_auto=".0f",
                 color_discrete_sequence=[GREEN["dark"], GREEN["light"]])
    fig.update_traces(textposition="outside")
    fig.update_layout(height=420, title="Average fare per destination by class",
                      legend_title="Class")
    style_fig(fig, ytitle="Average Fare (SAR)", xtitle="Destination")
    st.plotly_chart(fig, use_container_width=True)
    insight(
        "The <b>highest-cost trips run to or from Madinah</b>, then Makkah, then Rabigh — a direct "
        "consequence of the distance-based fare ladder, since the Madinah↔Makkah leg is the longest "
        "(450 km). Economy is used most heavily on the Madinah corridor, with Makkah close behind.",
    )

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE: KEY TAKEAWAYS
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📌 Key Takeaways":
    st.markdown("# 📌 Key Takeaways")
    explain(
        "Everything the analysis uncovered, gathered in one place. These are the author's own "
        "conclusions, drawn directly from the data."
    )

    takeaways = [
        ("🗺️", "Makkah is the hub",
         "Makkah Al-Russaifah is the most frequent station for both departures and arrivals, "
         "with Madinah second. The network is hub-and-spoke, centred on the holy city."),
        ("📅", "Travel follows the worship calendar",
         "Demand peaks from April (Shawwāl) to July (Muharram) — months tied to ʿUmrah and Hajj "
         "preparation. The data mirrors the religious season."),
        ("🔗", "Price follows distance",
         "A strong 0.82 correlation links distance to fare. Four distance bands (78/95/186/450 km) "
         "map to fixed Economy and Business prices."),
        ("🎫", "Economy dominates",
         "More than 70% of trips are Economy class — this is a price-sensitive, high-volume corridor."),
        ("💰", "Madinah is the priciest corridor",
         "Highest fares are on Madinah routes (longest leg at 450 km), then Makkah, then Rabigh."),
        ("⏱️", "Durations are fixed and reliable",
         "Each route has one fixed duration (min = max) — the mark of a well-run, fixed-schedule "
         "high-speed line. The spread comes from different origins, not outliers."),
        ("🛤️", "Mostly long-haul with stops",
         "Most journeys are Long Haul (186 & 450 km segments); the 78 & 95 km segments are quick "
         "Direct Routes."),
        ("💺", "Trains run consistently full",
         "Occupancy averages ~82% and never falls below 65% — stable, healthy demand network-wide."),
        ("🏛️", "A highly organised operator",
         "Balanced average times in both directions suggest Al-Haramain is highly organised and "
         "serves travellers well."),
    ]
    for i in range(0, len(takeaways), 3):
        cols = st.columns(3)
        for col, (icon, title, body) in zip(cols, takeaways[i:i+3]):
            with col:
                st.markdown(
                    f'<div class="insight" style="min-height:200px;">'
                    f'<div style="font-size:1.8rem;">{icon}</div>'
                    f'<b style="font-size:1.05rem;">{title}</b><br>'
                    f'<span style="font-size:0.93rem;">{body}</span></div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div class="explain" style="text-align:center; font-size:1.05rem;">'
        f'<b>Al-Haramain High-Speed Railway EDA</b> · 1,350 trips · 5 stations · Apr–Jul 2026<br>'
        f'Exploratory analysis & insights by the original author · Visualised with a light-green theme'
        f'</div>', unsafe_allow_html=True)
