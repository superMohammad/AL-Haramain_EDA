# 🚄 Al-Haramain Railway — EDA Dashboard (Streamlit)

An interactive Streamlit presentation of the exploratory analysis written in
`AL-Haramain_EDA.ipynb`. Every insight, question, and conclusion from the notebook
is preserved exactly as written — only the **chart colour theme** has been made
consistent using a **light-green palette**.

## ▶️ How to run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

## 📂 Files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit dashboard (9 sections, light-green theme) |
| `haramain.parquet` | The cleaned dataset the app reads |
| `build_data.py` | Rebuilds the dataset (matches the notebook's exact statistics) |
| `requirements.txt` | Python dependencies |

## 🔄 Using your own original Excel file

This package ships with a dataset reconstructed to match every statistic printed
in your notebook (station counts, fare ladder, durations, correlation, date range,
class split), because the original `HHR_Train (1).xlsx` was not bundled with the
notebook.

To plug in your **real** Excel file, replace the `load_data()` function in `app.py`
with the notebook's own cleaning steps:

```python
@st.cache_data
def load_data():
    df = pd.read_excel("HHR_Train (1).xlsx")
    df['Trip_Date'] = pd.to_datetime(df['Trip_Date'], format='%Y-%m-%d')
    df['Occupancy_Rate_%'] = df['Occupancy_Rate_%'].str.replace('%','').astype(float)/100
    df.drop(columns=['Trip_ID','Efficiency_Score'], inplace=True)
    df['Arrival_Time']   = pd.to_datetime(df['Arrival_Time'],  format='%H:%M')
    df['Departure_Time'] = pd.to_datetime(df['Departure_Time'],format='%H:%M')
    df['Time_taken'] = (df['Arrival_Time']-df['Departure_Time']).dt.total_seconds()/60
    df.loc[df['Time_taken'] < 0, 'Time_taken'] += 1440
    return df
```

Everything downstream (charts, insights, layout) works unchanged.

## 🗂️ Sections

1. **Overview** — KPIs and the network at a glance
2. **Data Preparation** — every cleaning step explained
3. **Class & Fares** — Economy vs Business
4. **Distance ↔ Price** — the correlation and fare ladder
5. **Stations & Routes** — hub-and-spoke + direct vs long-haul
6. **Seasonality** — the worship-calendar travel pattern
7. **Trip Duration** — fixed schedules, no real outliers
8. **Occupancy & Seats** — how full the trains run
9. **Key Takeaways** — all conclusions in one place
