import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AM2 Office Attendance Prediction", layout="wide")
st.title("AM2 Project: Office Attendance Prediction")

month_order = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
               'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

# ---------- Load 2025 training data ----------
df = pd.read_csv("data/am2_2025_data.csv")

st.write("### 2025 data preview")
st.dataframe(df.head())

# ---------- Aggregate to office level ----------
def aggregate_office(data):
    headcount = data.groupby(['Office', 'Year', 'Month'])['Full Name'].nunique().reset_index()
    headcount.columns = ['Office', 'Year', 'Month', 'Headcount']

    office_data = data.groupby(['Office', 'Year', 'Month', 'Office Desks per Office',
                                 'Working Days (In Month)']).agg(
        Total_Days_In=('Days In', 'sum')
    ).reset_index()

    office_data = office_data.merge(headcount, on=['Office', 'Year', 'Month'])
    office_data['utilisation_rate'] = (
        office_data['Total_Days_In'] / (office_data['Headcount'] * office_data['Working Days (In Month)'])
    )
    office_data['month_num'] = office_data['Month'].map(month_order)
    return office_data.sort_values(['Office', 'month_num']).reset_index(drop=True)

office_df = aggregate_office(df)

# ---------- Feature engineering ----------
office_df['month_sin'] = np.sin(2 * np.pi * office_df['month_num'] / 12)
office_df['month_cos'] = np.cos(2 * np.pi * office_df['month_num'] / 12)
office_df['lag_1'] = office_df.groupby('Office')['utilisation_rate'].shift(1)
office_df['lag_2'] = office_df.groupby('Office')['utilisation_rate'].shift(2)
office_df['lag_3'] = office_df.groupby('Office')['utilisation_rate'].shift(3)
office_df['rolling_3m_avg'] = (
    office_df.groupby('Office')['utilisation_rate']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
office_df['desk_pressure'] = office_df['Headcount'] / office_df['Office Desks per Office']

le = LabelEncoder()
office_df['office_encoded'] = le.fit_transform(office_df['Office'])

feature_cols = ['office_encoded', 'Working Days (In Month)', 'month_num', 'month_sin',
                 'month_cos', 'desk_pressure', 'lag_1', 'lag_2', 'lag_3', 'rolling_3m_avg']
target_col = 'utilisation_rate'

model_df = office_df.dropna(subset=feature_cols).reset_index(drop=True)

# ---------- Train XGBoost model ----------
X = model_df[feature_cols]
y = model_df[target_col]

tscv = TimeSeriesSplit(n_splits=3)
model = XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)
cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
model.fit(X, y)

st.write("### Cross-validation performance (2025 data)")
st.write(f"Average CV MAE: **{-cv_scores.mean()*100:.1f}%** utilisation rate")

# ---------- Generate 2026 predictions ----------
future_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July']
future_month_nums = [1, 2, 3, 4, 5, 6, 7]
working_days_2026 = {1:21, 2:20, 3:21, 4:21, 5:20, 6:22, 7:23}

predictions_list = []
for office in office_df['Office'].unique():
    office_history = office_df[office_df['Office'] == office].sort_values('month_num')
    desk_count = office_history['Office Desks per Office'].iloc[0]
    office_enc = office_history['office_encoded'].iloc[0]
    recent_rates = list(office_history['utilisation_rate'].values)

    for month, month_num in zip(future_months, future_month_nums):
        working_days = working_days_2026[month_num]
        headcount = office_history['Headcount'].iloc[-1]
        desk_pressure = headcount / desk_count
        lag_1, lag_2, lag_3 = recent_rates[-1], recent_rates[-2], recent_rates[-3]
        rolling_avg = np.mean(recent_rates[-3:])
        month_sin = np.sin(2 * np.pi * month_num / 12)
        month_cos = np.cos(2 * np.pi * month_num / 12)

        row = {
            'Office': office, 'Month': month, 'month_num': month_num, 'Year': 2026,
            'office_encoded': office_enc, 'Working Days (In Month)': working_days,
            'month_sin': month_sin, 'month_cos': month_cos, 'desk_pressure': desk_pressure,
            'lag_1': lag_1, 'lag_2': lag_2, 'lag_3': lag_3, 'rolling_3m_avg': rolling_avg
        }
        X_pred = pd.DataFrame([row])[feature_cols]
        predicted_rate = model.predict(X_pred)[0]
        row['predicted_utilisation_rate'] = predicted_rate
        predictions_list.append(row)
        recent_rates.append(predicted_rate)

predictions_df = pd.DataFrame(predictions_list)

st.write("### 2026 Predictions")
st.dataframe(predictions_df[['Office', 'Month', 'predicted_utilisation_rate']])

# ---------- Load 2026 actuals and evaluate ----------
st.write("### Prediction vs Actual (2026)")

try:
    actuals_df = pd.read_csv("data/am2_2026_actuals.csv")
    actuals_office = aggregate_office(actuals_df).rename(columns={'utilisation_rate': 'actual_utilisation_rate'})

    eval_df = predictions_df.merge(
        actuals_office[['Office', 'Month', 'actual_utilisation_rate']],
        on=['Office', 'Month'], how='inner'
    )
    eval_df['error'] = eval_df['predicted_utilisation_rate'] - eval_df['actual_utilisation_rate']
    eval_df['abs_error'] = eval_df['error'].abs()
    eval_df['pct_error'] = (eval_df['abs_error'] / eval_df['actual_utilisation_rate']) * 100

    mae = mean_absolute_error(eval_df['actual_utilisation_rate'], eval_df['predicted_utilisation_rate'])
    rmse = np.sqrt(mean_squared_error(eval_df['actual_utilisation_rate'], eval_df['predicted_utilisation_rate']))
    mape = eval_df['pct_error'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae*100:.1f}%")
    col2.metric("RMSE", f"{rmse*100:.1f}%")
    col3.metric("MAPE", f"{mape:.1f}%")

    st.write("#### Performance by office")
    office_metrics = eval_df.groupby('Office').agg(
        MAE=('abs_error', 'mean'), MAPE=('pct_error', 'mean'),
        Avg_Predicted=('predicted_utilisation_rate', 'mean'),
        Avg_Actual=('actual_utilisation_rate', 'mean')
    ).round(4)
    st.dataframe(office_metrics)

    st.write("#### Full prediction vs actual table")
    st.dataframe(eval_df[['Office', 'Month', 'predicted_utilisation_rate',
                           'actual_utilisation_rate', 'error', 'pct_error']])

    # ---------- Monitoring dashboard ----------
    st.write("### Model Monitoring Dashboard")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    months_seen = [m for m in future_months if m in eval_df['Month'].unique()]
    cumulative_mape, monthly_mape = [], []
    for i, month in enumerate(months_seen):
        subset = eval_df[eval_df['Month'].isin(months_seen[:i+1])]
        cumulative_mape.append(subset['pct_error'].mean())
        monthly_mape.append(eval_df[eval_df['Month'] == month]['pct_error'].mean())

    ax1 = axes[0, 0]
    x = range(len(months_seen))
    ax1.plot(x, monthly_mape, marker='o', label='Monthly MAPE', color='#FF5722')
    ax1.plot(x, cumulative_mape, marker='s', linestyle='--', label='Cumulative MAPE', color='#2196F3')
    ax1.axhline(y=10, color='red', linestyle=':', alpha=0.7, label='10% threshold')
    ax1.set_xticks(x); ax1.set_xticklabels(months_seen, rotation=45)
    ax1.set_title('MAPE Over Time'); ax1.legend(); ax1.grid(alpha=0.3)

    ax2 = axes[0, 1]
    for office in sorted(eval_df['Office'].unique()):
        od = eval_df[eval_df['Office'] == office].copy()
        od['m'] = od['Month'].map({m: i for i, m in enumerate(months_seen)})
        od = od.sort_values('m')
        ax2.plot(range(len(od)), od['error'] * 100, marker='o', label=office)
    ax2.axhline(y=0, color='black', linewidth=1)
    ax2.set_xticks(range(len(months_seen))); ax2.set_xticklabels(months_seen, rotation=45)
    ax2.set_title('Prediction Error Drift by Office'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    ax3 = axes[1, 0]
    monthly_avg = eval_df.groupby('Month').agg(
        Predicted=('predicted_utilisation_rate', 'mean'), Actual=('actual_utilisation_rate', 'mean')
    ).reindex(months_seen)
    x = range(len(months_seen))
    ax3.plot(x, monthly_avg['Actual']*100, marker='o', label='Actual', color='#2196F3')
    ax3.plot(x, monthly_avg['Predicted']*100, marker='s', linestyle='--', label='Predicted', color='#FF5722')
    ax3.fill_between(x, monthly_avg['Actual']*100, monthly_avg['Predicted']*100, alpha=0.15, color='red')
    ax3.set_xticks(x); ax3.set_xticklabels(months_seen, rotation=45)
    ax3.set_title('Avg Utilisation: Predicted vs Actual'); ax3.legend(); ax3.grid(alpha=0.3)

    ax4 = axes[1, 1]
    office_mape = eval_df.groupby('Office')['pct_error'].mean().sort_values()
    colours = ['#4CAF50' if v < 10 else '#FF9800' if v < 15 else '#F44336' for v in office_mape.values]
    ax4.barh(office_mape.index, office_mape.values, color=colours)
    ax4.axvline(x=10, color='orange', linestyle='--', alpha=0.7)
    ax4.axvline(x=15, color='red', linestyle='--', alpha=0.7)
    ax4.set_title('Model Accuracy by Office (MAPE)'); ax4.grid(alpha=0.3, axis='x')

    plt.tight_layout()
    st.pyplot(fig)

except FileNotFoundError:
    st.info("Upload `data/am2_2026_actuals.csv` to see prediction-vs-actual comparison and the monitoring dashboard.")
