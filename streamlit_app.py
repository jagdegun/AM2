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

"A supervised regression model trained on historical data and regression predicting a continuous numerical value."

month_order = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6,
               'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12}

# ---------- Load 2025 training data ----------
df = pd.read_csv("data/am2_2025_data.csv")

st.write("### 2025 data preview")
st.dataframe(df.drop(columns=['Full Name']))

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
st.write(f"Average CV MAE (Mean Absolute Error): **{-cv_scores.mean()*100:.1f}%** utilisation rate")

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
"MAE - Seeing the difference between what the model predicted and what actually happened"
"RMSE - This penalises large errors similar to MAE"
"MAPE - It expresses error as a percentage"

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
    col1.metric("MAE (Mean Absolute Error)", f"{mae*100:.1f}%")
    col2.metric("RMSE (Root Mean Squared Error)", f"{rmse*100:.1f}%")
    col3.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.1f}%")

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

except FileNotFoundError:
    st.info("Upload `data/am2_2026_actuals.csv` to see prediction-vs-actual comparison and the monitoring dashboard.")
