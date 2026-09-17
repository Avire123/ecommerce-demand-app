import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# Ensure target directories exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ====================================
# 1. DATA GENERATION
# ====================================

np.random.seed(42)
n_samples = 3000

categories = ["Electronics", "Apparel", "Home & Kitchen", "Beauty", "Sports"]
dates = pd.date_range(start="2024-01-01", periods=100, freq="D")

data = {
    "Date": np.random.choice(dates, n_samples),
    "Category": np.random.choice(categories, n_samples),
    "BasePrice": np.random.uniform(15.0, 3000, n_samples),
    "Discount": np.random.choice([0,5,10,15,20,30,50], n_samples),
    "Rating": np.random.uniform(2.5,5.0,n_samples),
    "CompetitorPrice": np.random.uniform(15.0, 310.0,n_samples),
}

df = pd.DataFrame(data)
df["EffectivePrice"] = df["BasePrice"]*(1-df["Discount"]/100)

# Simulate Demand (Target) with noise & price sensitivity
base_demand= 150
price_impact = -0.35 * df["EffectivePrice"]
discount_impact = 0.8 * df["Discount"]
rating_impact = 12.0 * df["Rating"]
comp_impact = 0.15 * (df["CompetitorPrice"] - df["EffectivePrice"])
noise = np.random.normal(0, 15, n_samples)

df["SalesVolume"] = np.maximum(
    5,
    (
        base_demand
        + price_impact
        + discount_impact
        + rating_impact
        + comp_impact
        + noise
    ).astype(int),
)

# ====================================
# 2. FEATURE ENGINEERING
# ====================================

def engineer_features(data_frame):
    df_feat = data_frame.copy()
    df_feat["Date"] = pd.to_datetime(df_feat["Date"])
    df_feat["DayOfWeek"] = df_feat["Date"].dt.dayofweek
    df_feat["Month"] = df_feat["Date"].dt.month
    df_feat["IsWeekend"] = df_feat["DayOfWeek"].apply(lambda x: 1 if x >=5 else 0)
    df_feat["PriceDiffvsCompetitor"] = (
        df_feat["CompetitorPrice"] - df_feat["EffectivePrice"]
    )
    df_feat["PriceRatio"] = (
        df_feat["EffectivePrice"] / df_feat["CompetitorPrice"]
    )
    return df_feat

df_processed = engineer_features(df)

features = [
    "Category",
    "BasePrice",
    "EffectivePrice",
    "Discount",
    "Rating",
    "CompetitorPrice",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "PriceDiffvsCompetitor",
    "PriceRatio",
]

X = df_processed[features]
y = df_processed["SalesVolume"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ====================================
# 3. PIPELINE & HYPERPARAMETER TUNING
# ====================================

numeric_features = [
    "BasePrice",
    "EffectivePrice",
    "Discount",
    "Rating",
    "CompetitorPrice",
    "DayOfWeek",
    "Month",
    "IsWeekend",
    "PriceDiffvsCompetitor",
    "PriceRatio",
]
categorical_features = ["Category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_features,
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(random_state=42)),
    ]
)

param_grid = {
    "regressor__n_estimators":[100, 150],
    "regressor__max_depth":[3,5],
    "regressor__learning_rate":[0.05, 0.1],
}

print("Running GridSearchCV model tuning...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_jobs=1,
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_

# ====================================
# 4. EVALUATION & ARTIFACT SAVING
# ====================================

y_pred =best_pipeline.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

# Save artifacts to paths app.py will read
joblib.dump(best_pipeline, "models/demand_xgboost_model.pkl")
df_processed.to_csv("data/cleaned_ecommerce_data.csv", index=False)

print("\nSaved artifacts to 'models/demand_xgboost_model.pkl' and 'data/cleaned_ecommerce_data.csv'.")
