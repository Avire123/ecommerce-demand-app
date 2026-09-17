# 📦 E-Commerce Demand Forecasting & Analytics App

An end-to-end Machine Learning web application that predicts product sales volume and simulated revenue using **XGBoost Regressor**, equipped with an interactive **Streamlit** dashboard and **Plotly** visualizations.

---

## 🌟 Features

- **Interactive Demand Predictor**: Input product parameters (Category, Base Price, Discount %, Competitor Price, Rating, and Forecast Date) to predict expected sales volume and revenue.
- **Price Sensitivity & Demand Curve**: Dynamically simulate sales volume across varying discount tiers (0% to 60%) to identify revenue-maximizing pricing strategies.
- **Competitor Benchmark**: Real-time comparison against competitor pricing with dynamic pricing advantage metrics.
- **Interactive Visualizations**: Interactive Plotly distribution plots, category volume box plots, and price elasticity scatter charts.
- **Trained ML Pipeline**: Uses `scikit-learn` Pipeline with `ColumnTransformer` (StandardScaler + OneHotEncoder) and tuned `XGBRegressor`.

---

## 📊 Model Performance

Tuned via 3-Fold `GridSearchCV` hyperparameter optimization:

| Metric | Score |
| :--- | :--- |
| **MAE (Mean Absolute Error)** | **2.15** |
| **RMSE (Root Mean Squared Error)** | **5.75** |
| **$R^2$ Score** | **0.9874** |

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [scikit-learn](https://scikit-learn.org/)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Visualizations**: [Plotly Express & Graph Objects](https://plotly.com/python/)
- **Model Persistence**: [Joblib](https://joblib.readthedocs.io/)

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Avire123/ecommerce-demand-app.git
cd ecommerce-demand-app
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-train the Machine Learning Model
To regenerate the dataset and re-train the XGBoost model:
```bash
python train_model.py
```

### 5. Launch the Streamlit App
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```plaintext
ecommerce-demand-app/
│
├── app.py                      # Streamlit interactive application
├── train_model.py              # ML data generation, training, and tuning pipeline
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
│
├── data/
│   └── cleaned_ecommerce_data.csv   # Preprocessed e-commerce dataset
│
└── models/
    └── demand_xgboost_model.pkl    # Serialized trained XGBoost pipeline
```

---

## 👤 Author
- GitHub: [@Avire123](https://github.com/Avire123)
