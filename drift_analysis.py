import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# 1. Generate Realistic Smart Grid Dataset (10,000 Hourly Records)
np.random.seed(42)
n_ref = 5000  # Reference Period (Winter/Spring)
n_curr = 5000 # Current Period (Summer Peak - Natural Seasonal Drift)

# Reference Data (Normal Grid Conditions)
ref_df = pd.DataFrame({
    'ambient_temperature': np.random.normal(24.0, 3.5, n_ref),
    'humidity': np.random.normal(55.0, 10.0, n_ref),
    'solar_irradiance': np.clip(np.random.normal(450, 150, n_ref), 0, 1000),
    'grid_voltage': np.random.normal(230.0, 2.5, n_ref)
})
ref_df['power_load_mw'] = (
    120 + 3.2 * ref_df['ambient_temperature'] 
    + 0.4 * ref_df['humidity'] 
    - 0.05 * ref_df['solar_irradiance'] 
    + np.random.normal(0, 5, n_ref)
)

# Current Data (Summer Heatwave -> Higher Temp & Higher AC Power Demand)
curr_df = pd.DataFrame({
    'ambient_temperature': np.random.normal(33.5, 4.0, n_curr), # Drifted!
    'humidity': np.random.normal(56.0, 10.5, n_curr),           # Stable
    'solar_irradiance': np.clip(np.random.normal(620, 160, n_curr), 0, 1000), # Drifted!
    'grid_voltage': np.random.normal(228.5, 3.0, n_curr)        # Slight sag
})
curr_df['power_load_mw'] = (
    140 + 4.5 * curr_df['ambient_temperature'] # Concept + Data shift due to heavy AC load
    + 0.4 * curr_df['humidity'] 
    - 0.05 * curr_df['solar_irradiance'] 
    + np.random.normal(0, 6, n_curr)
)

# Save datasets for GitHub reproducibility
ref_df.to_csv("reference_grid_data.csv", index=False)
curr_df.to_csv("current_grid_data.csv", index=False)

# 2. Configure MLflow Experiment
mlflow.set_experiment("Smart_Grid_Load_Forecasting_Drift")

features = ['ambient_temperature', 'humidity', 'solar_irradiance', 'grid_voltage']
target = 'power_load_mw'

X_train, X_test, y_train, y_test = train_test_split(
    ref_df[features], ref_df[target], test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="RF_Baseline_and_KS_Drift_Check"):
    # Log Model Hyperparameters
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    mlflow.log_params(params)

    # Train Model on Reference Data
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    # Evaluate on Reference Test Data vs. Current Drifted Data
    ref_preds = model.predict(X_test)
    curr_preds = model.predict(curr_df[features])

    ref_rmse = np.sqrt(mean_squared_error(y_test, ref_preds))
    curr_rmse = np.sqrt(mean_squared_error(curr_df[target], curr_preds))
    ref_r2 = r2_score(y_test, ref_preds)
    curr_r2 = r2_score(curr_df[target], curr_preds)

    # Log Performance Metrics
    mlflow.log_metric("ref_rmse", round(ref_rmse, 3))
    mlflow.log_metric("curr_rmse", round(curr_rmse, 3))
    mlflow.log_metric("ref_r2", round(ref_r2, 3))
    mlflow.log_metric("curr_r2", round(curr_r2, 3))

    # 3. Statistical Drift Detection (Kolmogorov-Smirnov Test)
    alpha = 0.05
    drift_results = []
    for col in features + [target]:
        stat, p_val = ks_2samp(ref_df[col], curr_df[col])
        drift_detected = p_val < alpha
        drift_results.append({
            "Feature": col,
            "KS_Statistic": round(stat, 4),
            "p_value": round(p_val, 6),
            "Drift_Detected": drift_detected
        })
        mlflow.log_metric(f"ks_stat_{col}", round(stat, 4))
        mlflow.log_metric(f"p_val_{col}", round(p_val, 6))

    drift_df = pd.DataFrame(drift_results)
    drift_df.to_csv("drift_report.csv", index=False)
    mlflow.log_artifact("drift_report.csv")

    # 4. Generate and Log Distribution Plot Artifact
    plt.figure(figsize=(8, 5))
    plt.hist(ref_df['ambient_temperature'], bins=40, alpha=0.6, label='Reference (Winter/Spring)', density=True)
    plt.hist(curr_df['ambient_temperature'], bins=40, alpha=0.6, label='Current (Summer Peak)', density=True)
    plt.title("Data Drift in Ambient Temperature (KS Test p < 0.05)")
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("temperature_drift_plot.png")
    mlflow.log_artifact("temperature_drift_plot.png")

    # Log the trained model
    mlflow.sklearn.log_model(
        model, "random_forest_grid_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("=== Drift Report ===")
    print(drift_df)
    print(f"\nReference RMSE: {ref_rmse:.2f} MW | R2: {ref_r2:.3f}")
    print(f"Current RMSE:   {curr_rmse:.2f} MW | R2: {curr_r2:.3f}")