import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# ==============================
# Load and Prepare Data
# ==============================
print("Loading dataset...")
df = pd.read_csv("./facemesh_landmarks_with_pose.csv")
print(f"Dataset shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

# Drop image_index if it exists
if "image_index" in df.columns:
    df = df.drop(columns=["image_index"])

# Define landmark columns
landmark_cols = [
    "nose_x",
    "nose_y",
    "chin_x",
    "chin_y",
    "right_eye_x",
    "right_eye_y",
    "left_eye_x",
    "left_eye_y",
    "mouth_right_x",
    "mouth_right_y",
    "mouth_left_x",
    "mouth_left_y",
    "forehead_x",
    "forehead_y",
]

# ==============================
# Feature Engineering
# ==============================
print("\nApplying feature engineering...")

# 1. Center landmarks around nose
for coord in ["x", "y"]:
    nose_coord = f"nose_{coord}"
    for col in landmark_cols:
        if col.endswith(f"_{coord}") and col != nose_coord:
            df[col] = df[col] - df[nose_coord]

# Set nose to origin
df["nose_x"] = 0.0
df["nose_y"] = 0.0

# 2. Scale normalization
scale_x = abs(df["right_eye_x"] - df["left_eye_x"])
scale_y = abs(df["forehead_y"] - df["nose_y"])

for col in landmark_cols:
    if col.endswith("_x"):
        df[col] /= scale_x
    elif col.endswith("_y"):
        df[col] /= scale_y

print("Feature engineering complete!")

# ==============================
# Prepare Features and Targets
# ==============================
X = df.drop(columns=["pitch", "yaw"])
y = df[["pitch", "yaw"]]

print(f"\nFeature shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nTarget statistics:")
print(y.describe())

# ==============================
# Train-Test Split
# ==============================
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ==============================
# Feature Scaling
# ==============================
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# Train Model
# ==============================
print("\nTraining Random Forest model...")
model = RandomForestRegressor()

model.fit(X_train_scaled, y_train)
print("Training complete!")

# ==============================
# Evaluate Model
# ==============================
print("\nEvaluating model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Training metrics
train_mse = mean_squared_error(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

# Testing metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)
print("\nTraining Set:")
print(f"  Mean Squared Error: {train_mse:.4f}")
print(f"  Mean Absolute Error: {train_mae:.4f}")
print(f"  R² Score: {train_r2:.4f}")

print("\nTest Set:")
print(f"  Mean Squared Error: {test_mse:.4f}")
print(f"  Mean Absolute Error: {test_mae:.4f}")
print(f"  R² Score: {test_r2:.4f}")

# Per-target metrics
print("\nPer-Target Test Performance:")
for i, target in enumerate(["pitch", "yaw"]):
    target_mse = mean_squared_error(y_test.iloc[:, i], y_pred_test[:, i])
    target_mae = mean_absolute_error(y_test.iloc[:, i], y_pred_test[:, i])
    target_r2 = r2_score(y_test.iloc[:, i], y_pred_test[:, i])
    print(f"\n  {target.upper()}:")
    print(f"    MSE: {target_mse:.4f}")
    print(f"    MAE: {target_mae:.4f}")
    print(f"    R²: {target_r2:.4f}")

# ==============================
# Feature Importance
# ==============================
print("\n" + "=" * 50)
print("FEATURE IMPORTANCE")
print("=" * 50)

# Get feature importance for pitch and yaw separately
feature_names = X.columns.tolist()
importance_pitch = model.estimators_[0].feature_importances_  # First output (pitch)
importance_yaw = model.estimators_[1].feature_importances_  # Second output (yaw)

# Create DataFrame for better visualization
importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Pitch_Importance": importance_pitch,
        "Yaw_Importance": importance_yaw,
        "Average_Importance": (importance_pitch + importance_yaw) / 2,
    }
)

importance_df = importance_df.sort_values("Average_Importance", ascending=False)
print("\nTop 10 Most Important Features:")
print(importance_df.head(10))

# ==============================
# Visualization (Optional)
# ==============================
try:
    print("\nGenerating visualizations...")

    # Plot 1: Actual vs Predicted for Pitch
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_test["pitch"], y_pred_test[:, 0], alpha=0.5)
    axes[0].plot(
        [y_test["pitch"].min(), y_test["pitch"].max()],
        [y_test["pitch"].min(), y_test["pitch"].max()],
        "r--",
        lw=2,
    )
    axes[0].set_xlabel("Actual Pitch")
    axes[0].set_ylabel("Predicted Pitch")
    axes[0].set_title(
        f"Pitch Prediction (R² = {r2_score(y_test['pitch'], y_pred_test[:, 0]):.3f})"
    )
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted for Yaw
    axes[1].scatter(y_test["yaw"], y_pred_test[:, 1], alpha=0.5)
    axes[1].plot(
        [y_test["yaw"].min(), y_test["yaw"].max()],
        [y_test["yaw"].min(), y_test["yaw"].max()],
        "r--",
        lw=2,
    )
    axes[1].set_xlabel("Actual Yaw")
    axes[1].set_ylabel("Predicted Yaw")
    axes[1].set_title(
        f"Yaw Prediction (R² = {r2_score(y_test['yaw'], y_pred_test[:, 1]):.3f})"
    )
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("model_predictions.png", dpi=150)
    print("Saved visualization to 'model_predictions.png'")

    # Plot 3: Feature Importance
    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(10)
    plt.barh(range(len(top_features)), top_features["Average_Importance"])
    plt.yticks(range(len(top_features)), top_features["Feature"])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    print("Saved feature importance to 'feature_importance.png'")

except Exception as e:
    print(f"Could not generate visualizations: {e}")

# ==============================
# Save Model and Scaler
# ==============================
print("\n" + "=" * 50)
print("SAVING MODEL")
print("=" * 50)

joblib.dump(model, "trained_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as: trained_model.pkl")
print("Scaler saved as: scaler.pkl")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print("\nYou can now use 'trained_model.pkl' and 'scaler.pkl'")
print("in your head pose estimation application.")
