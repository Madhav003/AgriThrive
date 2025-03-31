import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor as k
from sklearn.neighbors import KNeighborsClassifier as kn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error, accuracy_score,r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
import seaborn as sns
# Define SMAPE function
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
# Load dataset
a = pd.read_csv(r'C:\Users\MADHAV BHALODIA\Vscode(py)\hackathon\soil_predictions_full.csv', header=None).dropna().values
x = a[1:1590,1:19]
y = a[1:1590,20:].astype(float)  # Multiple target columns
test_sizes = np.arange(0.1, 0.41, 0.05)
best_test_size = None
best_test_smape = float('inf')
best_k = None

# Store results for plotting
test_size_results = []

# Loop through different test sizes
for test_size in test_sizes:
    xt, xte, yt, yte = train_test_split(x, y, test_size=test_size, random_state=42)

    # Find best k for this test size
    k_values = range(1, 30)
    smape_scores = []

    for k_val in k_values:
        knn = k(n_neighbors=k_val)
        scores = cross_val_score(knn, xt, yt, scoring='neg_mean_absolute_error', cv=5)
        smape_scores.append(abs(scores.mean()))  # Use mean absolute error as a proxy for SMAPE

    # Find best k for this test size
    optimal_k = k_values[np.argmin(smape_scores)]
    optimal_smape = min(smape_scores)

    # Save test size results
    test_size_results.append((test_size, optimal_smape))

    # Update best test size & k if this combination is better
    if optimal_smape < best_test_smape:
        best_test_smape = optimal_smape
        best_test_size = test_size
        best_k = optimal_k

# Print best test size and k
print(f"Optimal test size: {best_test_size*100:.0f}%")
print(f"Optimal value of k: {best_k}")

# Train final model using the best test size and k
xt, xte, yt, yte = train_test_split(x, y, test_size=best_test_size, random_state=42)
knn = k(n_neighbors=best_k)
knn.fit(xt, yt)
yp = knn.predict(xte)

# Print final model evaluation metrics
print("Final Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(yte, yp))
print("Mean Absolute Error:", mean_absolute_error(yte, yp))
print("SMAPE:", smape(yte, yp))

# Plot test size vs. SMAPE
test_sizes_percent = [ts * 100 for ts, _ in test_size_results]
smape_values = [smape for _, smape in test_size_results]

plt.figure(figsize=(6, 4))
plt.plot(test_sizes_percent, smape_values, marker='o', linestyle='-')
plt.xlabel("Test Size (%)")
plt.ylabel("SMAPE (%)")
plt.title("Finding the Best Test Size")
plt.axvline(best_test_size * 100, color='red', linestyle="--", label=f"Best Test Size = {best_test_size*100:.0f}%")
plt.legend()
plt.grid(True)
plt.show()

# Visualization: Actual vs. Predicted, Residuals, and Error Distribution
plt.figure(figsize=(15, 5))

# Actual vs. Predicted
plt.subplot(1, 3, 1)
plt.scatter(yte.flatten(), yp.flatten(), alpha=0.6, color="blue", edgecolors="k")
plt.plot([min(yte.flatten()), max(yte.flatten())], [min(yte.flatten()), max(yte.flatten())], linestyle="--", color="red")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted")
plt.grid(True)

# Residual Plot
residuals = yte-yp
plt.subplot(1, 3, 2)
plt.scatter(yp.flatten(), residuals.flatten(), alpha=0.5, color="purple", edgecolors="k")
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)

# Error Distribution
plt.subplot(1, 3, 3)
plt.hist(residuals.flatten(), color="green", edgecolor="black", bins=20)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution")
plt.show()

df = pd.read_csv(r'C:\Users\MADHAV BHALODIA\Vscode(py)\hackathon\soil_predictions_full.csv', header=None).dropna().values
# Splitting data
X = df[1:,1:19] 
Y = df[1:,19]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Training a Decision Tree Regressor
tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)
# Predicting
y_pred_tree = tree_model.predict(X_test)

# Evaluating the model
mae_tree = mean_absolute_error(y_test, y_pred_tree)
r2_tree = r2_score(y_test, y_pred_tree)
print("Mean Absolute Error:", mae_tree)
print("R2 Score:", r2_tree)
# Plot actual vs. predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_tree, alpha=0.6)
plt.xlabel("Actual Capacitity Moist")
plt.ylabel("Predicted Capacitity Moist")
plt.title(f"Decision Tree Regression (MAE={mae_tree:.2f}, R²={r2_tree:.2f})")
plt.grid(True)
plt.show()

mae_tree, r2_tree

# Train on the entire dataset using KNN
knn_full = k(n_neighbors=best_k)
knn_full.fit(x, y)

# Predict for the entire dataset
y_pred_full = knn_full.predict(x)

# Load the original dataset with record IDs and spectrometer data
df_original = pd.read_csv(r'C:\Users\MADHAV BHALODIA\Vscode(py)\hackathon\soil_predictions_full.csv', header=None).dropna()

# Ensure dimensions match
if len(df_original) != len(y_pred_full) + 1:
    print("Warning: Data dimension mismatch. Check the original dataset format.")
else:
    # Create a new DataFrame with the original data + predictions
    col_names = ['Record'] + [f'Spectro_{i}' for i in range(1, 19)]
    pred_cols = [f'Predicted_{i}' for i in range(y_pred_full.shape[1])]
    
    # Combine original data with predictions
    full_predictions_df = pd.DataFrame(
        np.hstack((df_original.iloc[1:, :19].values, y_pred_full)),
        columns=col_names + pred_cols
    )

    # Insert the 'Records' column at index 0
    full_predictions_df.insert(0, 'Records', df_original.iloc[1:, 0].values)

    # Save to CSV
    output_path = r'C:\Users\MADHAV BHALODIA\Vscode(py)\hackathon\soil_predictions_with_records.csv'
    full_predictions_df.to_csv(output_path, index=False)
    print(f"Full predictions with Records saved to {output_path}")
