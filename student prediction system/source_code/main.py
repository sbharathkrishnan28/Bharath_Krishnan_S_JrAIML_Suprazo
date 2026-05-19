"""
=============================================================
  STUDENT PERFORMANCE PREDICTION SYSTEM USING MACHINE LEARNING
=============================================================
  Algorithm: Linear Regression
  Dataset: 5000 Student Records
  Features: StudyHours, Attendance, SleepHours, PreviousScore
  Target: FinalScore
=============================================================
"""

# ============================================================
# STEP 1: Import Required Libraries
# ============================================================
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Create screenshots directory if it doesn't exist
screenshots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'screenshots')
os.makedirs(screenshots_dir, exist_ok=True)

print("=" * 60)
print("  STUDENT PERFORMANCE PREDICTION SYSTEM")
print("  Using Machine Learning (Linear Regression)")
print("=" * 60)

# ============================================================
# STEP 2: Load Dataset
# ============================================================
print("\n" + "=" * 60)
print("  STEP 1: LOADING DATASET")
print("=" * 60)

dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'student_performance_dataset.csv')
df = pd.read_csv(dataset_path)

print(f"\n✅ Dataset loaded successfully!")
print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📋 Column Names: {list(df.columns)}")
print(f"\n📝 Data Types:")
print(df.dtypes)
print(f"\n🔍 First 10 Records:")
print(df.head(10).to_string(index=False))

# Save dataset output screenshot
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
ax.set_title('Student Performance Dataset - First 20 Records', fontsize=16, fontweight='bold', pad=20)
table = ax.table(
    cellText=df.head(20).values,
    colLabels=df.columns,
    cellLoc='center',
    loc='center',
    colColours=['#4CAF50'] * len(df.columns)
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)
# Style header
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_facecolor('#2E7D32')
    elif row % 2 == 0:
        cell.set_facecolor('#E8F5E9')
    else:
        cell.set_facecolor('#FFFFFF')
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'dataset_output.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"\n📸 Dataset output screenshot saved!")

# ============================================================
# STEP 3: Data Cleaning & Preprocessing
# ============================================================
print("\n" + "=" * 60)
print("  STEP 2: DATA CLEANING & PREPROCESSING")
print("=" * 60)

# Check for missing values
print(f"\n🔍 Missing Values:")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing values: {missing.sum()}")

# Check for duplicates
duplicates = df.duplicated().sum()
print(f"\n🔍 Duplicate Records: {duplicates}")

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"✅ Removed {duplicates} duplicate records")

# Dataset statistics
print(f"\n📊 Dataset Statistics:")
print(df.describe().to_string())

print(f"\n✅ Data cleaning completed!")

# ============================================================
# STEP 4: Data Visualization
# ============================================================
print("\n" + "=" * 60)
print("  STEP 3: DATA VISUALIZATION")
print("=" * 60)

# --- 4.1: Scatter Plot (Study Hours vs Final Score) ---
print("\n📈 Generating Scatter Plot (Study Hours vs Final Score)...")
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(df['StudyHours'], df['FinalScore'],
                     c=df['Attendance'], cmap='viridis',
                     alpha=0.5, edgecolors='white', linewidths=0.5, s=30)
ax.set_xlabel('Study Hours (per day)', fontsize=14, fontweight='bold')
ax.set_ylabel('Final Score', fontsize=14, fontweight='bold')
ax.set_title('Study Hours vs Final Score\n(Color = Attendance %)', fontsize=16, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Attendance (%)', fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'scatterplot.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Scatter plot saved!")

# --- 4.2: Correlation Heatmap ---
print("\n📊 Generating Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))
correlation = df.corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
heatmap = sns.heatmap(correlation, annot=True, fmt='.3f', cmap='RdYlGn',
                      mask=mask, center=0, square=True,
                      linewidths=2, linecolor='white',
                      cbar_kws={'shrink': 0.8},
                      annot_kws={'size': 14, 'weight': 'bold'},
                      ax=ax)
ax.set_title('Correlation Heatmap\nStudent Performance Attributes', fontsize=16, fontweight='bold', pad=20)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Heatmap saved!")

# --- 4.3: Distribution Histograms ---
print("\n📊 Generating Distribution Histograms...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Data Distribution - Student Performance Dataset', fontsize=18, fontweight='bold', y=1.02)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']
for idx, (col, color) in enumerate(zip(df.columns, colors)):
    row, col_idx = divmod(idx, 3)
    axes[row][col_idx].hist(df[col], bins=30, color=color, edgecolor='white', alpha=0.8)
    axes[row][col_idx].set_title(col, fontsize=14, fontweight='bold')
    axes[row][col_idx].set_xlabel('Value', fontsize=11)
    axes[row][col_idx].set_ylabel('Frequency', fontsize=11)
    axes[row][col_idx].grid(axis='y', alpha=0.3)

# Remove empty subplot
axes[1][2].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Histograms saved!")

# --- 4.4: Pair Plot ---
print("\n📊 Generating Pair Plot...")
pair_plot = sns.pairplot(df, diag_kind='kde',
                         plot_kws={'alpha': 0.3, 's': 15, 'edgecolor': 'white', 'linewidth': 0.3},
                         diag_kws={'fill': True})
pair_plot.fig.suptitle('Pair Plot - All Attributes', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'pairplot.png'), dpi=100, bbox_inches='tight')
plt.close()
print("✅ Pair plot saved!")

print("\n✅ All visualizations generated and saved to 'screenshots/' folder!")

# ============================================================
# STEP 5: Feature Selection & Train-Test Split
# ============================================================
print("\n" + "=" * 60)
print("  STEP 4: TRAIN-TEST SPLIT")
print("=" * 60)

# Define features (X) and target (y)
X = df[['StudyHours', 'Attendance', 'SleepHours', 'PreviousScore']]
y = df['FinalScore']

print(f"\n📌 Features (X): {list(X.columns)}")
print(f"📌 Target (y): FinalScore")
print(f"📌 Total samples: {len(df)}")

# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📊 Training Data: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.0f}%)")
print(f"📊 Testing Data:  {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.0f}%)")

# ============================================================
# STEP 6: Model Training
# ============================================================
print("\n" + "=" * 60)
print("  STEP 5: MODEL TRAINING (Linear Regression)")
print("=" * 60)

# Initialize and train model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n✅ Linear Regression model trained successfully!")
print(f"\n📐 Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature}: {coef:.4f}")
print(f"   Intercept: {model.intercept_:.4f}")

# ============================================================
# STEP 7: Prediction
# ============================================================
print("\n" + "=" * 60)
print("  STEP 6: PREDICTION")
print("=" * 60)

# Predict on test data
y_pred = model.predict(X_test)

# Display predictions comparison
comparison = pd.DataFrame({
    'Actual Score': y_test.values[:15],
    'Predicted Score': np.round(y_pred[:15], 2),
    'Difference': np.round(np.abs(y_test.values[:15] - y_pred[:15]), 2)
})
print(f"\n📋 Prediction Results (First 15 Samples):")
print(comparison.to_string(index=False))

# ============================================================
# STEP 8: Performance Evaluation
# ============================================================
print("\n" + "=" * 60)
print("  STEP 7: PERFORMANCE EVALUATION")
print("=" * 60)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\n📊 Performance Metrics:")
print(f"   ├── Mean Absolute Error (MAE):  {mae:.4f}")
print(f"   ├── Mean Squared Error (MSE):   {mse:.4f}")
print(f"   ├── Root Mean Squared Error:    {rmse:.4f}")
print(f"   └── R² Score:                   {r2:.4f} ({r2*100:.2f}%)")

if r2 >= 0.9:
    print(f"\n🌟 Excellent model performance! R² = {r2:.4f}")
elif r2 >= 0.7:
    print(f"\n✅ Good model performance! R² = {r2:.4f}")
elif r2 >= 0.5:
    print(f"\n⚠️ Moderate model performance. R² = {r2:.4f}")
else:
    print(f"\n❌ Poor model performance. R² = {r2:.4f}")

# --- Prediction Result Visualization ---
print("\n📊 Generating Prediction Results Plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.4, color='#2196F3', edgecolors='white', linewidths=0.5, s=30)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', lw=2, label='Perfect Prediction Line')
axes[0].set_xlabel('Actual Score', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Predicted Score', fontsize=14, fontweight='bold')
axes[0].set_title('Actual vs Predicted Scores', fontsize=16, fontweight='bold')
axes[0].legend(fontsize=12)
axes[0].grid(True, alpha=0.3)

# Add metrics text box
metrics_text = f'MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}'
axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Residual Distribution
residuals = y_test - y_pred
axes[1].hist(residuals, bins=40, color='#4CAF50', edgecolor='white', alpha=0.8)
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error Line')
axes[1].set_xlabel('Prediction Error (Residual)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=14, fontweight='bold')
axes[1].set_title('Residual Distribution', fontsize=16, fontweight='bold')
axes[1].legend(fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

fig.suptitle('Model Performance - Linear Regression', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(screenshots_dir, 'prediction_result.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Prediction result plot saved!")

# ============================================================
# STEP 9: Custom Prediction
# ============================================================
print("\n" + "=" * 60)
print("  STEP 8: CUSTOM PREDICTION DEMO")
print("=" * 60)

# Example predictions
sample_students = pd.DataFrame({
    'StudyHours': [8.0, 3.0, 6.5, 10.0, 2.0],
    'Attendance': [90, 50, 75, 95, 40],
    'SleepHours': [7.0, 5.0, 6.5, 8.0, 4.0],
    'PreviousScore': [85, 45, 65, 92, 35]
})

predictions = model.predict(sample_students)

print(f"\n🎯 Custom Predictions:")
print(f"{'─' * 65}")
print(f"{'Student':^10} {'StudyHrs':^10} {'Attend%':^10} {'Sleep':^8} {'PrevScore':^10} {'Predicted':^10}")
print(f"{'─' * 65}")
for i in range(len(sample_students)):
    print(f"{'Student ' + str(i+1):^10} "
          f"{sample_students.iloc[i]['StudyHours']:^10.1f} "
          f"{sample_students.iloc[i]['Attendance']:^10.0f} "
          f"{sample_students.iloc[i]['SleepHours']:^8.1f} "
          f"{sample_students.iloc[i]['PreviousScore']:^10.0f} "
          f"{predictions[i]:^10.2f}")
print(f"{'─' * 65}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("  PROJECT SUMMARY")
print("=" * 60)
print(f"""
  📁 Dataset:       {len(df)} student records
  📊 Features:      StudyHours, Attendance, SleepHours, PreviousScore
  🎯 Target:        FinalScore
  🤖 Algorithm:     Linear Regression
  📈 R² Score:      {r2:.4f} ({r2*100:.2f}%)
  📉 MAE:           {mae:.4f}
  📉 MSE:           {mse:.4f}
  📂 Screenshots:   Saved to 'screenshots/' folder

  ✅ Project completed successfully!
""")
print("=" * 60)
