"""
Student Performance Prediction System - Flask Dashboard
Interactive web application for predicting student academic performance.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import json

app = Flask(__name__)

# ── Global variables ──
model = None
df = None
metrics = {}
coefficients = {}
X_test_global = None
y_test_global = None
y_pred_global = None


def load_and_train():
    """Load dataset and train the model on startup."""
    global model, df, metrics, coefficients, X_test_global, y_test_global, y_pred_global

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'dataset', 'student_performance_dataset.csv')
    df = pd.read_csv(dataset_path)

    X = df[['StudyHours', 'Attendance', 'SleepHours', 'PreviousScore']]
    y = df['FinalScore']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_global = X_test
    y_test_global = y_test

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_global = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mae': round(mae, 4),
        'mse': round(mse, 4),
        'rmse': round(np.sqrt(mse), 4),
        'r2': round(r2, 4),
        'r2_pct': round(r2 * 100, 2),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'total_size': len(df)
    }

    coefficients = {
        'features': dict(zip(X.columns.tolist(), [round(c, 4) for c in model.coef_])),
        'intercept': round(model.intercept_, 4)
    }

    print(f"Model trained! R2={metrics['r2']}, MAE={metrics['mae']}")


# ── Routes ──

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Return dataset statistics and model metrics."""
    stats = {
        'dataset': {
            'total_records': len(df),
            'features': 4,
            'missing_values': int(df.isnull().sum().sum()),
            'duplicates': int(df.duplicated().sum())
        },
        'metrics': metrics,
        'coefficients': coefficients,
        'feature_stats': json.loads(df.describe().to_json()),
        'correlation': json.loads(df.corr().to_json())
    }
    return jsonify(stats)


@app.route('/api/dataset')
def get_dataset():
    """Return paginated dataset."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    search = request.args.get('search', '', type=str)

    filtered = df.copy()
    if search:
        mask = filtered.apply(lambda row: search.lower() in str(row.values).lower(), axis=1)
        filtered = filtered[mask]

    total = len(filtered)
    start = (page - 1) * per_page
    end = start + per_page
    data = filtered.iloc[start:end]

    return jsonify({
        'data': data.to_dict('records'),
        'total': total,
        'page': page,
        'pages': (total + per_page - 1) // per_page
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction with the trained model."""
    data = request.get_json()
    try:
        study_hours = float(data['study_hours'])
        attendance = float(data['attendance'])
        sleep_hours = float(data['sleep_hours'])
        previous_score = float(data['previous_score'])

        features = np.array([[study_hours, attendance, sleep_hours, previous_score]])
        prediction = model.predict(features)[0]
        prediction = round(min(max(prediction, 0), 100), 2)

        # Determine performance level
        if prediction >= 90:
            level = 'Excellent'
            color = '#10b981'
        elif prediction >= 75:
            level = 'Good'
            color = '#3b82f6'
        elif prediction >= 60:
            level = 'Average'
            color = '#f59e0b'
        else:
            level = 'Needs Improvement'
            color = '#ef4444'

        return jsonify({
            'success': True,
            'prediction': prediction,
            'level': level,
            'color': color,
            'contributions': {
                'StudyHours': round(study_hours * coefficients['features']['StudyHours'], 2),
                'Attendance': round(attendance * coefficients['features']['Attendance'], 2),
                'SleepHours': round(sleep_hours * coefficients['features']['SleepHours'], 2),
                'PreviousScore': round(previous_score * coefficients['features']['PreviousScore'], 2),
                'Intercept': coefficients['intercept']
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/charts/scatter')
def chart_scatter():
    """Return scatter plot data."""
    sample = df.sample(min(500, len(df)), random_state=42)
    return jsonify({
        'x': sample['StudyHours'].tolist(),
        'y': sample['FinalScore'].tolist(),
        'colors': sample['Attendance'].tolist()
    })


@app.route('/api/charts/distribution')
def chart_distribution():
    """Return distribution data for all features."""
    result = {}
    for col in df.columns:
        hist, edges = np.histogram(df[col], bins=25)
        result[col] = {
            'counts': hist.tolist(),
            'edges': [round(e, 2) for e in edges.tolist()]
        }
    return jsonify(result)


@app.route('/api/charts/actual_vs_predicted')
def chart_actual_predicted():
    """Return actual vs predicted comparison."""
    return jsonify({
        'actual': y_test_global.tolist(),
        'predicted': [round(p, 2) for p in y_pred_global.tolist()]
    })


@app.route('/api/charts/residuals')
def chart_residuals():
    """Return residual distribution."""
    residuals = (y_test_global.values - y_pred_global).tolist()
    hist, edges = np.histogram(residuals, bins=30)
    return jsonify({
        'counts': hist.tolist(),
        'edges': [round(e, 2) for e in edges.tolist()]
    })


@app.route('/api/charts/feature_importance')
def chart_feature_importance():
    """Return feature importance based on coefficients."""
    features = list(coefficients['features'].keys())
    values = list(coefficients['features'].values())
    return jsonify({'features': features, 'values': values})


if __name__ == '__main__':
    load_and_train()
    app.run(debug=True, port=5000)
