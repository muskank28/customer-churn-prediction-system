from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(PROJECT_ROOT, 'frontend', 'build')
STATIC_DIR = os.path.join(BUILD_DIR, 'static')

app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=BUILD_DIR
)

CORS(app) 

try:
    rf_model = joblib.load(os.path.join(PROJECT_ROOT, "model/random_forest_model.pkl"))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, "model/scaler.pkl"))
    with open(os.path.join(PROJECT_ROOT, "model/columns.json"), "r") as f:
        final_columns = json.load(f)
    print("Model and preprocessing assets loaded successfully.")
except Exception as e:
    print(f"ERROR LOADING ML ASSETS: {e}")

numeric_cols = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", 
             "Total Spend", "Last Interaction"]
categorical_cols = ["Gender", "Subscription Type", "Contract Length"]


def preprocess_input(data):
    df = pd.DataFrame([data])
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df = df.reindex(columns=final_columns, fill_value=0)
    return df


# API ROUTES FIRST (before catch-all routes)
@app.route("/api/dashboard-data", methods=["GET"])
def get_dashboard_data():
    try:
        # Load your customer data
        df = pd.read_csv(os.path.join(PROJECT_ROOT, "customer_churn_dataset-testing-master.csv"))
        
        # Basic metrics
        total_customers = len(df)
        churned = df['Churn'].sum()
        churn_rate = churned / total_customers if total_customers > 0 else 0
        active_customers = total_customers - churned
        
        # TRENDS DATA - Monthly breakdown (simulated from Last Interaction)
        df['month_group'] = pd.cut(df['Last Interaction'], bins=6, labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
        
        trends_data = []
        for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']:
            month_df = df[df['month_group'] == month]
            trends_data.append({
                "month": month,
                "Existing": int(len(month_df[month_df['Churn'] == 0])),  # Active customers
                "New": int(len(month_df) * 0.3),  # Simulated new customers (30% of total)
                "Lost": int(month_df['Churn'].sum()),  # Churned customers
                "Active": int(len(month_df[month_df['Churn'] == 0]))
            })
        
        # CHURN DATA - By different categories
        churn_data = []
        
        # By Subscription Type
        for sub_type in df['Subscription Type'].unique():
            sub_df = df[df['Subscription Type'] == sub_type]
            churn_data.append({
                "category": sub_type,
                "churned": int(sub_df['Churn'].sum()),
                "active": int(len(sub_df[sub_df['Churn'] == 0]))
            })
        
        # By Contract Length
        for contract in df['Contract Length'].unique():
            contract_df = df[df['Contract Length'] == contract]
            churn_data.append({
                "category": f"Contract-{contract}",
                "churned": int(contract_df['Churn'].sum()),
                "active": int(len(contract_df[contract_df['Churn'] == 0]))
            })
        
        # By Gender
        for gender in df['Gender'].unique():
            gender_df = df[df['Gender'] == gender]
            churn_data.append({
                "category": gender,
                "churned": int(gender_df['Churn'].sum()),
                "active": int(len(gender_df[gender_df['Churn'] == 0]))
            })
        
        dashboard_data = {
            "trends": trends_data,  # This is what your React component expects!
            "churn": churn_data,    # This is what your React component expects!
            
            # Additional metrics (for future use)
            "total_customers": int(total_customers),
            "churned_customers": int(churned),
            "churn_rate": float(churn_rate),
            "active_customers": int(active_customers),
            "total_revenue": round(float(df['Total Spend'].sum()), 2)
        }
        
        print("✅ Dashboard data prepared successfully")
        return jsonify(dashboard_data)
        
    except Exception as e:
        app.logger.error(f"Dashboard data fetch failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "trends": [],  # Return empty arrays to prevent crashes
            "churn": []
        }), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        print("Received data for prediction:", input_data)

        processed_data = preprocess_input(input_data)

        prediction = rf_model.predict(processed_data)[0]
        probability = rf_model.predict_proba(processed_data)[0][1] 

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        })

    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({
            "error": "Prediction processing failed. Check input data or model files.",
            "details": str(e)
        }), 500


# REACT APP SERVING ROUTES (at the end)
@app.route('/')
def serve_react_app():
    try:
        return send_from_directory(app.template_folder, 'index.html')
    except FileNotFoundError:
        return "React build not found. Please run 'npm run build' in the frontend directory.", 500


@app.route('/<path:path>')
def catch_all(path):
    if os.path.exists(os.path.join(app.template_folder, path)):
        return send_from_directory(app.template_folder, path)
    
    return send_from_directory(app.template_folder, 'index.html')


if __name__ == "__main__":
    app.run(debug=True, port=5000)