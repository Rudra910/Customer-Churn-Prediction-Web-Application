from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

app = Flask(__name__)

# ======================
# Train Model
# ======================

train_ds = pd.read_csv('ecommerce_churn_5000_rows.csv')

numeric_features = ['Age','Total_Spend','Average_Order_Value',
                    'Purchase_Frequency','Last_Purchase_Days',
                    'Customer_Rating','Return_Count']

categorical_features = ['Gender', 'Complaint_Raised']

X = train_ds[numeric_features + categorical_features]
Y = train_ds['Churn']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=3000))
])

model.fit(X, Y)

# ======================
# Routes
# ======================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_data = pd.DataFrame([{
        'Age': int(request.form['age']),
        'Total_Spend': float(request.form['total_spend']),
        'Average_Order_Value': float(request.form['avg_order']),
        'Purchase_Frequency': int(request.form['purchase_freq']),
        'Last_Purchase_Days': int(request.form['last_purchase']),
        'Customer_Rating': float(request.form['rating']),
        'Return_Count': int(request.form['return_count']),
        'Gender': request.form['gender'],
        'Complaint_Raised': request.form['complaint']
    }])

    prediction = model.predict(user_data)[0]
    probability = model.predict_proba(user_data)[0][1] * 100

    result = "High Risk of Churn" if prediction == 1 else "Low Risk of Churn"

    return render_template("index.html",
                           prediction_text=result,
                           probability=round(probability,2))

if __name__ == "__main__":
    app.run(debug=True)
