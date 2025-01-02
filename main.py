from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle

# Load the Gradient Boosting Regressor model
with open('gradient_boosting_regressor_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Initialize FastAPI
app = FastAPI()

# Define the input data model
class InsuranceData(BaseModel):
    age: int
    sex: str  # 'male' or 'female'
    bmi: float
    children: int
    smoker: str  # 'yes' or 'no'
    region: str  # 'northeast', 'northwest', 'southeast', 'southwest'

# Create a prediction endpoint
@app.post("/predict")
def predict(data: InsuranceData):
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Preprocess the data
    input_data['sex'] = input_data['sex'].map({'male': 1, 'female': 0})
    input_data['smoker'] = input_data['smoker'].map({'yes': 1, 'no': 0})
    input_data = pd.get_dummies(input_data, columns=['region'], drop_first=True)

    # Ensure all columns are present
    expected_columns = ['age', 'sex', 'bmi', 'children', 'smoker',
                        'region_northwest', 'region_southeast', 'region_southwest']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Fill missing columns with 0

    # Reorder the columns
    input_data = input_data[expected_columns]

    # Make prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Return the prediction rounded to two decimal places
    return {
        "predicted_insurance_cost": round(prediction, 2)
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
