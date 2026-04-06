# MaternalGuard Backend

FastAPI service for maternal health risk prediction.

## Local Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   python main.py
   ```
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload
   ```

The API will be available at [http://localhost:8000](http://localhost:8000).
Check the interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

## Prediction Logic
ML logic is located in `predict_risk` endpoint in `main.py`. 
Models should be stored in the `model/` directory.
Prediction inputs: Age, SystolicBP, DiastolicBP, BloodGlucose, BodyTemp, HeartRate.
