# MaternalGuard: AI-Powered Health Risk Screener

MaternalGuard is an AI-powered tool designed for rural ASHA workers in India to identify maternal health risks early. 
By analyzing 6 key patient vitals, it identifies high-risk cases for immediate referral.

## Project Structure

- `backend/`: FastAPI-based ML inference service (Python).
- `frontend/`: React-Vite dashboard for screening (JavaScript).
- `design/`: Figma designs and asset exports.
- `presentation/`: Submission presentation.

## Tech Stack

- **Frontend**: React, Vite, Vanilla CSS.
- **Backend**: FastAPI, XGBoost, SHAP.
- **Tools**: GitHub, Figma.

## How to Run

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Features

1. **Risk Prediction**: Fast assessment based on 6 patient vitals.
2. **SHAP Integration**: Explains the "why" behind mid and high risk results.
3. **ASHA Workflow**: Designed for ease of use in rural settings.

---
**GitHub Repository**: https://github.com/dptel22/Byte.exe-Devcraft.git
**Submission for Byte.exe Devcraft Hackathon**
