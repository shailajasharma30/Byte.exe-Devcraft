# MaternalGuard Frontend

React application for maternal health risk screening. Built with Vite.

## Local Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Run the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:5173](http://localhost:5173) in your browser.

## Key Components

- `App.jsx`: Main application logic, form handling, and API integration.
- `components/RiskBadge.jsx`: Displays assigned risk level.
- `components/ReferralCard.jsx`: Shows if referral is needed and why (SHAP reasons).
- `components/PatientHistory.jsx`: Future table for patient records.

**Note for Developers:** 
- The form currently sends real-time POST requests to `http://localhost:8000/predict`. 
- Ensure the backend is running before testing the screening functionality.
- UI styling is minimal; please apply the design system here.
