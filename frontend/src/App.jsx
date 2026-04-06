import { useState } from 'react'
import './App.css'
import RiskBadge from './components/RiskBadge'
import ReferralCard from './components/ReferralCard'
import PatientHistory from './components/PatientHistory'

function App() {
  const [formData, setFormData] = useState({
    Age: '',
    SystolicBP: '',
    DiastolicBP: '',
    BloodGlucose: '',
    BodyTemp: '',
    HeartRate: ''
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    
    // FRONTEND: replace this fetch URL with actual backend localhost URL
    // e.g., http://localhost:8000/predict
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          Age: parseInt(formData.Age),
          SystolicBP: parseInt(formData.SystolicBP),
          DiastolicBP: parseInt(formData.DiastolicBP),
          BloodGlucose: parseFloat(formData.BloodGlucose),
          BodyTemp: parseFloat(formData.BodyTemp),
          HeartRate: parseInt(formData.HeartRate)
        }),
      })
      const data = await response.json()
      setResult(data)
    } catch (error) {
      console.error('Error calling prediction API:', error)
      alert('Failed to connect to the backend. Please check the console.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <h1>MaternalGuard: Health Risk Screener</h1>
      
      <form onSubmit={handleSubmit} style={{maxWidth: '400px', margin: '20px auto', textAlign: 'left'}}>
        <div className="form-group">
          <label>Age:</label>
          <input type="number" name="Age" value={formData.Age} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Systolic BP:</label>
          <input type="number" name="SystolicBP" value={formData.SystolicBP} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Diastolic BP:</label>
          <input type="number" name="DiastolicBP" value={formData.DiastolicBP} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Blood Glucose (mmol/L):</label>
          <input type="number" step="0.1" name="BloodGlucose" value={formData.BloodGlucose} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Body Temp (F):</label>
          <input type="number" step="0.1" name="BodyTemp" value={formData.BodyTemp} onChange={handleChange} required />
        </div>
        <div className="form-group">
          <label>Heart Rate (bpm):</label>
          <input type="number" name="HeartRate" value={formData.HeartRate} onChange={handleChange} required />
        </div>
        <button type="submit" disabled={loading} style={{marginTop: '20px', width: '100%', padding: '10px'}}>
          {loading ? 'Predicting...' : 'Screen Risk'}
        </button>
      </form>

      {result && (
        <div className="results-section">
          <RiskBadge level={result.risk_level} />
          <ReferralCard 
            referralNeeded={result.referral_needed} 
            message={result.message} 
            reasons={result.shap_reasons} 
          />
        </div>
      )}

      {/* Placeholder history - could be fetched from backend later */}
      <h2 style={{marginTop: '40px'}}>Patient Screen History</h2>
      <PatientHistory history={[]} />
    </div>
  )
}

export default App
