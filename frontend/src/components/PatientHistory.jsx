import React from 'react';

const PatientHistory = ({ history }) => {
  // FRONTEND: render a table with previous risk assessments
  return (
    <div className="patient-history">
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Vitals Summary</th>
            <th>Risk Level</th>
          </tr>
        </thead>
        <tbody>
          {history.length > 0 ? (
            history.map((record, index) => (
              <tr key={index}>
                <td>{record.date}</td>
                <td>{record.vitals_summary}</td>
                <td>{record.risk_level}</td>
              </tr>
            ))
          ) : (
            <tr>
              <td colSpan="3">No history records found.</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
};

export default PatientHistory;
