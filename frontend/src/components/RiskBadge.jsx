import React from 'react';

const RiskBadge = ({ level }) => {
  // FRONTEND: add styling based on risk level (Low=Green, Mid=Yellow, High=Red)
  return (
    <div className="risk-badge">
      <h3>Risk Level: {level}</h3>
    </div>
  );
};

export default RiskBadge;
