import React from 'react';

const ReferralCard = ({ referralNeeded, message, reasons }) => {
  // FRONTEND: style this card to highlight if referral is needed
  return (
    <div className="referral-card">
      <h2>Referral Status: {referralNeeded ? "REFERRAL REQUIRED" : "No Referral Needed"}</h2>
      <p>{message}</p>
      <ul>
        {reasons.map((reason, index) => (
          <li key={index}>{reason}</li>
        ))}
      </ul>
    </div>
  );
};

export default ReferralCard;
