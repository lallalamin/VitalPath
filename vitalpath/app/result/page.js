"use client"
import React, { useState } from 'react'


const Page = () => {
    const [prediction, setPrediction] = useState(null);

const handleSubmit = async (e) => {
  e.preventDefault();

  try {
    const response = await fetch('/api/predictDisease', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(formData),
    });

    const data = await response.json();
    setPrediction(data.disease_risk);  // Set the prediction result
  } catch (error) {
    console.error('Error making prediction:', error);
  }
};

return (
  <div>
    {/* Your form goes here */}
    {prediction && (
      <div>
        <h3>Prediction Result: {prediction}</h3>
      </div>
    )}
  </div>
);

}

export default Page;