import React from 'react';
import Survey from '@/component/survey';
import './App.css';
import Navbar from '@/component/Navbar';

const App = () => {
  return (
    <div>
      <Navbar />
      <div className='container'>
        <div className="survey-container">
          <h1 className="survey-title">VitalPath Health Survey</h1>
          <h3 className="survey-subtitle">
            Please provide answers for the following questions in the survey to take care of yourself anywhere and anytime.
          </h3>
          <Survey />
        </div>
      </div>
    </div>
  );
};

export default App;