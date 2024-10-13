import React from 'react';
import { Box, Typography, Button } from '@mui/material'; // Importing required Material UI components

const Hero = () => {
  return (
    <div
      id="header"
      style={{
        height: '100vh',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Box sx={{ textAlign: 'center', mt: 5 }}>
        <Typography variant="h2" sx={{ mb: 5, color: '#0096FF' }} >
          WELCOME
        </Typography>
        <Typography variant="h3" sx={{ mb: 4 }}>
          Review your health status with us!        
        </Typography>
        <Typography variant="h5" sx={{ mb: 5 }}>
          We are going to help you evaluate your habits<br /> 
          using open source data from medical research in just a few minutes.
        </Typography>
      </Box>
    </div>
    
  );
};

export default Hero;
