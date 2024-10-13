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
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Typography variant="h2" sx={{ mb: 5, color: '#0096FF' }} >
          WELCOME
        </Typography>
        <Typography variant="h3" sx={{ mb: 4 }}>
          Review your daily habits with us!        
        </Typography>
        <Typography variant="body1" sx={{ mb: 5 }}>
          We are going to help you asses your habits and the way you currently<br />
          feel using open source data from medical research in just a view minutes.
        </Typography>
      </Box>
    </div>
  );
};

export default Hero;
