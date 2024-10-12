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
      <Box sx={{ textAlign: 'center' }}>
        <Typography variant="h2" sx={{ mb: 2 }}>
          Welcome to VitalPath
        </Typography>
        <Typography variant="body1" sx={{ mb: 4 }}>
          Your one-stop solution for flashcard learning!        
        </Typography>
      </Box>
    </div>
  );
};

export default Hero;
