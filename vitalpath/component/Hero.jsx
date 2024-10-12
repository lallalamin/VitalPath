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
        <div>
          <Typography
            variant="h3"
            component="h1"
            gutterBottom
          >
            Master your learning with lightning-fast flashcards
          </Typography>
          <Typography variant="h5" component="h2" gutterBottom>
            The easiest way to create flashcards from your text.
          </Typography>
        </div>
        <Button
          className="button-white"
          variant="contained"
          color="primary"
          sx={{
            mt: 2,
            mr: 2,
            backgroundColor: 'white',
            color: 'black',
            fontWeight: 600,
            borderRadius: '10px',
            padding: '5px 15px',
            marginLeft: '10px',
            '&:hover': { backgroundColor: '#e2e2e2' },
          }}
          href="/generate"
        >
          Get Started
        </Button>
        <Button
          className="button-blue"
          variant="outlined"
          color="primary"
          sx={{
            mt: 2,
            backgroundColor: '#2E46CD',
            color: 'white',
            fontWeight: 600,
            borderRadius: '10px',
            padding: '5px 15px',
            marginLeft: '10px',
            '&:hover': { backgroundColor: '#1565C0' },
          }}
        >
          Learn More
        </Button>
      </Box>
    </div>
  );
};

export default Hero;
