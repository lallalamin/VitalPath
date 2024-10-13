"use client";

import React, { useState } from "react";
import {
  CardContent,
  CardHeader,
  Typography,
  Button,
  Slider,
  Box,
  Paper,
  LinearProgress,
} from "@mui/material";

const VitalPathSurvey = () => {
  const [currentSection, setCurrentSection] = useState(0);
  const [formData, setFormData] = useState({
    physicalActivity: 0,
    vigorousExercise: 0,
    physicalActivityDays: 0,
    sittingTime: 0,
    eatingHabits: 0,
    feelingDown: 0,
    sleepWeekdays: 0,
    sleepWeekends: 0,
  });

  const totalSections = 4;

  const handleChange = (e, value) => {
    const { name } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const nextSection = () => setCurrentSection(currentSection + 1);
  const prevSection = () => setCurrentSection(currentSection - 1);

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);
  };

  const renderSection = () => {
    return (
      <Paper elevation={3} sx={{ width: '100%', borderRadius: '8px' }}>
        <CardHeader
          title={
            <Typography variant="h5" component="div" sx={{ mb: 1 }}>
              {currentSection === 0 && "Physical Activity"}
              {currentSection === 1 && "Nutrition"}
              {currentSection === 2 && "Physical Activity (Continued)"}
              {currentSection === 3 && "Mental Health and Sleep"}
            </Typography>
          }
          subheader={
            <Typography variant="body2">
              {currentSection === 0 && "Let's start by looking at your physical activity habits."}
              {currentSection === 1 && "Next, let's talk about your eating habits."}
              {currentSection === 2 && "Let's continue looking at your physical activities."}
              {currentSection === 3 && "Lastly, let's check in on your mental health and sleep habits."}
            </Typography>
          }
          sx={{ pb: 0 }}
        />
        <CardContent sx={{ pt: 1 }}>
          {currentSection === 0 && (
            <>
              <SliderQuestion
                question="How often do you engage in physical activities?"
                name="physicalActivity"
                value={formData.physicalActivity}
                onChange={handleChange}
              />
              <SliderQuestion
                question="How often do you do vigorous-intensity leisure activities?"
                name="vigorousExercise"
                value={formData.vigorousExercise}
                onChange={handleChange}
              />
              <SliderQuestion
                question="How many days per week do you perform physical activities?"
                name="physicalActivityDays"
                value={formData.physicalActivityDays}
                onChange={handleChange}
              />
              <SliderQuestion
                question="On average, how many hours do you spend sitting each day?"
                name="sittingTime"
                value={formData.sittingTime}
                onChange={handleChange}
              />
            </>
          )}
          {currentSection === 1 && (
            <SliderQuestion
              question="On a scale of 1 to 10, how often do you eat fruits and vegetables?"
              name="eatingHabits"
              value={formData.eatingHabits}
              onChange={handleChange}
            />
          )}
          {currentSection === 2 && (
            <SliderQuestion
              question="How often do you engage in vigorous-intensity leisure-time activities?"
              name="vigorousExercise"
              value={formData.vigorousExercise}
              onChange={handleChange}
            />
          )}
          {currentSection === 3 && (
            <>
              <SliderQuestion
                question="How often have you felt down, depressed, or hopeless?"
                name="feelingDown"
                value={formData.feelingDown}
                onChange={handleChange}
              />
              <SliderQuestion
                question="How many hours of sleep do you get on weekdays?"
                name="sleepWeekdays"
                value={formData.sleepWeekdays}
                onChange={handleChange}
              />
              <SliderQuestion
                question="How many hours of sleep do you get on weekends?"
                name="sleepWeekends"
                value={formData.sleepWeekends}
                onChange={handleChange}
              />
            </>
          )}
        </CardContent>
        <Box display="flex" justifyContent="space-between" px={2} pb={2}>
          {currentSection > 0 && (
            <Button variant="contained" color="secondary" onClick={prevSection}>
              Back
            </Button>
          )}
          {currentSection < totalSections - 1 ? (
            <Button variant="contained" color="primary" onClick={nextSection}>
              Next
            </Button>
          ) : (
            <Button variant="contained" color="primary" onClick={handleSubmit}>
              Submit
            </Button>
          )}
        </Box>
        <LinearProgress
          variant="determinate"
          value={(currentSection + 1) * (100 / totalSections)}
        />
      </Paper>
    );
  };

  const SliderQuestion = ({ question, name, value, onChange }) => (
    <Box mb={2}>
      <Typography variant="body2" gutterBottom>
        {question}
      </Typography>
      <Slider
        name={name}
        value={value}
        onChange={onChange}
        step={1}
        min={1}
        max={10}
        valueLabelDisplay="auto"
        sx={{
          color: "#3f51b5",
          "& .MuiSlider-thumb": {
            transition: "0.3s",
            "&:hover": { transform: "scale(1.2)" },
          },
        }}
      />
    </Box>
  );

  return (
    <Box
      display="flex"
      justifyContent="center"
      alignItems="flex-start"
      width="100%"
      maxWidth="600px"
      margin="0 auto"
    >
      {renderSection()}
    </Box>
  );
};

export default VitalPathSurvey;