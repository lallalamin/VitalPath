import axios from "axios";

export default async function handler(req, res) {
    if (req.method === 'POST') {
      try {
        // Send the user's form data to the Python backend
        const { data } = await axios.post('https://vitalpath1-bfb4b0d0f4fc.herokuapp.com/', req.body);
        
        // Respond with the prediction result from the Python model
        res.status(200).json(data);
      } catch (error) {
        console.error('Error calling Python model:', error);
        res.status(500).json({ error: 'Error calling Python model' });
      }
    } else {
      res.status(405).json({ message: 'Method Not Allowed' });
    }
  }