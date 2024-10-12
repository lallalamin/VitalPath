// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
//import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyD_Sw-KaIGGihNa8U5pS9M43Bs1X9y9OkU",
  authDomain: "vitalpath-6d3ab.firebaseapp.com",
  projectId: "vitalpath-6d3ab",
  storageBucket: "vitalpath-6d3ab.appspot.com",
  messagingSenderId: "1014004541504",
  appId: "1:1014004541504:web:0ebbb5661d5081d8eb17a5",
  measurementId: "G-7S2RDQENY4"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
//const analytics = getAnalytics(app);