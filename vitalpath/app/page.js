import {
  SignInButton,
  SignedIn,
  SignedOut,
  UserButton
} from '@clerk/nextjs'
import { Box, Typography } from '@mui/material';

import Hero from '@/component/Hero.jsx';
import Navbar from '@/component/Navbar.jsx';

export default function Home() {
  return (
    <div>
      <Navbar />
      <Hero id = "header"/>

    </div>
  );
}


