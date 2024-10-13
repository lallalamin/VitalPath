import React from 'react'
import { Container, Box, Typography, AppBar, Toolbar, Button } from '@mui/material'
import { SignIn } from '@clerk/nextjs'
import Link from 'next/link'

export default function SignUpPage() {
  return(
    <Container maxWidth="sm">
        <AppBar position="static" sx={{backgroundColor: 'linear-gradient{to right, #2980b9, #6dd5ed}'}}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', margin: '15px' }}>
                <Typography variant="h6"  >
                VitalPath
                </Typography>
                <div>
                    <Button
                        sx={{
                        backgroundColor: 'white',
                        borderRadius: '50px',  
                        padding: '8px 15px',
                        marginRight: '10px', 
                        color: '#3f51b5', 
                        '&:hover': {
                            backgroundColor: '#f0f0f0',  
                        }
                        }}
                    >
                        <Link href="/sign-in" passHref style={{ textDecoration: 'none', color: 'inherit' }}>
                        Login
                        </Link>
                    </Button>
                    <Button 
                    sx={{
                        backgroundColor: 'white',
                        borderRadius: '50px',  
                        padding: '8px 15px', 
                        color: '#3f51b5', 
                        '&:hover': {
                            backgroundColor: '#f0f0f0',  
                        }
                        }}>
                        <Link href="/sign-up" passHref style={{ textDecoration: 'none', color: 'inherit' }}>
                            Sign-up
                        </Link>
                    </Button>
                </div>
            </div>
        </AppBar>
        <Box
        display="flex"
        flexDirection="column"
        justifyContent="center"
        alignItems="center"
        sx={{textAlign: 'center', my: 4}}
        >
            <SignIn></SignIn>
        </Box>
    </Container>
  )
}

