import { AppBar, Button, Toolbar, Typography } from "@mui/material";
import { SignedIn, SignedOut, UserButton } from "@clerk/nextjs";

export default function CustomAppBar() {
    return (
        <AppBar position="static" className="nav-bar" sx={{ backgroundColor:'white'}}>
            <Toolbar className="tool-bar">
                <div style={{ display: 'flex', alignItems: 'center' }}>
                    {/* <img src="\logo.png" width="50px"></img> */}
                    <Typography className="logo-title" component="a" href="/" variant="h6" sx={{  textDecoration: 'none', color: 'black', paddingLeft: '10px'}}>
                        VitalPath
                    </Typography>
                </div>
                <div>
                    <Button color="inherit" href="" className="nav-item" sx={{ color:'black'}}>Pricing</Button>
                    <Button color="inherit" href="" className="nav-item" sx={{ color:'black'}}>About</Button>
                    <Button color="inherit" href="" className="nav-item" sx={{ color:'black'}}>Resource</Button>
                    <Button color="inherit" href="" className="nav-item" sx={{ color:'black'}}>Contact</Button>    
                </div>
                <div>
                    <SignedOut>
                        <Button color="inherit" href="/sign-up" className="button-white" sx={{ mr: 2, backgroundColor: 'white', color: 'black', fontWeight: 600, borderRadius: '10px', padding: '5px 15px 5px 15px', marginLeft: '10px','&:hover': {backgroundColor: '#e2e2e2',}, }}>Sign Up</Button>
                        <Button color="inherit" href="/sign-in" className="button-blue" sx={{ mr: 2, backgroundColor: '#2E46CD', color: 'white', fontWeight: 600, borderRadius: '10px', padding: '5px 15px 5px 15px', marginLeft: '10px','&:hover': {backgroundColor: '#1565C0',}, }}>Login</Button>
                    </SignedOut>
                    <SignedIn>
                        <Button color="inherit" href="/question" sx={{ mr: 2, backgroundColor: '#2E46CD', color: 'white', fontWeight: 600, borderRadius: '10px', padding: '5px 15px 5px 15px', marginLeft: '10px','&:hover': {backgroundColor: '#1565C0',}, }}>Analyze</Button>
                        <UserButton sx = {{ marginTop: '5px'}} />
                    </SignedIn>
                </div>
                
            </Toolbar>
        </AppBar>
    );
}