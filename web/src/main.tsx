// Entry point: draw <App/> into <div id="root"> in index.html.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  // StrictMode (development only): React deliberately mounts components twice to expose
  // missing clean-ups -- which is why useRunEvents must close its connection in clean-up.
  <StrictMode>
    {/* BrowserRouter keeps React Router in sync with the address bar. */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
)
