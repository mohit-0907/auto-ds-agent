// Entry point: draw <App/> into <div id="root"> in index.html.

import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router'
// Bundled fonts: the files ship inside the build, so the app works offline and never
// contacts Google Fonts. Only the weights the design uses.
import '@fontsource/caprasimo/400.css'
import '@fontsource/figtree/400.css'
import '@fontsource/figtree/500.css'
import '@fontsource/figtree/600.css'
import '@fontsource/figtree/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'
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
