import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Modo oscuro por defecto; respeta la elección guardada del usuario.
try {
  if (localStorage.getItem('theme') === 'light') document.documentElement.classList.remove('dark')
} catch {
  /* sin almacenamiento */
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
