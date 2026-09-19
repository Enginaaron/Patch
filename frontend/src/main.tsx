import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import SearchPage from './pages/SearchPage.tsx'
import MemoryPage from './pages/MemoryPage.tsx'
import ItemPage from './pages/ItemPage.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/search/:searchId" element={<SearchPage />} />
        <Route path="/memory" element={<MemoryPage />} />
        <Route path="/items/:itemId" element={<ItemPage />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
