import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import HomePage from './components/HomePage';
import StockDetail from './components/StockDetail';

const App = () => {
    return (
        <Router>
            <Routes>
                <Route path="/" element={<HomePage />} />
                <Route path="/stocks/:stockName" element={<StockDetail />} />
            </Routes>
        </Router>
    );
};

export default App;
