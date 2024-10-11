import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const HomePage = () => {
    const [stocks, setStocks] = useState([]);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchStocks = async () => {
            try {
                const response = await axios.get('http://localhost:5003/api/stocks');
                // Assuming the response data is an object where the keys are stock names
                const stockNames = Object.keys(response.data).map((stock) =>
                    stock.replace('_preprocessed_data.csv', '') // Remove the suffix
                );
                setStocks(stockNames);
            } catch (error) {
                console.error('Error fetching stocks:', error);
            }
        };

        fetchStocks();
    }, []);

    const handleStockClick = (stockName) => {
        navigate(`/stocks/${stockName}`);
    };

    return (
        <div>
            <h1>Stocks Data</h1>
            {stocks.length === 0 ? (
                <p>Loading stocks...</p>
            ) : (
                stocks.map((stock) => (
                    <button key={stock} onClick={() => handleStockClick(stock)}>
                        {stock}
                    </button>
                ))
            )}
        </div>
    );
};

export default HomePage;
