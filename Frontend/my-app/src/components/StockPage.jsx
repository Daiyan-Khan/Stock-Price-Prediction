import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, LineElement, CategoryScale, LinearScale, PointElement, Title } from 'chart.js';

// Register Chart.js components
ChartJS.register(LineElement, CategoryScale, LinearScale, PointElement, Title);

const StockPage = () => {
  const { symbol } = useParams();
  const [currentPrice, setCurrentPrice] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [predictions, setPredictions] = useState([]);

  useEffect(() => {
    // Fetch current price, historical data, and predictions from the backend
    axios.get(`http://localhost:5001/api/stock/${symbol}`) // Update to include the base URL
      .then(response => {
        const { price, history, prediction } = response.data;
        setCurrentPrice(price);
        setHistoricalData(history);
        setPredictions(prediction);
      })
      .catch(error => console.error('Error fetching stock data:', error));
  }, [symbol]);

  const chartData = {
    labels: historicalData.map(data => data.timestamp),
    datasets: [
      {
        label: 'Historical Prices',
        data: historicalData.map(data => data.price),
        borderColor: 'blue',
        fill: false
      },
      {
        label: 'Predicted Prices',
        data: predictions.map(pred => pred.price),
        borderColor: 'red',
        fill: false
      }
    ]
  };

  return (
    <div>
      <h1>{symbol} Stock Page</h1>
      {currentPrice !== null ? (
        <div>
          <h2>Current Price: ${currentPrice}</h2>
          <div>
            <Line data={chartData} />
          </div>
        </div>
      ) : (
        <p>Loading data...</p>
      )}
    </div>
  );
};

export default StockPage;
