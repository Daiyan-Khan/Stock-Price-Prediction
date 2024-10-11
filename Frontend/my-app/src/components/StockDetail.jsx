import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const StockDetail = () => {
    const { stockName } = useParams();
    const [stockData, setStockData] = useState([]);
    const [predictedPrices, setPredictedPrices] = useState([]);
    const [nextDayPrediction, setNextDayPrediction] = useState(null);

    useEffect(() => {
        const fetchStockData = async () => {
            try {
                const response = await axios.get(`http://localhost:5003/api/stocks/${stockName}/details`);
                const { actual, predicted, next_day_prediction } = response.data;
                const latestStockData = actual.slice(-10); // Get latest 10 records
                setStockData(latestStockData);
                setPredictedPrices(predicted);
                setNextDayPrediction(next_day_prediction);
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };

        fetchStockData();
    }, [stockName]);

    // Prepare chart data
    const chartData = {
        labels: stockData.map((record, index) => `Record ${index + 1}`),
        datasets: [
            {
                label: `${stockName} Close Prices`,
                data: stockData.map(record => record.close),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                fill: true,
            },
            {
                label: 'Predicted Prices',
                data: predictedPrices.map(record => record.predicted_price), // Use the predicted prices
                borderColor: 'rgba(255, 99, 132, 1)',
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderWidth: 2,
                fill: true,
                pointRadius: 5, // Highlight the prediction points
                pointHoverRadius: 7,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: `Latest 10 Records and Prediction for ${stockName}`,
            },
        },
    };

    return (
        <div>
            <h1>{stockName} - Latest 10 Records</h1>
            {stockData.length === 0 ? (
                <p>Loading stock data...</p>
            ) : (
                <div>
                    <Line data={chartData} options={chartOptions} />

                    {/* Display the data in tabular format */}
                    <table border="1" style={{ marginTop: '20px', width: '100%', textAlign: 'center' }}>
                        <thead>
                            <tr>
                                <th>Index</th>
                                <th>Date</th>
                                <th>Close Price</th>
                                <th>Predicted Price</th>
                            </tr>
                        </thead>
                        <tbody>
                            {stockData.map((record, index) => (
                                <tr key={index}>
                                    <td>{index + 1}</td>
                                    <td>{new Date(record.date).toLocaleDateString()}</td>
                                    <td>{record.close.toFixed(2)}</td>
                                    <td>{predictedPrices[index] ? predictedPrices[index].predicted_price.toFixed(2) : 'N/A'}</td>
                                </tr>
                            ))}
                            {/* Display next day's predicted price in the table */}
                            {nextDayPrediction !== null && (
                                <tr>
                                    <td>{stockData.length + 1}</td>
                                    <td>Prediction (Next Day)</td>
                                    <td>{'N/A'}</td>
                                    <td>{nextDayPrediction.toFixed(2)}</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            )}

            <h2>Predicted Next Close Price</h2>
            {nextDayPrediction === null ? (
                <p>Loading prediction...</p>
            ) : (
                <p>The predicted next close price is: ${nextDayPrediction.toFixed(2)}</p>
            )}
        </div>
    );
};

export default StockDetail;
