import axios from 'axios';
import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

const StockDetail = () => {
    const { stockName } = useParams();
    const [stockData, setStockData] = useState([]);
    const [predictedPrice, setPredictedPrice] = useState(null);

    useEffect(() => {
        const fetchStockData = async () => {
            try {
                const response = await axios.get(`http://localhost:5003/api/stocks/${stockName}`);
                setStockData(response.data.slice(-10));  // Get latest 10 records
            } catch (error) {
                console.error('Error fetching stock data:', error);
            }
        };

        const fetchPrediction = async () => {
            try {
                const predictionResponse = await axios.post('http://localhost:5001/predict', {
                    stock_name: stockName
                });
                setPredictedPrice(predictionResponse.data.predicted_price);
            } catch (error) {
                console.error('Error fetching prediction:', error);
            }
        };

        fetchStockData();
        fetchPrediction();
    }, [stockName]);

    // Adding the predicted price to the chart data
    const extendedStockData = [...stockData]; // Copy stockData
    if (predictedPrice !== null) {
        extendedStockData.push({
            date: 'Prediction',  // Label for predicted price
            close: predictedPrice,
        });
    }

    const chartData = {
        labels: extendedStockData.map((record, index) => record.date === 'Prediction' ? 'Prediction' : `Record ${index + 1}`),
        datasets: [
            {
                label: `${stockName} Close Prices`,
                data: extendedStockData.map(record => record.close),  // Use the close price for chart data
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderWidth: 2,
                fill: true,
            }
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
                            </tr>
                        </thead>
                        <tbody>
                            {stockData.map((record, index) => (
                                <tr key={index}>
                                    <td>{index + 1}</td>
                                    <td>{new Date(record.date).toLocaleDateString()}</td>
                                    <td>{record.close.toFixed(2)}</td>
                                </tr>
                            ))}
                            {/* Display predicted price in the table */}
                            {predictedPrice !== null && (
                                <tr>
                                    <td>{stockData.length + 1}</td>
                                    <td>Prediction</td>
                                    <td>{predictedPrice.toFixed(2)}</td>
                                </tr>
                            )}
                        </tbody>
                    </table>
                </div>
            )}

            <h2>Predicted Next Close Price</h2>
            {predictedPrice === null ? (
                <p>Loading prediction...</p>
            ) : (
                <p>The predicted next close price is: ${predictedPrice.toFixed(2)}</p>
            )}
        </div>
    );
};

export default StockDetail;
