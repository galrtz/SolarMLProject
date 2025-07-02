# ☀️⚡SolarMLProject
## Overview
This repository documents our final project for the B.Sc. in Electrical Engineering at Tel Aviv University.
The project was conducted by Hadar Levy and Gal Schwartz, under the supervision of Khen Cohen.

## Objective
The goal of the project is to develop a machine learning model based on Graph Neural Networks (GNNs) to predict Global Horizontal Irradiance (GHI) for four future time horizons: 15, 30, 45, and 60 minutes ahead.

## Project Structure
- [`Model/`](./Model) – GNN model definitions and training logic.
- [`Metrics/`](./Metrics) – Evaluation scripts for error metrics and comparisons.
- [`Results/`](./Results) – Forecast output plots and visualizations. Visualized predictions for 15, 30, 45, and 60-minute horizons.
- [`Data/`](./Data) – Processed datasets for training and testing.
- [`data_proccessing/`](./data_proccessing) – Scripts to create and convert datasets into graph format.
- [`Batching_Nodes/`](./Batching_Nodes) – Code for node batching and preparation for GNN.
- `.idea/` – IDE config (excluded from Git tracking).
- [`README.md`](./README.md) – This documentation file.

## Results / Evaluation
| Time Horizon | NRMSE (%) | NMAE (%) |
|--------------|-----------|----------|
| 15 minutes   |    8.16   |          |
| 30 minutes   |    8.9    |          |
| 45 minutes   |    9.8    |          |
| 60 minutes   |    10.05  |          |

### 📈 Forecast vs. Actual for GHI @ t+15 minutes
![Actual Versus Prediction for 15 minutes forecasting](Results/Images/pv_forecast_t+15_all_range.png)

## Limitations and Future Work
- Improve accuracy for longer-term forecasts
- Add real-time weather data integration
- Explore transformer-based GNN models

## Credits
This project was developed by Hadar Levy and Gal Schwartz under the supervision of Khen Cohen at Tel Aviv University.

