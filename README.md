# ☀️⚡SolarMLProject
## Overview
This repository documents our final project for the B.Sc. in Electrical Engineering at Tel Aviv University.
The project was conducted by Hadar Levy and Gal Schwartz, under the supervision of Khen Cohen.

## Objective
The goal of the project is to develop a machine learning model based on Graph Neural Networks (GNNs) to predict Global Horizontal Irradiance (GHI) for four future time horizons: 15, 30, 45, and 60 minutes ahead.

## Repository Layout
- `.idea/` – IDE config.
- [`Batching_Nodes/`](./Batching_Nodes) – Code for node batching and preparation for GNN.
- [`Data/`](./Data) – Processed datasets for training and testing.
- [`Metrics/`](./Metrics) – Evaluation scripts for error metrics and comparisons.
- [`Model/`](./Model) – GNN model definitions and training logic.
- [`Results/`](./Results) – Forecast output plots and visualizations. Visualized predictions for 15, 30, 45, and 60-minute horizons.
- [`data_proccessing/`](./data_proccessing) – Scripts to create and convert datasets into graph format.
- [`README.md`](./README.md) – This documentation file.

## Results / Evaluation
| Time Horizon | NRMSE (%) | NMAE (%) |
|--------------|-----------|----------|
| 15 minutes   |    8.25   |   11.53  |
| 30 minutes   |    9.13   |   13.16  |
| 45 minutes   |    10.02  |   15.14  |
| 60 minutes   |    10.76  |   16.45  |

### 📈 Forecast vs. Actual for GHI @ t+15 minutes
![Actual Versus Prediction for 15 minutes forecasting](Results/Images/forecast_node0_t+15.png)

## Limitations and Future Work
- Improve accuracy for longer-term forecasts
- Add real-time weather data integration

## Credits
This project was developed by Hadar Levy and Gal Schwartz under the supervision of Khen Cohen at Tel Aviv University.

