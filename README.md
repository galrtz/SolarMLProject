# ☀️⚡SolarMLProject
## Overview
This repository documents our final project for the B.Sc. in Electrical Engineering at Tel Aviv University.
The project was conducted by Hadar Levy and Gal Schwartz, under the supervision of Khen Cohen.

## Objective
The goal of the project is to develop a machine learning model based on Graph Neural Networks (GNNs) to predict Global Horizontal Irradiance (GHI) for four future time horizons: 15, 30, 45, and 60 minutes ahead.

## Project Structure

## Results / Evaluation
| Time Horizon | NRMSE (%) | NMAE (%) |
|--------------|-----------|----------|
| 15 minutes   | 10.5      | 8.2      |
| 30 minutes   | 12.8      | 9.7      |
| 45 minutes   | 14.3      | 11.1     |
| 60 minutes   | 15.9      | 12.4     |

### 📈 Forecast vs. Actual for GHI @ t+15 minutes
![Actual Versus Prediction for 15 minutes forecasting](Results/Images/pv_forecast_t+15_all_range.png)

## Limitations and Future Work
- Improve accuracy for longer-term forecasts
- Add real-time weather data integration
- Explore transformer-based GNN models

## Credits
This project was developed by Hadar Levy and Gal Schwartz under the supervision of Khen Cohen at Tel Aviv University.

