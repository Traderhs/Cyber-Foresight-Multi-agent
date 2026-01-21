# Comparative Evaluation
This directory contains experiments for evaluating the performance of the B-MTGNN model against the MTGNN model as well as against four baseline models. Performance is assessed using two evaluation metrics: the Root Relative Squared Error (RSE) and the Relative Absolute Error (RAE).

## Directory Structure

### Baselines
The **Baselines** directory contains the implementation and evaluation of four baseline models:

1. [**ARIMA**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/Baselines/ARIMA): The ARIMA.py script trains and assesses the ARIMA model's forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). A distinct model is constructed for each trend. Evaluation of the final results is conducted based on RSE and RAE metrics. These errors are averaged over five experiments. The forecast values are saved in forecast_results_ARIMA.csv.

2. [**VAR**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/Baselines/VAR): The VAR.py script trains and assesses the VAR model's forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). A distinct model is constructed for each trend where the model uses additional features besides the ground truth. Evaluation of the final results is conducted based on RSE and RAE metrics. These errors are averaged over five experiments. The forecast values are saved in forecast_results_VAR.csv.

3. [**LSTM**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/Baselines/LSTM): The LSTM directory contains scripts for training and assessing both univariate (LSTM_u.py) and multivariate (LSTM_m.py) LSTM models' forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). Evaluation of the final results is conducted based on RSE and RAE metrics. These errors are averaged over five experiments.

4. [**Transformer**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/Baselines/Transformer): The Transformer directory contains scripts for training and assessing both univariate (transformer_u.py) and multivariate (transformer_m.py) Transformer models' forecasting performance for 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). Evaluation of the final results is conducted based on RSE and RAE metrics. These errors are averaged over five experiments.

### BMTGNN
The [**BMTGNN**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/BMTGNN) directory contains a script for evaluating the performance of the B-MTGNN model in forecasting 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). The model uses 10 iterations by default to approximate a Bayesian model. The experiment is repeated five times and the average errors (RSE and RAE) are computed and printed on the terminal. The best model is saved to the file modelb10.pt and used to forecast the unseen data. The directory contains the results of five different Bayesian models, each using a different number of iterations in the range 10-50.

### MTGNN
The [**MTGNN**](https://github.com/seongilhan/Multi-agent-AI-Framework-cyber-trend-forecasting/tree/main/Comparative_Evaluation/MTGNN) directory contains a script for evaluating the performance of the MTGNN model in forecasting 142 cyber trends, predicting 36 time-steps ahead (multistep forecast). The experiment is repeated five times and the average errors (RSE and RAE) are computed and printed on the terminal. The best model is saved to the file modelb1.pt and used to forecast the unseen data. The model can be thought of as a Bayesian model with a single iteration.

## Evaluation Metrics
The evaluation of all models is based on two metrics:
- **Root Relative Squared Error (RSE)**: Measures the relative error between the predicted and actual values.
- **Relative Absolute Error (RAE)**: Measures the absolute error between the predicted and actual values.
