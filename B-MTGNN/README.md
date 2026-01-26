# B-MTGNN
This is a PyTorch implementation of the Bayesian model proposed in the paper: **"An Explainable Multi-agent AI Framework for Forecasting Asymmetric Evolution Between Cyber Threats and Pertinent Mitigation Technologies"**.
The model forecasts the graph of cyber-attacks and pertinent technologies 3 years in advance, 
while extending the **MTGNN** model. 
The graph includes 26 rapidly increasing and emerging attacks and 98 pertinent technologies, 
each represented by a single node. The value of the node represents the trend.

In our extension for the model, we employ the Bayesian approach to capture epistemic uncertainty. 
Specifically, we employ the Monte Carlo dropout method where the use of dropout neurons during inference provides a Bayesian approximation of the deep Gaussian processes. 
Therefore, during the prediction phase, the trained model runs multiple times, which results in a distribution of prediction (representing the uncertainty) rather than a single point.

## Requirements
The model is implemented using Python3 with dependencies specified in requirements.txt

## Data Smoothing
All data files used by the model including the graph adjacency file can be found in the directory called **data**.

The file **smoothing.py** performs double exponential smoothing on the data (**data.txt**) and produces the file **sm_data.csv**.

## Hyper-parameter Optimisation
The hyper-parameter optimisation is performed in the file **train_test.py**. 
This script performs random search to produce the optimal set of hyper-parameters. 
These hyper-parameters are finally saved as an output in the file called **hp.txt**, which is in the directory **model/Bayesian**. 
The output also includes validation and testing results when using the optimal set of hyper-parameters. 
These results include plots for the predicted curves against the actual curves. 
These are saved in the directories called **Validation** and **Testing** within the directory **model/Bayesian**. 
For the evaluation, 2 metrics are used namely the Root Relative Squared Error (RSE) and the Relative Absolute Error (RAE). 
These metrics are saved in the same directories (Validation and Testing), and the average values of these metrics across 142 nodes are also displayed on the console as a final output. 

Below is an example for the model testing results, where the forecasts of the Malware and DDoS nodes are tested. The Relative Absolute Errors (RAE) are 0.46 for Malware and 0.57 for DDoS.
<p align="center">
<img src="./model/Bayesian/Testing/Malware_Testing.png" width="49%" height="auto" style="object-fit: cover;">
<img src="./model/Bayesian/Testing/DDoS_Testing.png" width="49%" height="auto" style="object-fit: cover;">
</p>

## Operational Model
The script in the file **train.py** trains the final model on the full data using the optimal hyper-parameters stored in the file **hp.txt**. 
The output is the operational model called **o_model.pt**, which can be used to forecast the graph up to 3 years in advance. 
The operational model is saved in the directory **model/Bayesian**. 

## Future Forecast
The script in the file **forecast.py** uses the operational model **o_model.pt** in the directory **model/Bayesian** to produce 3 years forecast for the trend of cyber-attacks and the pertinent technologies (graph). 
The results include numerical forecasts of each node, stored in the directory **model/Bayesian/forecast/data**. 
In addition, a plot for the trend of each attack along with its pertinent technologies, highlighting a projected future gap based on the forecast is provided in the directory **model/Bayesian/forecast/plots**. 
Within these plots, the gap is highlighted in a distinct colour. 
The numerical forecast for these gaps is also produced in the directory **model/Bayesian/forecast/gap** in a csv format, where each file contains the future gaps of a single attack with respect to its pertinent technologies.

Additionally, the script in the file **pt_plots.py** produces plots for the past and future data of each pertinent technology separately, which can be useful for the purpose of visualisation and producing a unified trend cycle. 
These plots are saved in the directory **model/Bayesian/forecast/pt_plots**.

Below is an example for the past and predicted future data for the Malware and DDoS attacks along with their pertinent technologies, with the gaps highlighted in distinct colours.
<p align="center">
<img src="./model/Bayesian/forecast/plots/Malware.png" width="50.8%" height="auto" style="object-fit: cover;">
<img src="./model/Bayesian/forecast/plots/DDoS.png" width="48.1%" height="auto" style="object-fit: cover;">
</p>

## Gap Quantification
To provide a more precise analysis of the disparity between cyber threats and defense mechanisms, we quantified the "Gap" for the **top 5 pertinent mitigation technologies** associated with each attack vector. 
These technologies were selected based on their critical relevance and the significant magnitude of the observed lag relative to the attack trends.  
The bar charts below illustrate the forecasted gap values for the years 2025, 2026, and 2027. 
In these figures, the numerical value represents the extent of the divergence between the evolving attack vectors (Malware and DDoS) and the specific defense solutions. 
**A higher gap value indicates a wider divergence**, suggesting that the development of that specific technology is not keeping pace with the rapid evolution of the threat.  
This quantitative insight highlights areas that require prioritized technological advancement. 
For instance, technologies showing larger gaps (e.g., *Game Theory* for Malware) signal an urgent need for concentrated research and development to effectively bridge the security vulnerability window. 
<p align="center">
<img src="figure/figure_Malware.png" width="49%" height="auto" style="object-fit: cover;">
<img src="figure/figure_DDoS.png" width="49%" height="auto" style="object-fit: cover;">
</p>