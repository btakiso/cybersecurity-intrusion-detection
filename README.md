# Cybersecurity Intrusion Detection - Regression Analysis

A machine learning project using sklearn regression to predict session duration based on network traffic characteristics.

## Project Overview

This project analyzes a cybersecurity intrusion detection dataset to predict user session duration using regression techniques. The analysis helps understand patterns in network behavior that could be useful for cybersecurity monitoring.

## Dataset

- **Source**: [Kaggle Cybersecurity Intrusion Detection Dataset](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset)
- **Size**: 9,537 records with 10 features
- **Target Variable (Y)**: `session_duration` - Length of user session in seconds
- **Input Variables (X)**: 
  - `network_packet_size` - Size of network packets in bytes
  - `login_attempts` - Number of login attempts in the session

## Regression Analysis Rationale

**Why these variables work well for regression:**

1. **session_duration (Y)**: A natural continuous variable ranging from 0.5 to 7,190 seconds, perfect for regression prediction
2. **network_packet_size (X1)**: Larger packets often indicate more data transfer, potentially correlating with longer sessions
3. **login_attempts (X2)**: Multiple login attempts might indicate either legitimate user activity or attack patterns, both affecting session length

**Why this is a good prediction problem:**

- **Business Value**: Predicting session duration helps identify unusual patterns that might indicate security threats
- **Practical Application**: Cybersecurity systems can use this to flag abnormally long/short sessions for investigation  
- **Data Quality**: Strong correlation potential between network characteristics and session behavior
- **Real-world Relevance**: Session duration prediction is valuable for network resource planning and security monitoring

## Project Structure

```
├── README.md
├── download_dataset.py
├── data/
│   └── cybersecurity_intrusion_data.csv
├── notebooks/
│   └── cybersecurity_regression_analysis.ipynb
├── src/
│   └── regression_model.py
└── results/
    ├── visualizations/
    │   ├── 01_training_dataset_analysis.png/pdf
    │   └── 02_regression_model_performance.png/pdf
    ├── model_results_summary.txt
    └── model_predictions.csv
```

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter

## Usage

1. Open the Google Colab notebook: `notebooks/cybersecurity_regression_analysis.ipynb`
2. Run all cells to reproduce the analysis
3. View results and visualizations in the output

## Results

The analysis generates comprehensive outputs automatically saved to the `results/` directory:

- **Visualizations**: High-resolution PNG and PDF charts for presentations
- **Model Summary**: Complete analysis report with performance metrics  
- **Predictions**: Actual vs predicted values for model evaluation
- **Performance Metrics**: R² = 0.0012, RMSE = 810.36 seconds

Key finding: Current features (packet size, login attempts) explain only 0.1% of session duration variance, indicating need for additional feature engineering in cybersecurity session prediction.

## Author

Bereket Takiso - Machine Learning Portfolio Project
