"""
Cybersecurity Intrusion Detection - Regression Model
Predicts session duration based on network traffic characteristics

Author: Bereket Takiso
Dataset: Kaggle Cybersecurity Intrusion Detection Dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

class CybersecurityRegressionModel:
    """
    A machine learning model to predict session duration based on
    network packet size and login attempts in cybersecurity context.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.model_results = {}
        
    def load_data(self, file_path):
        """Load dataset from CSV file"""
        self.df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {self.df.shape}")
        return self.df
        
    def prepare_data(self, target_var='session_duration', 
                     feature_vars=['network_packet_size', 'login_attempts']):
        """Prepare features and target variables"""
        self.target_var = target_var
        self.feature_vars = feature_vars
        
        X = self.df[feature_vars].copy()
        y = self.df[target_var].copy()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Data prepared: Train {self.X_train.shape}, Test {self.X_test.shape}")
        
    def train_models(self):
        """Train multiple regression models and compare performance"""
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Train model
            if model_name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Store results
            self.model_results[model_name] = {
                'model': model,
                'y_pred': y_pred,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"  R² Score: {r2:.4f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print()
        
        # Select best model
        self.best_model_name = max(self.model_results.keys(), 
                                   key=lambda k: self.model_results[k]['r2'])
        self.best_model = self.model_results[self.best_model_name]['model']
        print(f"Best Model: {self.best_model_name}")
        
    def predict(self, network_packet_size, login_attempts):
        """Make prediction for new data"""
        if self.best_model is None:
            raise ValueError("Model not trained. Call train_models() first.")
            
        input_data = [[network_packet_size, login_attempts]]
        
        if self.best_model_name == 'Linear Regression':
            input_scaled = self.scaler.transform(input_data)
            prediction = self.best_model.predict(input_scaled)[0]
        else:
            prediction = self.best_model.predict(input_data)[0]
            
        return prediction
        
    def visualize_results(self):
        """Create visualizations of model performance"""
        if not self.model_results:
            raise ValueError("No models trained. Call train_models() first.")
            
        best_results = self.model_results[self.best_model_name]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Regression Model Performance - {self.best_model_name}', 
                     fontsize=14, fontweight='bold')
        
        # Actual vs Predicted
        axes[0, 0].scatter(self.y_test, best_results['y_pred'], alpha=0.7)
        min_val = min(min(self.y_test), min(best_results['y_pred']))
        max_val = max(max(self.y_test), max(best_results['y_pred']))
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        axes[0, 0].set_xlabel('Actual Session Duration')
        axes[0, 0].set_ylabel('Predicted Session Duration')
        axes[0, 0].set_title('Actual vs Predicted')
        
        # Residuals
        residuals = self.y_test - best_results['y_pred']
        axes[0, 1].scatter(best_results['y_pred'], residuals, alpha=0.7)
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Values')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        
        # Feature importance/coefficients
        if self.best_model_name == 'Random Forest':
            importance = self.best_model.feature_importances_
            axes[1, 0].bar(self.feature_vars, importance)
            axes[1, 0].set_title('Feature Importance')
        else:
            coef = self.best_model.coef_
            axes[1, 0].bar(self.feature_vars, coef)
            axes[1, 0].set_title('Feature Coefficients')
        
        # Model comparison
        models = list(self.model_results.keys())
        r2_scores = [self.model_results[m]['r2'] for m in models]
        axes[1, 1].bar(models, r2_scores)
        axes[1, 1].set_title('Model Comparison (R² Score)')
        axes[1, 1].set_ylabel('R² Score')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = CybersecurityRegressionModel()
    
    # Load data (you need to download the dataset first)
    # model.load_data('../data/cybersecurity_intrusion_data.csv')
    # model.prepare_data()
    # model.train_models()
    # model.visualize_results()
    
    # Make predictions
    # prediction = model.predict(network_packet_size=500, login_attempts=3)
    # print(f"Predicted session duration: {prediction:.2f} seconds")
    
    print("CybersecurityRegressionModel class loaded successfully!")
    print("Download the dataset to use this model.")
