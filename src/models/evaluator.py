"""
Model evaluation utilities for AgroWeather AI
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
from typing import Dict, Tuple, Any, Optional
import os

from .lstm_model import RainfallLSTM


class ModelEvaluator:
    """
    Comprehensive model evaluation for rainfall prediction
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize evaluator
        
        Args:
            device: Device to use for evaluation
        """
        self.device = device
        self.metrics = {}
    
    def load_test_data(self, data_dir: str = 'data/processed') -> Tuple[np.ndarray, np.ndarray, Any, Dict]:
        """
        Load test data and scalers
        
        Args:
            data_dir: Directory containing processed data
            
        Returns:
            Tuple of (X_test, y_test, target_scaler, metadata)
        """
        print("📂 Loading test data...")
        
        X_test = np.load(f'{data_dir}/X_test.npy')
        y_test = np.load(f'{data_dir}/y_test.npy')
        
        with open(f'{data_dir}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        with open(f'{data_dir}/target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)
        
        print(f"✅ Test data loaded!")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {metadata['n_features']}")
        
        return X_test, y_test, target_scaler, metadata
    
    def load_trained_model(self, model_path: str, metadata: Dict) -> RainfallLSTM:
        """
        Load trained model
        
        Args:
            model_path: Path to saved model
            metadata: Model metadata
            
        Returns:
            Loaded model
        """
        print(f"\n🧠 Loading trained model from {model_path}...")
        
        # Create model with correct architecture
        model = RainfallLSTM(input_size=metadata['n_features'])
        
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"✅ Model loaded successfully!")
        
        return model
    
    def make_predictions(self, model: RainfallLSTM, X_test: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions on test set
        
        Args:
            model: Trained model
            X_test: Test features
            batch_size: Batch size for prediction
            
        Returns:
            Predictions array
        """
        print(f"\n🔮 Making predictions on test set...")
        
        model.eval()
        predictions = []
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test)
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            for i in range(0, len(X_test_tensor), batch_size):
                batch = X_test_tensor[i:i+batch_size].to(self.device)
                batch_pred = model(batch)
                predictions.append(batch_pred.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        
        print(f"✅ Predictions complete!")
        print(f"   Shape: {predictions.shape}")
        
        return predictions
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scaler: Any) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics
        
        Args:
            y_true: True values (scaled)
            y_pred: Predicted values (scaled)
            scaler: Target scaler for inverse transformation
            
        Returns:
            Dictionary of metrics
        """
        print(f"\n📊 Calculating metrics...")
        
        # Inverse transform to get actual mm values
        y_true_mm = scaler.inverse_transform(y_true)
        y_pred_mm = scaler.inverse_transform(y_pred)
        
        # Regression metrics
        mse = mean_squared_error(y_true_mm, y_pred_mm)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true_mm, y_pred_mm)
        r2 = r2_score(y_true_mm, y_pred_mm)
        
        # Baseline comparison (always predict mean)
        baseline_pred = np.full_like(y_true_mm, y_true_mm.mean())
        baseline_rmse = np.sqrt(mean_squared_error(y_true_mm, baseline_pred))
        improvement = ((baseline_rmse - rmse) / baseline_rmse) * 100
        
        # Classification metrics (rain vs no rain threshold = 1mm)
        y_true_binary = (y_true_mm > 1.0).astype(int)
        y_pred_binary = (y_pred_mm > 1.0).astype(int)
        
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Store metrics
        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'baseline_rmse': baseline_rmse,
            'improvement_over_baseline': improvement,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'y_true_mm': y_true_mm,
            'y_pred_mm': y_pred_mm,
            'y_true_binary': y_true_binary,
            'y_pred_binary': y_pred_binary
        }
        
        return self.metrics
    
    def print_metrics(self, metrics: Optional[Dict] = None) -> None:
        """
        Print metrics in readable format
        
        Args:
            metrics: Metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print("\n" + "="*70)
        print("📈 MODEL PERFORMANCE METRICS")
        print("="*70)
        
        print(f"\n🎯 REGRESSION METRICS (Rainfall Amount Prediction):")
        print(f"   RMSE:  {metrics['rmse']:.3f} mm/day")
        print(f"   MAE:   {metrics['mae']:.3f} mm/day")
        print(f"   MSE:   {metrics['mse']:.3f}")
        print(f"   R² Score: {metrics['r2']:.3f}")
        
        print(f"\n📊 BASELINE COMPARISON:")
        print(f"   Baseline RMSE (always predict mean): {metrics['baseline_rmse']:.3f} mm")
        print(f"   Our Model RMSE: {metrics['rmse']:.3f} mm")
        print(f"   Improvement: {metrics['improvement_over_baseline']:.1f}%")
        
        print(f"\n🎲 CLASSIFICATION METRICS (Will it rain? >1mm):")
        print(f"   Accuracy:  {metrics['accuracy']*100:.1f}%")
        print(f"   Precision: {metrics['precision']*100:.1f}%")
        print(f"   Recall:    {metrics['recall']*100:.1f}%")
        
        # Interpretation guide
        print(f"\n💡 INTERPRETATION:")
        if metrics['rmse'] < 5:
            print(f"   RMSE < 5mm: ⭐⭐⭐ EXCELLENT! Commercial-grade accuracy")
        elif metrics['rmse'] < 10:
            print(f"   RMSE < 10mm: ⭐⭐ GOOD! Suitable for agricultural decisions")
        elif metrics['rmse'] < 15:
            print(f"   RMSE < 15mm: ⭐ ACCEPTABLE for general forecasting")
        else:
            print(f"   RMSE > 15mm: ⚠️  Needs improvement")
        
        if metrics['r2'] > 0.7:
            print(f"   R² > 0.7: ✅ Model explains {metrics['r2']*100:.0f}% of variance")
        elif metrics['r2'] > 0.5:
            print(f"   R² > 0.5: ⚠️  Moderate explanatory power")
        else:
            print(f"   R² < 0.5: ❌ Weak predictive power")
        
        if metrics['accuracy'] > 0.8:
            print(f"   Accuracy > 80%: ✅ Reliable rain/no-rain predictions")
        
        print("\n" + "="*70)
    
    def plot_predictions_vs_actual(self, output_dir: str = 'outputs/plots', metrics: Optional[Dict] = None) -> None:
        """
        Create comprehensive evaluation plots
        
        Args:
            output_dir: Directory to save plots
            metrics: Metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print(f"\n📊 Creating evaluation plots...")
        
        y_true = metrics['y_true_mm'].flatten()
        y_pred = metrics['y_pred_mm'].flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LSTM Model Evaluation - Test Set Performance',
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Predicted vs Actual (Scatter)
        axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=20)
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0, 0].set_xlabel('Actual Rainfall (mm)', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Rainfall (mm)', fontsize=11)
        axes[0, 0].set_title('Predicted vs Actual Rainfall', fontsize=12, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² text
        textstr = f'R² = {metrics["r2"]:.3f}\nRMSE = {metrics["rmse"]:.2f} mm'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        axes[0, 0].text(0.05, 0.95, textstr, transform=axes[0, 0].transAxes,
                        fontsize=10, verticalalignment='top', bbox=props)
        
        # Plot 2: Time Series (first 200 predictions)
        n_show = min(200, len(y_true))
        indices = range(n_show)
        
        axes[0, 1].plot(indices, y_true[:n_show], label='Actual', linewidth=1.5, alpha=0.7)
        axes[0, 1].plot(indices, y_pred[:n_show], label='Predicted', linewidth=1.5, alpha=0.7)
        axes[0, 1].set_xlabel('Test Sample', fontsize=11)
        axes[0, 1].set_ylabel('Rainfall (mm)', fontsize=11)
        axes[0, 1].set_title(f'Time Series Comparison (First {n_show} samples)',
                             fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution
        errors = y_true - y_pred
        axes[1, 0].hist(errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[1, 0].axvline(x=errors.mean(), color='green', linestyle='--',
                           linewidth=2, label=f'Mean Error: {errors.mean():.2f}mm')
        axes[1, 0].set_xlabel('Prediction Error (mm)', fontsize=11)
        axes[1, 0].set_ylabel('Frequency', fontsize=11)
        axes[1, 0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Confusion Matrix (Rain/No Rain)
        cm = confusion_matrix(metrics['y_true_binary'], metrics['y_pred_binary'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                    xticklabels=['No Rain', 'Rain'],
                    yticklabels=['No Rain', 'Rain'])
        axes[1, 1].set_xlabel('Predicted', fontsize=11)
        axes[1, 1].set_ylabel('Actual', fontsize=11)
        axes[1, 1].set_title('Confusion Matrix (Rain vs No Rain)',
                             fontsize=12, fontweight='bold')
        
        # Add accuracy text
        acc_text = f'Accuracy: {metrics["accuracy"]*100:.1f}%\nPrecision: {metrics["precision"]*100:.1f}%\nRecall: {metrics["recall"]*100:.1f}%'
        axes[1, 1].text(0.02, 0.98, acc_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f'{output_dir}/model_evaluation.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Evaluation plots saved: {plot_path}")
        plt.close()
    
    def plot_error_by_rainfall_intensity(self, output_dir: str = 'outputs/plots', metrics: Optional[Dict] = None) -> None:
        """
        Analyze how errors vary by rainfall intensity
        
        Args:
            output_dir: Directory to save plots
            metrics: Metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print(f"\n📊 Creating error analysis by rainfall intensity...")
        
        y_true = metrics['y_true_mm'].flatten()
        y_pred = metrics['y_pred_mm'].flatten()
        errors = np.abs(y_true - y_pred)
        
        # Categorize rainfall
        categories = []
        for val in y_true:
            if val <= 1:
                categories.append('No Rain (≤1mm)')
            elif val <= 10:
                categories.append('Light (1-10mm)')
            elif val <= 25:
                categories.append('Moderate (10-25mm)')
            elif val <= 50:
                categories.append('Heavy (25-50mm)')
            else:
                categories.append('Very Heavy (>50mm)')
        
        # Create dataframe
        df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Error': errors,
            'Category': categories
        })
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.suptitle('Error Analysis by Rainfall Intensity', fontsize=14, fontweight='bold')
        
        # Box plot of errors by category
        category_order = ['No Rain (≤1mm)', 'Light (1-10mm)', 'Moderate (10-25mm)',
                          'Heavy (25-50mm)', 'Very Heavy (>50mm)']
        
        df_plot = df[df['Category'].isin(category_order)]
        
        axes[0].boxplot([df_plot[df_plot['Category']==cat]['Error'].values
                         for cat in category_order if cat in df_plot['Category'].values],
                       labels=[cat for cat in category_order if cat in df_plot['Category'].values])
        axes[0].set_ylabel('Absolute Error (mm)', fontsize=11)
        axes[0].set_title('Error Distribution by Rainfall Category', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Average error by category
        avg_errors = df.groupby('Category')['Error'].mean().reindex(category_order)
        avg_errors = avg_errors.dropna()
        
        axes[1].bar(range(len(avg_errors)), avg_errors.values, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(len(avg_errors)))
        axes[1].set_xticklabels(avg_errors.index, rotation=45, ha='right')
        axes[1].set_ylabel('Mean Absolute Error (mm)', fontsize=11)
        axes[1].set_title('Average Error by Rainfall Category', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for i, v in enumerate(avg_errors.values):
            axes[1].text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plot_path = f'{output_dir}/error_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   ✅ Error analysis saved: {plot_path}")
        plt.close()
    
    def save_evaluation_report(self, output_dir: str = 'outputs/reports', metrics: Optional[Dict] = None) -> None:
        """
        Save evaluation metrics to file
        
        Args:
            output_dir: Directory to save reports
            metrics: Metrics dictionary (uses self.metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print(f"\n💾 Saving evaluation report...")
        
        report = {
            'regression_metrics': {
                'rmse_mm': float(metrics['rmse']),
                'mae_mm': float(metrics['mae']),
                'mse': float(metrics['mse']),
                'r2_score': float(metrics['r2'])
            },
            'baseline_comparison': {
                'baseline_rmse': float(metrics['baseline_rmse']),
                'improvement_percentage': float(metrics['improvement_over_baseline'])
            },
            'classification_metrics': {
                'accuracy': float(metrics['accuracy']),
                'precision': float(metrics['precision']),
                'recall': float(metrics['recall'])
            }
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as pickle
        report_path = f'{output_dir}/evaluation_report.pkl'
        with open(report_path, 'wb') as f:
            pickle.dump(report, f)
        
        print(f"   ✅ Report saved: {report_path}")
        
        # Also save as readable text
        text_path = f'{output_dir}/evaluation_report.txt'
        with open(text_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("AGROWEATHER AI - MODEL EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("REGRESSION METRICS:\n")
            f.write(f"  RMSE: {metrics['rmse']:.3f} mm/day\n")
            f.write(f"  MAE:  {metrics['mae']:.3f} mm/day\n")
            f.write(f"  R² Score: {metrics['r2']:.3f}\n\n")
            
            f.write("BASELINE COMPARISON:\n")
            f.write(f"  Baseline RMSE: {metrics['baseline_rmse']:.3f} mm\n")
            f.write(f"  Model RMSE: {metrics['rmse']:.3f} mm\n")
            f.write(f"  Improvement: {metrics['improvement_over_baseline']:.1f}%\n\n")
            
            f.write("CLASSIFICATION METRICS:\n")
            f.write(f"  Accuracy: {metrics['accuracy']*100:.1f}%\n")
            f.write(f"  Precision: {metrics['precision']*100:.1f}%\n")
            f.write(f"  Recall: {metrics['recall']*100:.1f}%\n")
        
        print(f"   ✅ Text report saved: {text_path}")
    
    def evaluate_model(self, 
                      model_path: str = 'models/saved/rainfall_lstm_best.pth',
                      data_dir: str = 'data/processed',
                      output_dir: str = 'outputs') -> Dict[str, Any]:
        """
        Complete model evaluation pipeline
        
        Args:
            model_path: Path to saved model
            data_dir: Directory containing processed data
            output_dir: Directory to save outputs
            
        Returns:
            Evaluation metrics dictionary
        """
        print("🌦️  AgroWeather AI - Model Evaluation")
        print("="*70)
        
        # Load test data
        X_test, y_test, target_scaler, metadata = self.load_test_data(data_dir)
        
        # Load trained model
        model = self.load_trained_model(model_path, metadata)
        
        # Make predictions
        predictions = self.make_predictions(model, X_test)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, predictions, target_scaler)
        
        # Print metrics
        self.print_metrics(metrics)
        
        # Create visualizations
        self.plot_predictions_vs_actual(f'{output_dir}/plots', metrics)
        self.plot_error_by_rainfall_intensity(f'{output_dir}/plots', metrics)
        
        # Save report
        self.save_evaluation_report(f'{output_dir}/reports', metrics)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE!")
        print("="*70)
        
        return metrics