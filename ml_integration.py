"""
Machine Learning Integration Module for CUDA Finance Tool

This module provides machine learning capabilities for volatility forecasting,
pattern recognition, and trading signal generation. It integrates with the
CUDA-based financial computing platform to provide AI-powered insights.

Author: CUDA Finance Tool Team
Version: 1.0.0
License: MIT
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from logging_config import log_error, performance_monitor

class VolatilityForecaster:
    """Machine learning model for volatility forecasting"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for volatility forecasting"""
        df = market_data.copy()
        
        # Technical indicators
        df['returns'] = df['price'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['price_ma_5'] = df['price'].rolling(window=5).mean()
        df['price_ma_20'] = df['price'].rolling(window=20).mean()
        df['rsi'] = self._calculate_rsi(df['price'], window=14)
        df['bollinger_upper'] = df['price_ma_20'] + 2 * df['volatility']
        df['bollinger_lower'] = df['price_ma_20'] - 2 * df['volatility']
        
        # Market microstructure features
        df['volume_ma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        df['price_volume_corr'] = df['price'].rolling(10).corr(df['volume'])
        
        # Time-based features
        df['day_of_week'] = pd.to_datetime(df.index).dayofweek
        df['month'] = pd.to_datetime(df.index).month
        df['is_month_end'] = pd.to_datetime(df.index).is_month_end.astype(int)
        
        # Lagged features
        for lag in [1, 2, 3, 5, 10]:
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, market_data: pd.DataFrame, target_horizon: int = 5):
        """Train the volatility forecasting model"""
        try:
            performance_monitor.start_timer('volatility_forecast_training')
            
            # Prepare features
            df = self.prepare_features(market_data)
            
            # Define target (future volatility)
            df['target_volatility'] = df['volatility'].shift(-target_horizon)
            df = df.dropna()
            
            # Select features
            feature_columns = [col for col in df.columns if col not in 
                             ['target_volatility', 'price', 'volume']]
            self.feature_names = feature_columns
            
            X = df[feature_columns]
            y = df['target_volatility']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            elif self.model_type == 'gradient_boosting':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            self.is_trained = True
            
            performance_monitor.end_timer('volatility_forecast_training', {
                'mse': mse,
                'mae': mae,
                'feature_count': len(feature_columns)
            })
            
            logging.info(f"Volatility forecaster trained - MSE: {mse:.6f}, MAE: {mae:.6f}")
            
        except Exception as e:
            log_error("ml_training_error", f"Failed to train volatility forecaster: {str(e)}")
            raise
    
    def predict_volatility(self, market_data: pd.DataFrame, horizon: int = 5) -> float:
        """Predict future volatility"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare features
            df = self.prepare_features(market_data)
            
            # Get latest features
            latest_features = df[self.feature_names].iloc[-1:].values
            latest_features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            prediction = self.model.predict(latest_features_scaled)[0]
            
            return max(0.01, prediction)  # Ensure positive volatility
            
        except Exception as e:
            log_error("ml_prediction_error", f"Failed to predict volatility: {str(e)}")
            return 0.2  # Return default volatility
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.is_trained:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            joblib.dump(model_data, filepath)
            logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = True
            logging.info(f"Model loaded from {filepath}")

class PatternRecognizer:
    """Pattern recognition for trading signals"""
    
    def __init__(self):
        self.patterns = {
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'head_and_shoulders': self._detect_head_and_shoulders,
            'triangle': self._detect_triangle,
            'breakout': self._detect_breakout
        }
    
    def detect_patterns(self, prices: pd.Series, window: int = 50) -> Dict[str, bool]:
        """Detect technical patterns in price data"""
        if len(prices) < window:
            return {}
        
        recent_prices = prices.tail(window)
        detected_patterns = {}
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                detected_patterns[pattern_name] = pattern_func(recent_prices)
            except Exception as e:
                log_error("pattern_detection_error", f"Error detecting {pattern_name}: {str(e)}")
                detected_patterns[pattern_name] = False
        
        return detected_patterns
    
    def _detect_double_top(self, prices: pd.Series) -> bool:
        """Detect double top pattern"""
        peaks = self._find_peaks(prices)
        if len(peaks) < 2:
            return False
        
        # Check if two peaks are similar in height
        peak1, peak2 = peaks[-2:]
        height_diff = abs(prices.iloc[peak1] - prices.iloc[peak2]) / prices.iloc[peak1]
        
        return height_diff < 0.02  # 2% tolerance
    
    def _detect_double_bottom(self, prices: pd.Series) -> bool:
        """Detect double bottom pattern"""
        troughs = self._find_troughs(prices)
        if len(troughs) < 2:
            return False
        
        # Check if two troughs are similar in depth
        trough1, trough2 = troughs[-2:]
        depth_diff = abs(prices.iloc[trough1] - prices.iloc[trough2]) / prices.iloc[trough1]
        
        return depth_diff < 0.02  # 2% tolerance
    
    def _detect_head_and_shoulders(self, prices: pd.Series) -> bool:
        """Detect head and shoulders pattern"""
        peaks = self._find_peaks(prices)
        if len(peaks) < 3:
            return False
        
        # Check if middle peak is higher than side peaks
        left_peak, middle_peak, right_peak = peaks[-3:]
        left_height = prices.iloc[left_peak]
        middle_height = prices.iloc[middle_peak]
        right_height = prices.iloc[right_peak]
        
        return (middle_height > left_height and 
                middle_height > right_height and
                abs(left_height - right_height) / left_height < 0.05)
    
    def _detect_triangle(self, prices: pd.Series) -> bool:
        """Detect triangle pattern"""
        # Simplified triangle detection
        highs = prices.rolling(window=5).max()
        lows = prices.rolling(window=5).min()
        
        # Check if highs are decreasing and lows are increasing
        high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
        low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
        
        return high_slope < 0 and low_slope > 0
    
    def _detect_breakout(self, prices: pd.Series) -> bool:
        """Detect breakout pattern"""
        # Check if price breaks above resistance or below support
        resistance = prices.rolling(window=20).max().iloc[-2]
        support = prices.rolling(window=20).min().iloc[-2]
        current_price = prices.iloc[-1]
        
        return (current_price > resistance * 1.01 or 
                current_price < support * 0.99)
    
    def _find_peaks(self, prices: pd.Series) -> List[int]:
        """Find peaks in price series"""
        peaks = []
        for i in range(1, len(prices) - 1):
            if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                peaks.append(i)
        return peaks
    
    def _find_troughs(self, prices: pd.Series) -> List[int]:
        """Find troughs in price series"""
        troughs = []
        for i in range(1, len(prices) - 1):
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                troughs.append(i)
        return troughs

class TradingSignalGenerator:
    """Generate trading signals based on ML predictions and patterns"""
    
    def __init__(self, volatility_forecaster: VolatilityForecaster, 
                 pattern_recognizer: PatternRecognizer):
        self.volatility_forecaster = volatility_forecaster
        self.pattern_recognizer = pattern_recognizer
        self.signal_history = []
    
    def generate_signals(self, market_data: pd.DataFrame) -> Dict[str, any]:
        """Generate trading signals"""
        try:
            performance_monitor.start_timer('signal_generation')
            
            # Predict volatility
            predicted_volatility = self.volatility_forecaster.predict_volatility(market_data)
            
            # Detect patterns
            patterns = self.pattern_recognizer.detect_patterns(market_data['price'])
            
            # Generate signals
            signals = {
                'volatility_forecast': predicted_volatility,
                'patterns_detected': patterns,
                'trading_recommendation': self._generate_recommendation(
                    predicted_volatility, patterns, market_data
                ),
                'confidence_score': self._calculate_confidence(
                    predicted_volatility, patterns
                ),
                'timestamp': datetime.now().isoformat()
            }
            
            self.signal_history.append(signals)
            
            performance_monitor.end_timer('signal_generation', {
                'predicted_volatility': predicted_volatility,
                'patterns_count': sum(patterns.values())
            })
            
            return signals
            
        except Exception as e:
            log_error("signal_generation_error", f"Failed to generate signals: {str(e)}")
            return {}
    
    def _generate_recommendation(self, volatility: float, patterns: Dict[str, bool], 
                                market_data: pd.DataFrame) -> str:
        """Generate trading recommendation"""
        current_price = market_data['price'].iloc[-1]
        price_change = market_data['price'].pct_change().iloc[-1]
        
        # High volatility suggests caution
        if volatility > 0.3:
            return "HOLD - High volatility detected"
        
        # Pattern-based recommendations
        if patterns.get('double_top', False):
            return "SELL - Double top pattern detected"
        elif patterns.get('double_bottom', False):
            return "BUY - Double bottom pattern detected"
        elif patterns.get('head_and_shoulders', False):
            return "SELL - Head and shoulders pattern detected"
        elif patterns.get('breakout', False):
            if price_change > 0:
                return "BUY - Bullish breakout detected"
            else:
                return "SELL - Bearish breakout detected"
        
        # Volatility-based recommendations
        if volatility < 0.1:
            return "BUY - Low volatility, potential for upside"
        elif volatility > 0.2:
            return "HOLD - Moderate volatility, wait for clearer signals"
        
        return "HOLD - No clear signals"
    
    def _calculate_confidence(self, volatility: float, patterns: Dict[str, bool]) -> float:
        """Calculate confidence score for signals"""
        confidence = 0.5  # Base confidence
        
        # Adjust for volatility
        if 0.1 <= volatility <= 0.2:
            confidence += 0.2
        elif volatility > 0.3:
            confidence -= 0.3
        
        # Adjust for patterns
        pattern_count = sum(patterns.values())
        if pattern_count == 1:
            confidence += 0.2
        elif pattern_count > 1:
            confidence += 0.3
        
        return min(1.0, max(0.0, confidence))

class MLModelManager:
    """Manager for all ML models"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.volatility_forecaster = VolatilityForecaster()
        self.pattern_recognizer = PatternRecognizer()
        self.signal_generator = TradingSignalGenerator(
            self.volatility_forecaster, self.pattern_recognizer
        )
    
    def train_all_models(self, market_data: pd.DataFrame):
        """Train all ML models"""
        try:
            logging.info("Training ML models...")
            
            # Train volatility forecaster
            self.volatility_forecaster.train(market_data)
            
            # Save models
            self.save_models()
            
            logging.info("ML models trained successfully")
            
        except Exception as e:
            log_error("ml_training_error", f"Failed to train ML models: {str(e)}")
            raise
    
    def load_models(self):
        """Load trained models"""
        try:
            volatility_model_path = os.path.join(self.models_dir, "volatility_forecaster.pkl")
            if os.path.exists(volatility_model_path):
                self.volatility_forecaster.load_model(volatility_model_path)
                logging.info("ML models loaded successfully")
            else:
                logging.warning("No trained models found")
                
        except Exception as e:
            log_error("ml_loading_error", f"Failed to load ML models: {str(e)}")
    
    def save_models(self):
        """Save trained models"""
        try:
            volatility_model_path = os.path.join(self.models_dir, "volatility_forecaster.pkl")
            self.volatility_forecaster.save_model(volatility_model_path)
            logging.info("ML models saved successfully")
            
        except Exception as e:
            log_error("ml_saving_error", f"Failed to save ML models: {str(e)}")
    
    def get_trading_signals(self, market_data: pd.DataFrame) -> Dict[str, any]:
        """Get trading signals from all models"""
        return self.signal_generator.generate_signals(market_data)
    
    def update_models(self, new_data: pd.DataFrame):
        """Update models with new data"""
        try:
            # Retrain models with updated data
            self.train_all_models(new_data)
            logging.info("ML models updated successfully")
            
        except Exception as e:
            log_error("ml_update_error", f"Failed to update ML models: {str(e)}")

# Global ML manager instance
ml_manager = MLModelManager() 