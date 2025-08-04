from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

Base = declarative_base()

class OptionPricingResult(Base):
    """Model for storing option pricing results"""
    __tablename__ = 'option_pricing_results'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), nullable=False)
    option_type = Column(String(50), nullable=False)
    stock_price = Column(Float, nullable=False)
    strike_price = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    time_to_maturity = Column(Float, nullable=False)
    risk_free_rate = Column(Float, nullable=False)
    option_price = Column(Float, nullable=False)
    delta = Column(Float)
    gamma = Column(Float)
    vega = Column(Float)
    theta = Column(Float)
    execution_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'option_type': self.option_type,
            'stock_price': self.stock_price,
            'strike_price': self.strike_price,
            'volatility': self.volatility,
            'time_to_maturity': self.time_to_maturity,
            'risk_free_rate': self.risk_free_rate,
            'option_price': self.option_price,
            'delta': self.delta,
            'gamma': self.gamma,
            'vega': self.vega,
            'theta': self.theta,
            'execution_time_ms': self.execution_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class RiskCalculation(Base):
    """Model for storing risk calculation results"""
    __tablename__ = 'risk_calculations'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), nullable=False)
    risk_type = Column(String(50), nullable=False)  # VaR, CVaR, etc.
    portfolio_size = Column(Integer, nullable=False)
    confidence_level = Column(Float, nullable=False)
    var_value = Column(Float)
    cvar_value = Column(Float)
    stress_test_results = Column(Text)  # JSON string
    scenario_results = Column(Text)     # JSON string
    execution_time_ms = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'risk_type': self.risk_type,
            'portfolio_size': self.portfolio_size,
            'confidence_level': self.confidence_level,
            'var_value': self.var_value,
            'cvar_value': self.cvar_value,
            'stress_test_results': json.loads(self.stress_test_results) if self.stress_test_results else None,
            'scenario_results': json.loads(self.scenario_results) if self.scenario_results else None,
            'execution_time_ms': self.execution_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class JobHistory(Base):
    """Model for storing job execution history"""
    __tablename__ = 'job_history'
    
    id = Column(Integer, primary_key=True)
    job_id = Column(String(50), unique=True, nullable=False)
    job_type = Column(String(50), nullable=False)  # pricing, risk, batch
    status = Column(String(20), nullable=False)    # pending, running, completed, failed
    parameters = Column(Text)  # JSON string
    result_summary = Column(Text)  # JSON string
    error_message = Column(Text)
    progress = Column(Integer, default=0)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'job_id': self.job_id,
            'job_type': self.job_type,
            'status': self.status,
            'parameters': json.loads(self.parameters) if self.parameters else None,
            'result_summary': json.loads(self.result_summary) if self.result_summary else None,
            'error_message': self.error_message,
            'progress': self.progress,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class PerformanceMetrics(Base):
    """Model for storing performance metrics"""
    __tablename__ = 'performance_metrics'
    
    id = Column(Integer, primary_key=True)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20), default='ms')
    context = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'context': json.loads(self.context) if self.context else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class SystemEvents(Base):
    """Model for storing system events and errors"""
    __tablename__ = 'system_events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False)  # cuda_event, error, alert
    severity = Column(String(20), default='info')    # info, warning, error, critical
    message = Column(Text, nullable=False)
    details = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'event_type': self.event_type,
            'severity': self.severity,
            'message': self.message,
            'details': json.loads(self.details) if self.details else None,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class MarketData(Base):
    """Model for storing market data"""
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False)
    price = Column(Float, nullable=False)
    volume = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    data_source = Column(String(50), default='manual')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'price': self.price,
            'volume': self.volume,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'data_source': self.data_source
        }

class Portfolio(Base):
    """Model for storing portfolio information"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    positions = Column(Text)  # JSON string of portfolio positions
    total_value = Column(Float)
    risk_metrics = Column(Text)  # JSON string of risk metrics
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'positions': json.loads(self.positions) if self.positions else None,
            'total_value': self.total_value,
            'risk_metrics': json.loads(self.risk_metrics) if self.risk_metrics else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        } 