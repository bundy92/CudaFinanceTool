"""
Logging Configuration for CUDA Finance Tool

This module provides comprehensive logging capabilities including structured logging,
performance monitoring, error tracking, and alert management for the CUDA Finance Tool.

Author: CUDA Finance Tool Team
Version: 1.0.0
License: MIT
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Dict, Any
import json
import structlog
from structlog.stdlib import LoggerFactory

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

class CUDAFinanceLogger:
    """Comprehensive logging system for CUDA Finance Tool"""
    
    def __init__(self, name: str = "cuda_finance", log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> structlog.BoundLogger:
        """Setup structured logger with multiple handlers"""
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=self.log_level
        )
        
        # Create structlog logger
        logger = structlog.get_logger(self.name)
        
        # Add file handlers
        self._add_file_handlers(logger)
        
        return logger
    
    def _add_file_handlers(self, logger: structlog.BoundLogger):
        """Add file handlers for different log levels"""
        
        # Application logs
        app_handler = logging.handlers.RotatingFileHandler(
            "logs/cuda_finance.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        app_handler.setLevel(logging.INFO)
        
        # Error logs
        error_handler = logging.handlers.RotatingFileHandler(
            "logs/errors.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        
        # Performance logs
        perf_handler = logging.handlers.RotatingFileHandler(
            "logs/performance.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        perf_handler.setLevel(logging.INFO)
        
        # Add handlers to root logger
        logging.getLogger().addHandler(app_handler)
        logging.getLogger().addHandler(error_handler)
        logging.getLogger().addHandler(perf_handler)
    
    def log_cuda_event(self, event_type: str, details: Dict[str, Any]):
        """Log CUDA-specific events"""
        self.logger.info(
            "cuda_event",
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            **details
        )
    
    def log_option_pricing(self, option_type: str, parameters: Dict[str, Any], 
                          result: Dict[str, Any], execution_time: float):
        """Log option pricing events"""
        self.logger.info(
            "option_pricing",
            option_type=option_type,
            parameters=parameters,
            result=result,
            execution_time_ms=execution_time * 1000,
            timestamp=datetime.now().isoformat()
        )
    
    def log_risk_calculation(self, risk_type: str, portfolio_size: int, 
                            confidence_level: float, result: Dict[str, Any]):
        """Log risk calculation events"""
        self.logger.info(
            "risk_calculation",
            risk_type=risk_type,
            portfolio_size=portfolio_size,
            confidence_level=confidence_level,
            result=result,
            timestamp=datetime.now().isoformat()
        )
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str = "ms", context: Dict[str, Any] = None):
        """Log performance metrics"""
        log_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }
        if context:
            log_data.update(context)
        
        self.logger.info("performance_metric", **log_data)
    
    def log_error(self, error_type: str, error_message: str, 
                  stack_trace: str = None, context: Dict[str, Any] = None):
        """Log error events"""
        log_data = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        if stack_trace:
            log_data["stack_trace"] = stack_trace
        if context:
            log_data.update(context)
        
        self.logger.error("error", **log_data)
    
    def log_api_request(self, endpoint: str, method: str, status_code: int,
                       response_time: float, user_agent: str = None):
        """Log API requests"""
        self.logger.info(
            "api_request",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time_ms=response_time * 1000,
            user_agent=user_agent,
            timestamp=datetime.now().isoformat()
        )
    
    def log_job_event(self, job_id: str, event_type: str, 
                     details: Dict[str, Any] = None):
        """Log job-related events"""
        log_data = {
            "job_id": job_id,
            "event_type": event_type,
            "timestamp": datetime.now().isoformat()
        }
        if details:
            log_data.update(details)
        
        self.logger.info("job_event", **log_data)

class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self, logger: CUDAFinanceLogger):
        self.logger = logger
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start a performance timer"""
        self.metrics[name] = {
            'start_time': datetime.now(),
            'end_time': None
        }
    
    def end_timer(self, name: str, context: Dict[str, Any] = None):
        """End a performance timer and log the result"""
        if name not in self.metrics:
            self.logger.log_error("performance_monitor", f"Timer '{name}' not found")
            return
        
        self.metrics[name]['end_time'] = datetime.now()
        duration = (self.metrics[name]['end_time'] - self.metrics[name]['start_time']).total_seconds()
        
        self.logger.log_performance_metric(
            metric_name=name,
            value=duration * 1000,  # Convert to milliseconds
            unit="ms",
            context=context
        )
        
        del self.metrics[name]
    
    def record_metric(self, name: str, value: float, unit: str = "count", 
                     context: Dict[str, Any] = None):
        """Record a custom metric"""
        self.logger.log_performance_metric(
            metric_name=name,
            value=value,
            unit=unit,
            context=context
        )

class AlertManager:
    """Alert management system"""
    
    def __init__(self, logger: CUDAFinanceLogger):
        self.logger = logger
        self.alert_thresholds = {
            'execution_time_ms': 5000,  # 5 seconds
            'memory_usage_mb': 1024,    # 1GB
            'error_rate': 0.1,          # 10%
            'gpu_utilization': 0.9      # 90%
        }
    
    def check_threshold(self, metric_name: str, value: float, 
                       threshold: float = None) -> bool:
        """Check if a metric exceeds its threshold"""
        if threshold is None:
            threshold = self.alert_thresholds.get(metric_name, float('inf'))
        
        if value > threshold:
            self.logger.log_error(
                error_type="threshold_exceeded",
                error_message=f"Metric '{metric_name}' exceeded threshold",
                context={
                    "metric_name": metric_name,
                    "value": value,
                    "threshold": threshold
                }
            )
            return True
        return False
    
    def alert(self, alert_type: str, message: str, severity: str = "warning",
              context: Dict[str, Any] = None):
        """Send an alert"""
        self.logger.log_error(
            error_type=f"alert_{alert_type}",
            error_message=message,
            context={
                "severity": severity,
                "alert_type": alert_type,
                **(context or {})
            }
        )

# Global logger instance
logger = CUDAFinanceLogger()
performance_monitor = PerformanceMonitor(logger)
alert_manager = AlertManager(logger)

# Convenience functions
def log_cuda_event(event_type: str, **details):
    """Log CUDA event"""
    logger.log_cuda_event(event_type, details)

def log_option_pricing(option_type: str, parameters: Dict[str, Any], 
                      result: Dict[str, Any], execution_time: float):
    """Log option pricing"""
    logger.log_option_pricing(option_type, parameters, result, execution_time)

def log_risk_calculation(risk_type: str, portfolio_size: int, 
                        confidence_level: float, result: Dict[str, Any]):
    """Log risk calculation"""
    logger.log_risk_calculation(risk_type, portfolio_size, confidence_level, result)

def log_error(error_type: str, error_message: str, **context):
    """Log error"""
    logger.log_error(error_type, error_message, context=context)

def log_api_request(endpoint: str, method: str, status_code: int,
                   response_time: float, **context):
    """Log API request"""
    logger.log_api_request(endpoint, method, status_code, response_time, **context)

def log_job_event(job_id: str, event_type: str, **details):
    """Log job event"""
    logger.log_job_event(job_id, event_type, details)

# Performance monitoring decorator
def monitor_performance(name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            performance_monitor.start_timer(name)
            try:
                result = func(*args, **kwargs)
                performance_monitor.end_timer(name, {"function": func.__name__})
                return result
            except Exception as e:
                performance_monitor.end_timer(name, {"function": func.__name__, "error": str(e)})
                raise
        return wrapper
    return decorator 