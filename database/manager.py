"""
Database Manager for CUDA Finance Tool

This module provides database management capabilities including CRUD operations
for option pricing results, risk calculations, job history, and performance metrics.

Author: CUDA Finance Tool Team
Version: 1.0.0
License: MIT
"""

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Any, Optional
import json
from datetime import datetime, timedelta
from .models import Base, OptionPricingResult, RiskCalculation, JobHistory, PerformanceMetrics, SystemEvents, MarketData, Portfolio
from logging_config import log_error

class DatabaseManager:
    """Database manager for CUDA Finance Tool"""
    
    def __init__(self, connection_string: str = None):
        """Initialize database connection"""
        if connection_string is None:
            # Default to SQLite for development
            connection_string = "sqlite:///cuda_finance.db"
        
        self.engine = create_engine(connection_string)
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def close_session(self, session):
        """Close database session"""
        session.close()
    
    # Option Pricing Results
    def save_option_pricing_result(self, result_data: Dict[str, Any]) -> Optional[int]:
        """Save option pricing result to database"""
        session = self.get_session()
        try:
            result = OptionPricingResult(**result_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save option pricing result: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_option_pricing_results(self, job_id: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get option pricing results from database"""
        session = self.get_session()
        try:
            query = session.query(OptionPricingResult)
            if job_id:
                query = query.filter(OptionPricingResult.job_id == job_id)
            query = query.order_by(OptionPricingResult.created_at.desc()).limit(limit)
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get option pricing results: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Risk Calculations
    def save_risk_calculation(self, risk_data: Dict[str, Any]) -> Optional[int]:
        """Save risk calculation result to database"""
        session = self.get_session()
        try:
            # Convert JSON fields
            if 'stress_test_results' in risk_data and isinstance(risk_data['stress_test_results'], dict):
                risk_data['stress_test_results'] = json.dumps(risk_data['stress_test_results'])
            if 'scenario_results' in risk_data and isinstance(risk_data['scenario_results'], dict):
                risk_data['scenario_results'] = json.dumps(risk_data['scenario_results'])
            
            result = RiskCalculation(**risk_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save risk calculation: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_risk_calculations(self, risk_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get risk calculations from database"""
        session = self.get_session()
        try:
            query = session.query(RiskCalculation)
            if risk_type:
                query = query.filter(RiskCalculation.risk_type == risk_type)
            query = query.order_by(RiskCalculation.created_at.desc()).limit(limit)
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get risk calculations: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Job History
    def save_job_history(self, job_data: Dict[str, Any]) -> Optional[int]:
        """Save job history to database"""
        session = self.get_session()
        try:
            # Convert JSON fields
            if 'parameters' in job_data and isinstance(job_data['parameters'], dict):
                job_data['parameters'] = json.dumps(job_data['parameters'])
            if 'result_summary' in job_data and isinstance(job_data['result_summary'], dict):
                job_data['result_summary'] = json.dumps(job_data['result_summary'])
            
            result = JobHistory(**job_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save job history: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def update_job_status(self, job_id: str, status: str, progress: int = None, 
                         error_message: str = None, result_summary: Dict[str, Any] = None):
        """Update job status"""
        session = self.get_session()
        try:
            job = session.query(JobHistory).filter(JobHistory.job_id == job_id).first()
            if job:
                job.status = status
                if progress is not None:
                    job.progress = progress
                if error_message is not None:
                    job.error_message = error_message
                if result_summary is not None:
                    job.result_summary = json.dumps(result_summary)
                
                if status in ['running', 'completed', 'failed']:
                    if status == 'running':
                        job.started_at = datetime.utcnow()
                    else:
                        job.completed_at = datetime.utcnow()
                
                session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to update job status: {str(e)}")
        finally:
            self.close_session(session)
    
    def get_job_history(self, job_id: str = None, status: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get job history from database"""
        session = self.get_session()
        try:
            query = session.query(JobHistory)
            if job_id:
                query = query.filter(JobHistory.job_id == job_id)
            if status:
                query = query.filter(JobHistory.status == status)
            query = query.order_by(JobHistory.created_at.desc()).limit(limit)
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get job history: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Performance Metrics
    def save_performance_metric(self, metric_data: Dict[str, Any]) -> Optional[int]:
        """Save performance metric to database"""
        session = self.get_session()
        try:
            # Convert context to JSON if it's a dict
            if 'context' in metric_data and isinstance(metric_data['context'], dict):
                metric_data['context'] = json.dumps(metric_data['context'])
            
            result = PerformanceMetrics(**metric_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save performance metric: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_performance_metrics(self, metric_name: str = None, 
                              time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance metrics from database"""
        session = self.get_session()
        try:
            query = session.query(PerformanceMetrics)
            if metric_name:
                query = query.filter(PerformanceMetrics.metric_name == metric_name)
            
            # Filter by time range
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            query = query.filter(PerformanceMetrics.timestamp >= cutoff_time)
            query = query.order_by(PerformanceMetrics.timestamp.desc())
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get performance metrics: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # System Events
    def save_system_event(self, event_data: Dict[str, Any]) -> Optional[int]:
        """Save system event to database"""
        session = self.get_session()
        try:
            # Convert details to JSON if it's a dict
            if 'details' in event_data and isinstance(event_data['details'], dict):
                event_data['details'] = json.dumps(event_data['details'])
            
            result = SystemEvents(**event_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save system event: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_system_events(self, event_type: str = None, severity: str = None, 
                         time_range_hours: int = 24) -> List[Dict[str, Any]]:
        """Get system events from database"""
        session = self.get_session()
        try:
            query = session.query(SystemEvents)
            if event_type:
                query = query.filter(SystemEvents.event_type == event_type)
            if severity:
                query = query.filter(SystemEvents.severity == severity)
            
            # Filter by time range
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            query = query.filter(SystemEvents.timestamp >= cutoff_time)
            query = query.order_by(SystemEvents.timestamp.desc())
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get system events: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Market Data
    def save_market_data(self, market_data: Dict[str, Any]) -> Optional[int]:
        """Save market data to database"""
        session = self.get_session()
        try:
            result = MarketData(**market_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save market data: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_market_data(self, symbol: str = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get market data from database"""
        session = self.get_session()
        try:
            query = session.query(MarketData)
            if symbol:
                query = query.filter(MarketData.symbol == symbol)
            query = query.order_by(MarketData.timestamp.desc()).limit(limit)
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get market data: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Portfolios
    def save_portfolio(self, portfolio_data: Dict[str, Any]) -> Optional[int]:
        """Save portfolio to database"""
        session = self.get_session()
        try:
            # Convert JSON fields
            if 'positions' in portfolio_data and isinstance(portfolio_data['positions'], dict):
                portfolio_data['positions'] = json.dumps(portfolio_data['positions'])
            if 'risk_metrics' in portfolio_data and isinstance(portfolio_data['risk_metrics'], dict):
                portfolio_data['risk_metrics'] = json.dumps(portfolio_data['risk_metrics'])
            
            result = Portfolio(**portfolio_data)
            session.add(result)
            session.commit()
            return result.id
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to save portfolio: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def get_portfolios(self, portfolio_id: int = None) -> List[Dict[str, Any]]:
        """Get portfolios from database"""
        session = self.get_session()
        try:
            query = session.query(Portfolio)
            if portfolio_id:
                query = query.filter(Portfolio.id == portfolio_id)
            query = query.order_by(Portfolio.updated_at.desc())
            
            results = query.all()
            return [result.to_dict() for result in results]
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get portfolios: {str(e)}")
            return []
        finally:
            self.close_session(session)
    
    # Analytics and Reporting
    def get_analytics_summary(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """Get analytics summary for the specified time range"""
        session = self.get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            
            # Count jobs by status
            job_counts = session.query(JobHistory.status, session.query(JobHistory).count()).filter(
                JobHistory.created_at >= cutoff_time
            ).group_by(JobHistory.status).all()
            
            # Average execution time for option pricing
            avg_execution_time = session.query(session.query(OptionPricingResult.execution_time_ms).filter(
                OptionPricingResult.created_at >= cutoff_time
            ).avg()).scalar()
            
            # Error count
            error_count = session.query(SystemEvents).filter(
                SystemEvents.severity == 'error',
                SystemEvents.timestamp >= cutoff_time
            ).count()
            
            return {
                'job_counts': dict(job_counts),
                'avg_execution_time_ms': avg_execution_time or 0,
                'error_count': error_count,
                'time_range_hours': time_range_hours
            }
        except SQLAlchemyError as e:
            log_error("database_error", f"Failed to get analytics summary: {str(e)}")
            return {}
        finally:
            self.close_session(session)
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data from database"""
        session = self.get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Delete old records
            session.query(OptionPricingResult).filter(
                OptionPricingResult.created_at < cutoff_time
            ).delete()
            
            session.query(RiskCalculation).filter(
                RiskCalculation.created_at < cutoff_time
            ).delete()
            
            session.query(PerformanceMetrics).filter(
                PerformanceMetrics.timestamp < cutoff_time
            ).delete()
            
            session.query(SystemEvents).filter(
                SystemEvents.timestamp < cutoff_time
            ).delete()
            
            session.query(MarketData).filter(
                MarketData.timestamp < cutoff_time
            ).delete()
            
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            log_error("database_error", f"Failed to cleanup old data: {str(e)}")
        finally:
            self.close_session(session)

# Global database manager instance
db_manager = DatabaseManager() 