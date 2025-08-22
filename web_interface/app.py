"""
CUDA Finance Tool Web Interface

This Flask application provides a REST API and web interface for the CUDA Finance Tool,
enabling option pricing, risk analysis, and job management through HTTP endpoints.

Author: CUDA Finance Tool Team
Version: 1.0.0
License: MIT
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import json
import os
import sys
import subprocess
import tempfile
import threading
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'cuda_finance_tool_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for job tracking
jobs = {}
job_counter = 0

class OptionPricingJob:
    def __init__(self, job_id, parameters):
        self.job_id = job_id
        self.parameters = parameters
        self.status = 'pending'
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.progress = 0

    def to_dict(self):
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'result': self.result,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

def run_cuda_pricing(parameters):
    """Run CUDA option pricing with given parameters"""
    try:
        # Create temporary input file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(parameters, f)
            input_file = f.name

        # Run CUDA executable
        cmd = ['./bin/cuda_finance_tool', '--input', input_file, '--output', 'json']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Clean up input file
        os.unlink(input_file)

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {'error': result.stderr}

    except subprocess.TimeoutExpired:
        return {'error': 'Computation timed out'}
    except Exception as e:
        return {'error': str(e)}

def process_job(job):
    """Process a job in a separate thread"""
    job.status = 'running'
    job.start_time = datetime.now()
    
    try:
        # Simulate progress updates
        for i in range(10):
            time.sleep(0.5)  # Simulate computation time
            job.progress = (i + 1) * 10
            
        # Run actual computation
        job.result = run_cuda_pricing(job.parameters)
        job.status = 'completed'
        
    except Exception as e:
        job.status = 'failed'
        job.error = str(e)
    
    job.end_time = datetime.now()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/pricing', methods=['POST'])
def create_pricing_job():
    """Create a new option pricing job"""
    global job_counter
    
    try:
        data = request.get_json()
        
        # Validate required parameters
        required_fields = ['stock_prices', 'strike_prices', 'volatilities', 
                         'time_to_maturity', 'risk_free_rates']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Create job
        job_counter += 1
        job_id = f"job_{job_counter}"
        
        job = OptionPricingJob(job_id, data)
        jobs[job_id] = job
        
        # Start processing in background
        thread = threading.Thread(target=process_job, args=(job,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'created',
            'message': 'Job created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Get job status and results"""
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    return jsonify(job.to_dict())

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [job.to_dict() for job in jobs.values()]
    })

@app.route('/api/pricing/quick', methods=['POST'])
def quick_pricing():
    """Quick option pricing without job creation"""
    try:
        data = request.get_json()
        
        # Validate parameters
        if not all(field in data for field in ['stock_price', 'strike_price', 
                                              'volatility', 'time_to_maturity', 'risk_free_rate']):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Create parameters for single option
        parameters = {
            'num_options': 1,
            'stock_prices': [data['stock_price']],
            'strike_prices': [data['strike_price']],
            'volatilities': [data['volatility']],
            'time_to_maturity': [data['time_to_maturity']],
            'risk_free_rates': [data['risk_free_rate']]
        }
        
        # Run computation
        result = run_cuda_pricing(parameters)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify({
            'option_price': result.get('option_prices', [0])[0],
            'delta': result.get('deltas', [0])[0],
            'gamma': result.get('gammas', [0])[0]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/risk/var', methods=['POST'])
def calculate_var():
    """Calculate Value at Risk"""
    try:
        data = request.get_json()
        
        # Validate parameters
        if not all(field in data for field in ['portfolio_values', 'confidence_level']):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Calculate VaR (simplified)
        portfolio_values = data['portfolio_values']
        confidence_level = data['confidence_level']
        
        # Sort values and find VaR
        sorted_values = sorted(portfolio_values)
        var_index = int((1 - confidence_level) * len(sorted_values))
        var = sorted_values[var_index] if var_index < len(sorted_values) else sorted_values[-1]
        
        # Calculate CVaR
        cvar_values = [v for v in sorted_values if v <= var]
        cvar = sum(cvar_values) / len(cvar_values) if cvar_values else var
        
        return jsonify({
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/options/types', methods=['GET'])
def get_option_types():
    """Get available option types"""
    option_types = [
        {'id': 0, 'name': 'European Call'},
        {'id': 1, 'name': 'European Put'},
        {'id': 2, 'name': 'American Call'},
        {'id': 3, 'name': 'American Put'},
        {'id': 4, 'name': 'Barrier Up-and-Out'},
        {'id': 5, 'name': 'Barrier Down-and-Out'},
        {'id': 6, 'name': 'Barrier Up-and-In'},
        {'id': 7, 'name': 'Barrier Down-and-In'},
        {'id': 8, 'name': 'Asian Call'},
        {'id': 9, 'name': 'Asian Put'},
        {'id': 10, 'name': 'Basket Call'},
        {'id': 11, 'name': 'Basket Put'}
    ]
    
    return jsonify({'option_types': option_types})

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status and GPU information"""
    try:
        # Check if CUDA executable exists
        cuda_exists = os.path.exists('./bin/cuda_finance_tool')
        
        # Get GPU information (simplified)
        gpu_info = {
            'available': cuda_exists,
            'name': 'NVIDIA GPU' if cuda_exists else 'Not available',
            'memory': '8GB' if cuda_exists else 'Unknown'
        }
        
        return jsonify({
            'cuda_available': cuda_exists,
            'gpu_info': gpu_info,
            'active_jobs': len([j for j in jobs.values() if j.status == 'running'])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000) 