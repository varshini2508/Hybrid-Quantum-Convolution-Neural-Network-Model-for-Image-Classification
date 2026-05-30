from app import app
import os

def create_directories():
    """Create all necessary directories automatically"""
    directories = [
        'static/uploads',
        'trained_models', 
        'qiskit_datasets',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

if __name__ == '__main__':
    # Create all necessary directories
    create_directories()
    
    print("🚀 Starting Hybrid Quantum-Classical CNN")
    print("🔬 Using Qiskit for Quantum Machine Learning")
    print("🌐 Access at: http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True)