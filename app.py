# app.py 
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
import os
import numpy as np
from datetime import datetime
import time
import traceback
from functools import wraps

# Import custom modules
from quantum_model import QiskitHybridCNN
from dataset_manager import DatasetManager
from image_processor import ImageProcessor
from database_manager import DatabaseManager

def create_app_directories():
    """Create all necessary directories for the app"""
    directories = [
        'static/uploads',
        'trained_models',
        'qiskit_datasets',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

app = Flask(__name__)
app.secret_key = 'quantum-cnn-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize components
dataset_manager = DatasetManager()
image_processor = ImageProcessor()
db = DatabaseManager()

# Global variable to store the current model (avoids session storage)
current_model = None
current_training_results = None

# Create directories when app starts
create_app_directories()

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required', 'error')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    if 'user_id' in session and session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template('home.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard after login"""
    if session.get('role') == 'admin':
        return redirect(url_for('admin_dashboard'))
    return render_template('index.html')



# Update your existing index route to redirect to dashboard


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('All fields are required', 'error')
            return render_template('register.html')
        
        # Check if user already exists
        existing_user = db.execute_query(
            "SELECT id FROM users WHERE username = %s OR email = %s", 
            (username, email)
        )
        
        if existing_user:
            flash('Username or email already exists', 'error')
            return render_template('register.html')
        
        # Create user with 'pending' status
        user_id = db.execute_query(
            "INSERT INTO users (username, email, password, role, status) VALUES (%s, %s, %s, %s, %s)",
            (username, email, password, 'user', 'pending')  # In production, hash the password!
        )
        
        if user_id:
            flash('Registration successful! Please wait for admin approval.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = db.execute_query(
            "SELECT * FROM users WHERE username = %s AND password = %s", 
            (username, password)  # In production, use proper password hashing!
        )
        
        if user:
            user = user[0]  # Get first result
            if user['status'] == 'approved':
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['role'] = user['role']
                flash(f'Welcome back, {user["username"]}!', 'success')
                return redirect(url_for('dashboard'))
            elif user['status'] == 'pending':
                flash('Your account is pending admin approval', 'warning')
            elif user['status'] == 'rejected':
                flash('Your account has been rejected. Please contact admin.', 'error')
            elif user['status'] == 'hold':
                flash('Your account is on hold. Please contact admin.', 'warning')
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out successfully', 'success')
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    # Get all users except admin
    users = db.execute_query("SELECT * FROM users WHERE role != 'admin' ORDER BY created_at DESC")
    
    # Get stats
    stats = {
        'total_users': len(users),
        'pending_users': len([u for u in users if u['status'] == 'pending']),
        'approved_users': len([u for u in users if u['status'] == 'approved']),
        'rejected_users': len([u for u in users if u['status'] == 'rejected']),
        'hold_users': len([u for u in users if u['status'] == 'hold'])
    }
    
    return render_template('admin_dashboard.html', users=users, stats=stats)

@app.route('/admin/update_user_status', methods=['POST'])
@admin_required
def update_user_status():
    user_id = request.form.get('user_id')
    new_status = request.form.get('status')
    
    if user_id and new_status:
        try:
            result = db.execute_query(
                "UPDATE users SET status = %s WHERE id = %s",
                (new_status, user_id)
            )
            
            # Check if the update was successful
            if result is not None:
                flash(f'User status updated to {new_status}', 'success')
            else:
                flash('Failed to update user status - database error', 'error')
                
        except Exception as e:
            flash(f'Error updating user status: {str(e)}', 'error')
            print(f"Update error: {traceback.format_exc()}")
    
    return redirect(url_for('admin_dashboard'))

@app.route('/generate_dataset', methods=['GET', 'POST'])
@login_required
def generate_dataset():
    if request.method == 'POST':
        num_samples = int(request.form.get('num_samples', 1000))
        try:
            start_time = time.time()
            dataset_path = dataset_manager.create_synthetic_dataset(num_samples)
            generation_time = time.time() - start_time
            
            # Get dataset info
            dataset_info = dataset_manager.get_dataset_info(dataset_path)
            
            session['current_dataset'] = dataset_path
            session['dataset_info'] = dataset_info
            
            # Clear previous model when generating new dataset
            global current_model, current_training_results
            current_model = None
            current_training_results = None
            
            # Clear training results from session
            if 'training_results' in session:
                session.pop('training_results')
            if 'model_trained' in session:
                session.pop('model_trained')
            
            flash(f'✅ Dataset generated with {dataset_info["total_images"]} samples in {generation_time:.2f}s', 'success')
            
        except Exception as e:
            flash(f'❌ Error generating dataset: {str(e)}', 'error')
            print(f"Detailed error: {traceback.format_exc()}")
    
    # Show dataset info if available
    dataset_info = session.get('dataset_info', {})
    return render_template('generate_dataset.html', dataset_info=dataset_info)

@app.route('/train_model', methods=['GET', 'POST'])
@login_required
def train_model():
    global current_model, current_training_results
    
    # Clear training results when page loads (GET request)
    if request.method == 'GET':
        if 'training_results' in session:
            session.pop('training_results')
        if 'model_trained' in session:
            session.pop('model_trained')
    
    if request.method == 'POST':
        if 'current_dataset' not in session:
            flash('❌ Please generate a dataset first', 'error')
            return redirect(url_for('generate_dataset'))
        
        try:
            dataset_path = session['current_dataset']
            
            # Print dataset info for debugging
            print(f"🔄 Loading dataset from: {dataset_path}")
            
            # Load dataset with more realistic split
            (X_train, y_train), (X_test, y_test), class_names = image_processor.load_dataset(
                dataset_path, img_size=(64, 64), test_size=0.3  # More test data
            )
            
            # Print dataset info for debugging
            print(f"📊 Training with {len(X_train)} samples, {len(X_test)} test samples")
            print(f"🎯 Classes: {class_names}")
            
            # Build and train model
            model_type = request.form.get('model_type', 'hybrid')
            epochs = int(request.form.get('epochs', 15))
            
            current_model = QiskitHybridCNN(
                input_shape=(64, 64, 3),
                num_classes=len(class_names)
            )
            
            start_time = time.time()
            print(f"🚀 Starting {model_type.upper()} model training...")
            
            history = current_model.train_model(
                X_train, y_train, X_test, y_test, 
                epochs=epochs, 
                model_type=model_type
            )
            training_time = time.time() - start_time
            
            # Evaluate model
            test_loss, test_accuracy = current_model.evaluate(X_test, y_test)
            print(f"📈 Model evaluation - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
            
            # Save model to disk
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"{model_type}_model_{timestamp}"
            model_path = current_model.save_model(model_filename)
            
            # Save to database
            session_data = {
                'session_name': f"{model_type.upper()}_Training_{timestamp}",
                'model_type': model_type,
                'dataset_size': len(X_train) + len(X_test),
                'accuracy': float(test_accuracy),
                'quantum_accuracy': float(test_accuracy) if model_type == 'hybrid' else 0.0,
                'training_time': training_time,
                'epochs': epochs
            }
            
            db.save_training_session(session_data)
            
            # Store results
            current_training_results = {
                'dataset_path': dataset_path,
                'class_names': class_names,
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'model_type': model_type,
                'training_time': training_time,
                'model_path': model_path,
                'timestamp': datetime.now().isoformat(),
                'history': {k: [float(v) for v in vals] for k, vals in history.history.items()}
            }
            
            session['training_results'] = current_training_results
            session['model_trained'] = True
            
            accuracy_msg = f"{test_accuracy:.4f}"
            if model_type == 'hybrid' and test_accuracy > 0.95:
                flash(f'🚀 Quantum Hybrid Model achieved {accuracy_msg} accuracy!', 'success')
            elif model_type == 'classical' and test_accuracy > 0.75:
                flash(f'✅ Classical Model achieved {accuracy_msg} accuracy', 'success')
            else:
                flash(f'📊 {model_type.upper()} Model accuracy: {accuracy_msg}', 'info')
            
        except Exception as e:
            flash(f'❌ Training error: {str(e)}', 'error')
            print(f"Training error details: {traceback.format_exc()}")
    
    return render_template('train_model.html')

@app.route('/classify', methods=['GET', 'POST'])
@login_required
def classify_image():
    global current_model, current_training_results
    
    if request.method == 'POST':
        if current_model is None or current_training_results is None:
            flash('❌ Please train a model first', 'error')
            return redirect(url_for('train_model'))
        
        try:
            if 'image' in request.files:
                image_file = request.files['image']
                if image_file.filename != '':
                    # Save uploaded image
                    filename = f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    image_file.save(filepath)
                    
                    # Preprocess and predict
                    img_array = image_processor.preprocess_image(filepath, target_size=(64, 64))
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    predictions = current_model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    class_names = current_training_results['class_names']
                    
                    result = {
                        'predicted_class': class_names[predicted_class],
                        'confidence': float(confidence),
                        'all_predictions': predictions[0].tolist(),
                        'class_names': class_names,
                        'image_path': filepath
                    }
                    
                    return render_template('results.html', result=result)
        
        except Exception as e:
            flash(f'❌ Classification error: {str(e)}', 'error')
    
    return render_template('classify.html')

@app.route('/quantum_circuit')
@login_required
def quantum_circuit():
    return render_template('quantum_circuit.html')

@app.route('/results')
@login_required
def show_results():
    """Display training results"""
    global current_training_results
    if current_training_results:
        return render_template('results.html', results=current_training_results)
    else:
        flash('❌ No training results available', 'error')
        return redirect(url_for('index'))

@app.route('/clear')
@login_required
def clear_all():
    """Clear everything and start fresh"""
    global current_model, current_training_results
    current_model = None
    current_training_results = None
    session.clear()
    flash('✅ All data cleared. You can start fresh!', 'success')
    return redirect(url_for('index'))

@app.route('/api/dataset_info')
@login_required
def api_dataset_info():
    """API endpoint to get dataset information"""
    if 'current_dataset' in session:
        dataset_info = dataset_manager.get_dataset_info(session['current_dataset'])
        return jsonify(dataset_info)
    return jsonify({'error': 'No dataset available'})

if __name__ == '__main__':
    app.run(debug=True)