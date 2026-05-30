# database_manager.py 

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime

class DatabaseManager:
    def __init__(self):
        self.connection = None
        self.connect()
    
    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host='localhost',
                database='quantum_cnn_db',
                user='root',
                password='',
                port=3306
            )
            if self.connection.is_connected():
                print("✅ Database connected successfully")
                self.initialize_tables()
        except Error as e:
            print(f"⚠️ Database connection error: {e}")
            print("💡 Please ensure MySQL is running in XAMPP")
            self.connection = None
    
    def initialize_tables(self):
        """Initialize all database tables"""
        tables = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(100) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                role ENUM('admin', 'user') DEFAULT 'user',
                status ENUM('pending', 'approved', 'rejected', 'hold') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_name VARCHAR(255),
                model_type VARCHAR(50),
                dataset_size INT,
                accuracy FLOAT,
                quantum_accuracy FLOAT,
                training_time FLOAT,
                epochs INT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_predictions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT,
                image_path VARCHAR(500),
                predicted_class VARCHAR(100),
                confidence FLOAT,
                actual_class VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions(id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS quantum_circuits (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT,
                qubits INT,
                depth INT,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES training_sessions(id)
            )
            """
        ]
        
        for table in tables:
            self.execute_query(table)
        
        # Create default admin user if not exists
        self.execute_query(
            "INSERT IGNORE INTO users (username, email, password, role, status) VALUES (%s, %s, %s, %s, %s)",
            ('admin', 'admin@quantumcnn.com', 'admin123', 'admin', 'approved')
        )
        
        print("✅ Database tables initialized")
    
    def execute_query(self, query, params=None):
        if self.connection is None:
            print("❌ No database connection")
            return None
        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            # For SELECT queries, return results
            if query.strip().lower().startswith('select'):
                result = cursor.fetchall()
            else:
                # For UPDATE, INSERT, DELETE - return number of affected rows
                self.connection.commit()
                result = cursor.rowcount  # Return number of affected rows
            
            cursor.close()
            return result
            
        except Error as e:
            print(f"❌ Query error: {e}")
            print(f"Query: {query}")
            print(f"Params: {params}")
            return None
    
    def save_training_session(self, session_data):
        """Save training session to database"""
        query = """
        INSERT INTO training_sessions 
        (session_name, model_type, dataset_size, accuracy, quantum_accuracy, training_time, epochs)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        params = (
            session_data.get('session_name', f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            session_data.get('model_type'),
            session_data.get('dataset_size'),
            session_data.get('accuracy'),
            session_data.get('quantum_accuracy'),
            session_data.get('training_time'),
            session_data.get('epochs')
        )
        return self.execute_query(query, params)
    
    def save_prediction(self, prediction_data):
        """Save prediction results"""
        query = """
        INSERT INTO model_predictions 
        (session_id, image_path, predicted_class, confidence, actual_class)
        VALUES (%s, %s, %s, %s, %s)
        """
        return self.execute_query(query, prediction_data)
    
    def get_training_sessions(self, limit=10):
        """Get recent training sessions"""
        query = "SELECT * FROM training_sessions ORDER BY created_at DESC LIMIT %s"
        return self.execute_query(query, (limit,))
    
    def get_session_predictions(self, session_id):
        """Get predictions for a specific session"""
        query = "SELECT * FROM model_predictions WHERE session_id = %s ORDER BY created_at DESC"
        return self.execute_query(query, (session_id,))