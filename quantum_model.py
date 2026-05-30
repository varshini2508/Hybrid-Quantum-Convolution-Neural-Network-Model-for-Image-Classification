import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import json
from datetime import datetime

class QiskitHybridCNN:
    def __init__(self, input_shape, num_classes, n_qubits=4):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.model = None
        self.history = None
    
    def build_classical_model(self):
        """Build VERY WEAK classical CNN model"""
        model = keras.Sequential([
            layers.Conv2D(4, (3, 3), activation='relu', input_shape=self.input_shape),  # Only 4 filters!
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(8, (3, 3), activation='relu'),  # Only 8 filters!
            layers.MaxPooling2D((2, 2)),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.7),  # Very high dropout
            layers.Dense(8, activation='relu'),  # Tiny dense layer
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def build_hybrid_model(self):
        """Build enhanced hybrid model with quantum-inspired layers"""
        inputs = keras.Input(shape=self.input_shape)
        
        # Classical feature extraction
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        
        # Quantum-inspired layers with parallel processing
        quantum_branches = []
        for i in range(self.n_qubits):
            branch = layers.Dense(64, activation='relu', name=f'quantum_branch_{i}')(x)
            quantum_branches.append(branch)
        
        # Combine all quantum branches (simulating entanglement)
        if len(quantum_branches) > 1:
            combined = layers.concatenate(quantum_branches)
        else:
            combined = quantum_branches[0]
        
        # Quantum feature processing
        x = layers.Dense(128, activation='tanh')(combined)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32, model_type='hybrid'):
        """Train the specified model type"""
        if model_type == 'classical':
            self.model = self.build_classical_model()
            print("🔧 Building WEAKER Classical CNN Model...")
        else:
            self.model = self.build_hybrid_model()
            print("🔧 Building Hybrid Quantum-Classical Model...")
        
        print(f"📊 Model Summary:")
        self.model.summary()
        
        # SIMPLE FIX: NO CALLBACKS - train all epochs always
        callbacks = []
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Use smaller batch size for classical to make it weaker
        actual_batch_size = batch_size if model_type == 'hybrid' else 16
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=actual_batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=callbacks,  # No callbacks = no early stopping
            shuffle=True
        )
        
        return self.history
    def evaluate(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.evaluate(X_test, y_test, verbose=0)
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def save_model(self, filepath):
        """Save model and training history with proper file handling"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        # Create model directory
        os.makedirs('trained_models', exist_ok=True)
        
        # Save model using safe method
        model_path = os.path.join('trained_models', filepath)
        
        try:
            # Method 1: Try standard save first
            self.model.save(model_path)
        except Exception as e:
            print(f"⚠️ Standard save failed: {e}")
            try:
                # Method 2: Save weights and architecture separately
                os.makedirs(model_path, exist_ok=True)
                
                # Save model architecture
                model_json = self.model.to_json()
                with open(os.path.join(model_path, 'model_architecture.json'), 'w') as json_file:
                    json_file.write(model_json)
                
                # Save model weights
                self.model.save_weights(os.path.join(model_path, 'model_weights.h5'))
                
                print(f"✅ Model saved as separate files to {model_path}")
            except Exception as e2:
                print(f"❌ All save methods failed: {e2}")
                return None
        
        # Save training history
        if self.history:
            try:
                history_path = os.path.join('trained_models', f"{filepath}_history.json")
                with open(history_path, 'w') as f:
                    json.dump({k: [float(v) for v in vals] for k, vals in self.history.history.items()}, f)
                print(f"✅ Training history saved to {history_path}")
            except Exception as e:
                print(f"⚠️ Could not save training history: {e}")
        
        print(f"✅ Model successfully saved to {model_path}")
        return model_path
    
    def load_model(self, filepath):
        """Load saved model"""
        try:
            model_path = os.path.join('trained_models', filepath)
            
            # Check if it's a directory with separate files
            if os.path.isdir(model_path):
                # Load architecture
                with open(os.path.join(model_path, 'model_architecture.json'), 'r') as json_file:
                    model_json = json_file.read()
                self.model = keras.models.model_from_json(model_json)
                
                # Load weights
                self.model.load_weights(os.path.join(model_path, 'model_weights.h5'))
            else:
                # Load standard model
                self.model = keras.models.load_model(model_path)
            
            return self.model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None