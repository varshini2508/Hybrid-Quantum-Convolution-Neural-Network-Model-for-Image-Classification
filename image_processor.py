import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

class ImageProcessor:
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.tiff']
    
    def load_dataset(self, dataset_path, img_size=(64, 64), test_size=0.3):
        """Load and preprocess dataset"""
        images = []
        labels = []
        class_names = []
        
        print(f"🔄 Loading dataset from {dataset_path}...")
        
        for class_idx, class_name in enumerate(sorted(os.listdir(dataset_path))):
            class_dir = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_dir):
                class_names.append(class_name)
                class_images = []
                
                for img_file in os.listdir(class_dir):
                    if any(img_file.lower().endswith(fmt) for fmt in self.supported_formats):
                        img_path = os.path.join(class_dir, img_file)
                        img = self.preprocess_image(img_path, img_size)
                        if img is not None:
                            class_images.append(img)
                
                # Add all images for this class
                images.extend(class_images)
                labels.extend([class_idx] * len(class_images))
                print(f"✅ Loaded {len(class_images)} {class_name} images")
        
        if not images:
            raise ValueError("No images found in the dataset!")
        
        images = np.array(images)
        labels = np.array(labels)
        
        print(f"📊 Dataset loaded: {len(images)} total images, {len(class_names)} classes")
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            images, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        print(f"📈 Train set: {len(X_train)} images, Test set: {len(X_test)} images")
        
        return (X_train, y_train), (X_test, y_test), class_names
    
    def preprocess_image(self, image_path, target_size=(64, 64)):
        """Preprocess single image"""
        try:
            img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(target_size)
            img_array = np.array(img)
            
            # Normalize to [0, 1]
            img_array = img_array.astype(np.float32) / 255.0
            
            return img_array
            
        except Exception as e:
            print(f"⚠️ Error processing image {image_path}: {e}")
            return None
    
    def validate_image(self, image_path):
        """Validate if image can be processed"""
        try:
            img = Image.open(image_path)
            img.verify()
            return True
        except Exception as e:
            print(f"❌ Invalid image: {e}")
            return False