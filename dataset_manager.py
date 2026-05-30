import os
import numpy as np
from PIL import Image
import random

class DatasetManager:
    def __init__(self):
        self.dataset_path = "qiskit_datasets/"
        os.makedirs(self.dataset_path, exist_ok=True)
    
    def create_synthetic_dataset(self, num_samples=1000, image_size=(64, 64)):
        """Create EXTREMELY CHALLENGING synthetic EO dataset"""
        synthetic_dir = os.path.join(self.dataset_path, "synthetic_eo")
        os.makedirs(synthetic_dir, exist_ok=True)
        
        classes = ['urban', 'vegetation', 'water', 'agriculture', 'barren']
        samples_per_class = num_samples // len(classes)
        
        print(f"🔄 Generating {num_samples} samples ({samples_per_class} per class)...")
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(synthetic_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for i in range(samples_per_class):
                img = self._create_class_pattern(class_name, image_size)
                
                # Convert to PIL Image and save
                img_uint8 = (img * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8)
                pil_img.save(os.path.join(class_dir, f"{class_name}_{i:04d}.jpg"))
            
            print(f"✅ Generated {samples_per_class} {class_name} images")
        
        print(f"🎉 Dataset generation complete! Total: {num_samples} images")
        return synthetic_dir
    
    def _create_class_pattern(self, class_name, size):
        h, w = size
        
        # EXTREME NOISE - MAKE IT VERY HARD TO DISTINGUISH
        base_noise = np.random.normal(0, 0.4, (h, w))  # Very high noise
        
        # ALL CLASSES LOOK VERY SIMILAR - MINIMAL DIFFERENCES
        if class_name == 'urban':
            # Urban - almost random noise
            pattern = np.clip(0.5 + base_noise*0.8, 0, 1)
            # Very subtle color differences
            img = np.stack([pattern*0.65, pattern*0.60, pattern*0.55], axis=-1)
            
        elif class_name == 'vegetation':
            # Vegetation - almost identical to others
            pattern = np.clip(0.5 + base_noise*0.75, 0, 1)
            img = np.stack([pattern*0.55, pattern*0.65, pattern*0.50], axis=-1)
            
        elif class_name == 'water':
            # Water - minimal wave pattern
            base = np.ones((h, w)) * 0.5
            wave_x = np.sin(np.linspace(0, 4*np.pi, w)) * 0.08  # Very weak waves
            wave_pattern = np.outer(np.ones(h), wave_x)
            
            pattern = np.clip(base + wave_pattern*0.2 + base_noise*0.6, 0, 1)
            img = np.stack([pattern*0.45, pattern*0.55, pattern*0.65], axis=-1)
            
        elif class_name == 'agriculture':
            # Agriculture - very weak grid
            grid = np.zeros((h, w))
            grid[::15, :] = 0.2  # Very sparse grid
            grid[:, ::15] = 0.2
            
            pattern = np.clip(0.5 + grid*0.3 + base_noise*0.7, 0, 1)
            img = np.stack([pattern*0.60, pattern*0.62, pattern*0.48], axis=-1)
            
        else:  # barren
            # Barren - almost pure noise
            pattern = np.clip(0.5 + base_noise*0.85, 0, 1)
            img = np.stack([pattern*0.62, pattern*0.58, pattern*0.52], axis=-1)
        
        # ADD EXTREME FINAL NOISE
        final_noise = np.random.normal(0, 0.15, (h, w, 3))
        img = np.clip(img + final_noise, 0, 1)
        
        return img

    def get_dataset_info(self, dataset_path):
        """Get information about the generated dataset"""
        classes = []
        total_images = 0
        
        if os.path.exists(dataset_path):
            for class_name in os.listdir(dataset_path):
                class_dir = os.path.join(dataset_path, class_name)
                if os.path.isdir(class_dir):
                    num_images = len([f for f in os.listdir(class_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    classes.append({
                        'name': class_name,
                        'count': num_images
                    })
                    total_images += num_images
        
        return {
            'total_images': total_images,
            'classes': classes,
            'dataset_path': dataset_path
        }