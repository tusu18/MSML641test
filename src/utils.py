import os
import time
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'results',
        'results/plots',
        'results/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directories created successfully")

class TimeHistory(Callback):
    """Callback to track time per epoch"""
    def on_train_begin(self, logs=None):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.epoch_time_start)

def print_system_info():
    """Print system information"""
    print("System Information:")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print(f"CPU threads: {os.cpu_count()}")

if __name__ == "__main__":
    create_directories()
    print_system_info()
