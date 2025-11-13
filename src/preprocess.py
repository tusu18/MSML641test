import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import os

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def load_imdb_data():
    """Load IMDb dataset from Keras"""
    print("Loading IMDb dataset...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
    
    # Get word index
    word_index = tf.keras.datasets.imdb.get_word_index()
    reverse_word_index = {v: k for k, v in word_index.items()}
    
    return (x_train, y_train), (x_test, y_test), word_index, reverse_word_index

def preprocess_data(max_words=10000, max_len=50):

    print(f"Preprocessing with max_words={max_words}, max_len={max_len}")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=max_words
    )
    
    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
    x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Positive samples in training: {np.sum(y_train)}/{len(y_train)}")
    
    return x_train, y_train, x_test, y_test

def get_dataset_statistics(x_train, y_train):
    """Get dataset statistics"""
    stats = {
        'num_samples': len(x_train),
        'num_positive': np.sum(y_train),
        'num_negative': len(y_train) - np.sum(y_train),
        'avg_length': np.mean([np.count_nonzero(seq) for seq in x_train]),
        'max_length': np.max([np.count_nonzero(seq) for seq in x_train])
    }
    return stats

if __name__ == "__main__":
    # Test preprocessing
    x_train, y_train, x_test, y_test = preprocess_data(max_words=10000, max_len=50)
    stats = get_dataset_statistics(x_train, y_train)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
