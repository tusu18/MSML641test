import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Dense, Dropout, 
    SimpleRNN, LSTM, Bidirectional,
    GRU
)
from tensorflow.keras.regularizers import l2

def create_rnn_model(
    vocab_size=10000,
    embedding_dim=100,
    hidden_size=64,
    max_length=50,
    architecture='lstm',
    activation='relu',
    dropout_rate=0.4,
    bidirectional=False
):

    model = Sequential(name=f'{architecture}_model')
    
    # Embedding layer
    model.add(Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        input_length=max_length,
        name='embedding'
    ))
    
    model.add(Dropout(dropout_rate))
    
    # Select RNN architecture
    if architecture.lower() == 'rnn':
        RNN_Layer = SimpleRNN
    elif architecture.lower() == 'lstm':
        RNN_Layer = LSTM
    elif architecture.lower() == 'gru':
        RNN_Layer = GRU
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # First RNN layer
    rnn_layer_1 = RNN_Layer(
        hidden_size,
        activation=activation,
        return_sequences=True,
        name=f'{architecture}_layer_1'
    )
    
    if bidirectional:
        model.add(Bidirectional(rnn_layer_1))
    else:
        model.add(rnn_layer_1)
    
    model.add(Dropout(dropout_rate))
    
    # Second RNN layer
    rnn_layer_2 = RNN_Layer(
        hidden_size,
        activation=activation,
        return_sequences=False,
        name=f'{architecture}_layer_2'
    )
    
    if bidirectional:
        model.add(Bidirectional(rnn_layer_2))
    else:
        model.add(rnn_layer_2)
    
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid', name='output'))
    
    return model

def compile_model(model, optimizer='adam', learning_rate=0.001, clipnorm=None):
    """
    Compile the model with specified optimizer and loss
    
    Args:
        model: Keras model
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate
        clipnorm: Gradient clipping norm (None for no clipping)
    """
    # Create optimizer with gradient clipping if specified
    if optimizer.lower() == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    elif optimizer.lower() == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate, clipnorm=clipnorm)
    elif optimizer.lower() == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, clipnorm=clipnorm)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Test model creation
    model = create_rnn_model(architecture='lstm', activation='relu')
    model = compile_model(model, optimizer='adam')
    print(model.summary())
