"""
LSTM model for emotion and sentiment classification
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import torch
import torch.nn as nn
import torch.nn.functional as F


# TensorFlow Implementation
class LSTMEmotionModelTF(tf.keras.Model):
    """LSTM model for emotion and sentiment classification (TensorFlow)"""
    
    def __init__(self, vocab_size, embedding_dim=300, lstm_units=256,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3, **kwargs):
        super(LSTMEmotionModelTF, self).__init__()
        
        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm1 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate)
        )
        self.lstm2 = layers.Bidirectional(
            layers.LSTM(lstm_units, dropout=dropout_rate)
        )
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Task-specific heads
        self.emotion_dense = layers.Dense(128, activation='relu')
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        
        self.sentiment_dense = layers.Dense(128, activation='relu')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')
    
    def call(self, inputs, training=False):
        # Embedding
        x = self.embedding(inputs['input_ids'])
        
        # LSTM encoding
        x = self.lstm1(x, training=training)
        x = self.lstm2(x, training=training)
        
        # Dropout
        x = self.dropout(x, training=training)
        
        # Task-specific predictions
        emotion_features = self.emotion_dense(x)
        emotion_logits = self.emotion_output(emotion_features)
        
        sentiment_features = self.sentiment_dense(x)
        sentiment_logits = self.sentiment_output(sentiment_features)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# PyTorch Implementation
class LSTMEmotionModelPT(nn.Module):
    """LSTM model for emotion and sentiment classification (PyTorch)"""
    
    def __init__(self, vocab_size, embedding_dim=300, lstm_units=256,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3, **kwargs):
        super(LSTMEmotionModelPT, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm1 = nn.LSTM(
            embedding_dim, lstm_units, 
            num_layers=1, bidirectional=True, 
            batch_first=True, dropout=dropout_rate
        )
        self.lstm2 = nn.LSTM(
            lstm_units * 2, lstm_units,
            num_layers=1, bidirectional=True,
            batch_first=True, dropout=dropout_rate
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Task-specific heads
        self.emotion_dense = nn.Linear(lstm_units * 2, 128)
        self.emotion_output = nn.Linear(128, num_emotions)
        
        self.sentiment_dense = nn.Linear(lstm_units * 2, 128)
        self.sentiment_output = nn.Linear(128, num_sentiments)
        
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask=None):
        # Embedding
        x = self.embedding(input_ids)
        
        # LSTM encoding
        x, _ = self.lstm1(x)
        x, (h_n, _) = self.lstm2(x)
        
        # Get last hidden state
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        hidden = self.dropout(hidden)
        
        # Task-specific predictions
        emotion_features = self.relu(self.emotion_dense(hidden))
        emotion_logits = self.emotion_output(emotion_features)
        
        sentiment_features = self.relu(self.sentiment_dense(hidden))
        sentiment_logits = self.sentiment_output(sentiment_features)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# Unified interface
def LSTMEmotionModel(framework='tensorflow', **kwargs):
    """Create LSTM model for specified framework"""
    if framework == 'tensorflow':
        return LSTMEmotionModelTF(**kwargs)
    else:
        return LSTMEmotionModelPT(**kwargs)