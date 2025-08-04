"""
DialogueRNN model for emotion and sentiment classification
"""

import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoModel
import torch
import torch.nn as nn


# TensorFlow Implementation
class DialogueRNNTF(tf.keras.Model):
    """DialogueRNN for emotion and sentiment classification (TensorFlow)"""
    
    def __init__(self, base_model_name='bert-base-uncased', hidden_dim=128,
                 num_emotions=7, num_sentiments=3, num_speakers=6, 
                 dropout_rate=0.3, **kwargs):
        super(DialogueRNNTF, self).__init__()
        
        # Base encoder
        self.encoder = TFAutoModel.from_pretrained(base_model_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        
        # Global GRU for context
        self.global_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True)
        )
        
        # Speaker GRUs
        self.speaker_grus = [
            layers.GRU(hidden_dim, return_sequences=True)
            for _ in range(num_speakers)
        ]
        
        # Emotion GRU
        self.emotion_gru = layers.GRU(hidden_dim, return_sequences=False)
        
        # Attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=hidden_dim
        )
        
        # Fusion layer
        self.fusion = layers.Dense(hidden_dim, activation='relu')
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Classification heads
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')
    
    def call(self, inputs, training=False):
        # Ensure proper dtypes
        input_ids = tf.cast(inputs['input_ids'], tf.int32)
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)
        
        # Encode utterance
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        
        # Get CLS token representation
        utterance_features = encoded.pooler_output
        
        # Add sequence dimension for GRU
        utterance_features = tf.expand_dims(utterance_features, axis=1)
        
        # Global context
        global_context = self.global_gru(utterance_features, training=training)
        
        # Apply attention
        attended_features = self.attention(
            utterance_features, global_context,
            training=training
        )
        
        # Emotion modeling
        emotion_features = self.emotion_gru(attended_features, training=training)
        emotion_features = self.dropout(emotion_features, training=training)
        
        # Fusion
        final_features = self.fusion(emotion_features)
        
        # Predictions
        emotion_logits = self.emotion_output(final_features)
        sentiment_logits = self.sentiment_output(final_features)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# PyTorch Implementation
class DialogueRNNPT(nn.Module):
    """DialogueRNN for emotion and sentiment classification (PyTorch)"""
    
    def __init__(self, base_model_name='bert-base-uncased', hidden_dim=128,
                 num_emotions=7, num_sentiments=3, num_speakers=6,
                 dropout_rate=0.3, **kwargs):
        super(DialogueRNNPT, self).__init__()
        
        # Base encoder
        self.encoder = AutoModel.from_pretrained(base_model_name)
        encoder_hidden_size = self.encoder.config.hidden_size
        
        # Global GRU for context
        self.global_gru = nn.GRU(
            encoder_hidden_size, hidden_dim,
            batch_first=True, bidirectional=True
        )
        
        # Speaker GRUs
        self.speaker_grus = nn.ModuleList([
            nn.GRU(encoder_hidden_size, hidden_dim, batch_first=True)
            for _ in range(num_speakers)
        ])
        
        # Emotion GRU
        self.emotion_gru = nn.GRU(
            encoder_hidden_size + hidden_dim * 2, hidden_dim,
            batch_first=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification heads
        self.emotion_head = nn.Linear(hidden_dim, num_emotions)
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiments)
    
    def forward(self, input_ids, attention_mask, speaker_ids=None):
        # Encode utterance
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get CLS token
        utterance_features = outputs.pooler_output
        
        # Add sequence dimension
        utterance_features = utterance_features.unsqueeze(1)
        
        # Global context
        global_out, _ = self.global_gru(utterance_features)
        
        # Apply attention
        attended_features, _ = self.attention(
            utterance_features, global_out, global_out
        )
        
        # Combine features
        combined = torch.cat([utterance_features, attended_features], dim=-1)
        
        # Emotion modeling
        emotion_out, _ = self.emotion_gru(combined)
        emotion_features = emotion_out.squeeze(1)
        emotion_features = self.dropout(emotion_features)
        
        # Fusion
        final_features = self.fusion(emotion_features)
        
        # Predictions
        emotion_logits = self.emotion_head(final_features)
        sentiment_logits = self.sentiment_head(final_features)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# Unified interface
def DialogueRNN(framework='tensorflow', **kwargs):
    """Create DialogueRNN model for specified framework"""
    if framework == 'tensorflow':
        return DialogueRNNTF(**kwargs)
    else:
        return DialogueRNNPT(**kwargs)