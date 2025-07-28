"""
BERT model for multi-task emotion and sentiment classification
"""

import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFBertModel, BertModel, TFAutoModel, AutoModel
import torch
import torch.nn as nn


# TensorFlow Implementation
class BERTMultiTaskModelTF(tf.keras.Model):
    """BERT model for multi-task classification (TensorFlow)"""
    
    def __init__(self, model_name='bert-base-uncased', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.3, **kwargs):
        super(BERTMultiTaskModelTF, self).__init__()
        
        # BERT encoder
        self.bert = TFAutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Dropout
        self.dropout = layers.Dropout(dropout_rate)
        
        # Emotion classification head
        self.emotion_dense1 = layers.Dense(hidden_size, activation='relu')
        self.emotion_dropout = layers.Dropout(dropout_rate)
        self.emotion_dense2 = layers.Dense(hidden_size // 2, activation='relu')
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        
        # Sentiment classification head
        self.sentiment_dense1 = layers.Dense(hidden_size, activation='relu')
        self.sentiment_dropout = layers.Dropout(dropout_rate)
        self.sentiment_dense2 = layers.Dense(hidden_size // 2, activation='relu')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')
    
    def call(self, inputs, training=False):
        # Ensure proper dtypes
        input_ids = tf.cast(inputs['input_ids'], tf.int32)
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)
        
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        
        # Use pooled output (CLS token)
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output, training=training)
        
        # Emotion prediction
        emotion_x = self.emotion_dense1(pooled_output)
        emotion_x = self.emotion_dropout(emotion_x, training=training)
        emotion_x = self.emotion_dense2(emotion_x)
        emotion_logits = self.emotion_output(emotion_x)
        
        # Sentiment prediction
        sentiment_x = self.sentiment_dense1(pooled_output)
        sentiment_x = self.sentiment_dropout(sentiment_x, training=training)
        sentiment_x = self.sentiment_dense2(sentiment_x)
        sentiment_logits = self.sentiment_output(sentiment_x)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# PyTorch Implementation
class BERTMultiTaskModelPT(nn.Module):
    """BERT model for multi-task classification (PyTorch)"""
    
    def __init__(self, model_name='bert-base-uncased', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.3, **kwargs):
        super(BERTMultiTaskModelPT, self).__init__()
        
        # BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT layers (optional)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Emotion classification head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_emotions)
        )
        
        # Sentiment classification head
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_sentiments)
        )
    
    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use pooled output
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Task predictions
        emotion_logits = self.emotion_classifier(pooled_output)
        sentiment_logits = self.sentiment_classifier(pooled_output)
        
        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }


# Unified interface
def BERTMultiTaskModel(framework='tensorflow', **kwargs):
    """Create BERT model for specified framework"""
    if framework == 'tensorflow':
        return BERTMultiTaskModelTF(**kwargs)
    else:
        return BERTMultiTaskModelPT(**kwargs)