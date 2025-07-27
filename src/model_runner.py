from data_loader import *
# Model Implementation
# Implementation of LSTM, BERT, RoBERTa with context-aware models using TensorFlow
# Import Required Libraries and Setup
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_hub as hub
import tensorflow_text as text
from transformers import TFAutoModel, AutoTokenizer, TFBertModel, TFRobertaModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# Prepare Data for Transformer Models
# Load datasets
train_df, dev_df, test_df = load_datasets()
# Create a function to prepare data for transformer models
def prepare_for_transformers(df):
    """
    Prepare the dataset for transformer-based models
    """
    # Create emotion to index mapping
    emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(df['Emotion'].unique()))}
    sentiment_to_idx = {sentiment: idx for idx, sentiment in enumerate(sorted(df['Sentiment'].unique()))}

    # Add numerical labels
    df['emotion_label'] = df['Emotion'].map(emotion_to_idx)
    df['sentiment_label'] = df['Sentiment'].map(sentiment_to_idx)

    # Group by dialogue for context modeling
    dialogues = []
    for dialogue_id in df['Dialogue_ID'].unique():
        dialogue = df[df['Dialogue_ID'] == dialogue_id].sort_values('Utterance_ID')
        dialogues.append({
            'dialogue_id': dialogue_id,
            'utterances': dialogue['Utterance'].tolist(),
            'speakers': dialogue['Speaker'].tolist(),
            'emotions': dialogue['emotion_label'].tolist(),
            'sentiments': dialogue['sentiment_label'].tolist()
        })

    return dialogues, emotion_to_idx, sentiment_to_idx

# Prepare training data
train_dialogues, emotion_to_idx, sentiment_to_idx = prepare_for_transformers(train_df)
print(f"Prepared {len(train_dialogues)} dialogues for training")
print(f"\nEmotion mapping: {emotion_to_idx}")
print(f"\nSentiment mapping: {sentiment_to_idx}")

# Save mappings for later use
if not os.path.exists('data'):
    os.makedirs('data')

mappings = {
    'emotion_to_idx': emotion_to_idx,
    'sentiment_to_idx': sentiment_to_idx,
    'idx_to_emotion': {v: k for k, v in emotion_to_idx.items()},
    'idx_to_sentiment': {v: k for k, v in sentiment_to_idx.items()}
}

with open('data/label_mappings.json', 'w') as f:
    json.dump(mappings, f, indent=2)
print("\nLabel mappings saved to 'data/label_mappings.json'")


# Set random seeds
def set_seed(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("GPU is available")
    # Enable mixed precision for better performance
    from tensorflow.keras.mixed_precision import set_global_policy
    set_global_policy('mixed_float16')

# Data Preprocessing and Dataset Classes
class MELDDataGenerator(tf.keras.utils.Sequence):
    """Fixed data generator for MELD dataset with proper tokenization"""
    def __init__(self, dialogues, tokenizer, batch_size=32, max_length=128,
                 context_window=3, shuffle=True):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_window = context_window
        self.shuffle = shuffle

        # IMPORTANT: Set tokenizer max length
        self.tokenizer.model_max_length = max_length

        # Prepare data
        self.utterances = []
        self.contexts = []
        self.emotions = []
        self.sentiments = []
        self._prepare_data()

        self.indices = np.arange(len(self.utterances))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _prepare_data(self):
        """Flatten dialogues into utterances with context"""
        for dialogue in self.dialogues:
            utterances = dialogue['utterances']
            emotions = dialogue['emotions']
            sentiments = dialogue['sentiments']
            speakers = dialogue['speakers']

            for i in range(len(utterances)):
                # Current utterance - ensure it's a string
                self.utterances.append(str(utterances[i]))
                # Ensure emotions and sentiments are integers
                self.emotions.append(int(emotions[i]))
                self.sentiments.append(int(sentiments[i]))

                # Context (previous utterances)
                context = []
                for j in range(max(0, i - self.context_window), i):
                    context.append(f"{speakers[j]}: {utterances[j]}")
                self.contexts.append(" [SEP] ".join(context) if context else "")

    def __len__(self):
        return int(np.ceil(len(self.utterances) / self.batch_size))

    def __getitem__(self, idx):
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.utterances))
        batch_indices = self.indices[start_idx:end_idx]

        batch_texts = []
        batch_emotions = []
        batch_sentiments = []

        for i in batch_indices:
            # Combine context and utterance
            if self.contexts[i]:
                text = f"{self.contexts[i]} [SEP] {self.utterances[i]}"
            else:
                text = self.utterances[i]

            batch_texts.append(text)
            batch_emotions.append(self.emotions[i])
            batch_sentiments.append(self.sentiments[i])

        # Tokenize batch with explicit parameters
        encoded = self.tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='np',
            return_attention_mask=True,
            return_token_type_ids=False  # BERT might return this, we don't need it
        )

        # Ensure we have the right shape
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']

        # Pad manually if needed (shouldn't be necessary but just in case)
        if input_ids.shape[1] < self.max_length:
            pad_length = self.max_length - input_ids.shape[1]
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_length)),
                              constant_values=self.tokenizer.pad_token_id)
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_length)),
                                   constant_values=0)
        elif input_ids.shape[1] > self.max_length:
            input_ids = input_ids[:, :self.max_length]
            attention_mask = attention_mask[:, :self.max_length]

        # Convert to proper numpy arrays with correct dtypes
        input_ids = np.array(input_ids, dtype=np.int32)
        attention_mask = np.array(attention_mask, dtype=np.int32)

        # Return dictionaries with numpy arrays
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }, {
            'emotion': np.array(batch_emotions, dtype=np.int32),
            'sentiment': np.array(batch_sentiments, dtype=np.int32)
        }

    def on_epoch_end(self):
        """Shuffle indices after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

# Test function to verify the generator works correctly
def test_data_generator(dialogues, tokenizer, batch_size=16, max_length=128):
    """Test the data generator to ensure it produces correct shapes"""
    print("Testing MELDDataGenerator...")

    # Create generator
    generator = MELDDataGenerator(
        dialogues[:10],  # Use only first 10 dialogues for testing
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )

    print(f"Total batches: {len(generator)}")
    print(f"Total samples: {len(generator.utterances)}")

    # Test first few batches
    for i in range(min(3, len(generator))):
        batch_x, batch_y = generator[i]
        print(f"\nBatch {i}:")
        print(f"  input_ids: {batch_x['input_ids'].shape}, dtype: {batch_x['input_ids'].dtype}")
        print(f"  attention_mask: {batch_x['attention_mask'].shape}, dtype: {batch_x['attention_mask'].dtype}")
        print(f"  emotion: {batch_y['emotion'].shape}, dtype: {batch_y['emotion'].dtype}")
        print(f"  sentiment: {batch_y['sentiment'].shape}, dtype: {batch_y['sentiment'].dtype}")

        # Check for consistency
        assert batch_x['input_ids'].shape[0] == batch_y['emotion'].shape[0], "Batch size mismatch!"
        assert batch_x['input_ids'].shape[1] == max_length, f"Max length mismatch! Expected {max_length}, got {batch_x['input_ids'].shape[1]}"

    print("\n✓ Data generator test passed!")

def create_tf_dataset(dialogues, tokenizer, batch_size=32, max_length=128,
                     context_window=3, shuffle=True):
    """Create TensorFlow dataset from dialogues"""
    utterances = []
    contexts = []
    emotions = []
    sentiments = []

    # Prepare data
    for dialogue in dialogues:
        utts = dialogue['utterances']
        emos = dialogue['emotions']
        sents = dialogue['sentiments']
        speakers = dialogue['speakers']

        for i in range(len(utts)):
            utterances.append(utts[i])
            emotions.append(emos[i])
            sentiments.append(sents[i])

            # Context
            context = []
            for j in range(max(0, i - context_window), i):
                context.append(f"{speakers[j]}: {utts[j]}")
            contexts.append(" [SEP] ".join(context) if context else "")

    # Create dataset from tensors
    dataset = tf.data.Dataset.from_tensor_slices((utterances, contexts, emotions, sentiments))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    # Map function to combine context and utterance and tokenize
    def map_fn(utterance, context, emotion, sentiment):
        text = tf.cond(tf.equal(tf.strings.length(context), 0),
                       lambda: utterance,
                       lambda: tf.strings.join([context, "[SEP]", utterance], separator=" "))

        # Tokenize
        encoded = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )

        return {
            'input_ids': tf.squeeze(encoded['input_ids'], axis=0),
            'attention_mask': tf.squeeze(encoded['attention_mask'], axis=0)
        }, {
            'emotion': tf.cast(emotion, tf.int32),
            'sentiment': tf.cast(sentiment, tf.int32)
        }

    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch the dataset
    dataset = dataset.batch(batch_size)

    # Set the shapes of the batched tensors
    dataset = dataset.map(
        lambda x, y: (
            {'input_ids': tf.ensure_shape(x['input_ids'], (None, max_length)),
             'attention_mask': tf.ensure_shape(x['attention_mask'], (None, max_length))},
            {'emotion': tf.ensure_shape(y['emotion'], (None,)),
             'sentiment': tf.ensure_shape(y['sentiment'], (None,))}
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )


    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def create_tf_dataset_2(dialogues, tokenizer, batch_size=32, max_length=128,
                     context_window=3, shuffle=True):
    """Create TensorFlow dataset from dialogues"""
    utterances = []
    contexts = []
    emotions = []
    sentiments = []

    # Prepare data
    for dialogue in dialogues:
        utts = dialogue['utterances']
        emos = dialogue['emotions']
        sents = dialogue['sentiments']
        speakers = dialogue['speakers']

        for i in range(len(utts)):
            utterances.append(utts[i])
            emotions.append(emos[i])
            sentiments.append(sents[i])

            # Context
            context = []
            for j in range(max(0, i - context_window), i):
                context.append(f"{speakers[j]}: {utts[j]}")
            contexts.append(" [SEP] ".join(context) if context else "")

    # Create dataset
    def generator():
        indices = np.arange(len(utterances))
        if shuffle:
            np.random.shuffle(indices)

        for i in indices:
            text = f"{contexts[i]} [SEP] {utterances[i]}" if contexts[i] else utterances[i]
            yield text, emotions[i], sentiments[i]

    # Define output signature
    output_signature = (
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=output_signature
    )

    # Tokenize and batch
    def tokenize_batch(texts, emotions, sentiments):
        # This will be called with batched data
        encoded = tokenizer(
            texts.numpy().tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='tf'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'emotion_labels': emotions,
            'sentiment_labels': sentiments
        }

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(
        lambda x, y, z: tf.py_function(
            tokenize_batch,
            [x, y, z],
            {
                'input_ids': tf.int32,
                'attention_mask': tf.int32,
                'emotion_labels': tf.int32,
                'sentiment_labels': tf.int32
            }
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

"""
## Model Architectures
### LSTM-based Model
"""
class LSTMEmotionModel(tf.keras.Model):
    """LSTM model for emotion and sentiment classification"""
    def __init__(self, vocab_size, embedding_dim=300, lstm_units=256,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3):
        super(LSTMEmotionModel, self).__init__()

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

"""### BERT-based Multi-Task Model"""
# Disable mixed precision for inference to avoid dtype issues
tf.keras.mixed_precision.set_global_policy('float32')

class BERTMultiTaskModel(tf.keras.Model):
    """BERT model for multi-task emotion and sentiment classification"""
    def __init__(self, model_name='bert-base-uncased', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.3):
        super(BERTMultiTaskModel, self).__init__()
        
        # BERT encoder
        self.bert = TFAutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Freeze BERT layers initially (optional)
        self.bert.trainable = True
        
        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
        # Emotion classification head
        self.emotion_dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.emotion_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.emotion_dense2 = tf.keras.layers.Dense(hidden_size // 2, activation='relu')
        self.emotion_output = tf.keras.layers.Dense(num_emotions, name='emotion')
        
        # Sentiment classification head
        self.sentiment_dense1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.sentiment_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.sentiment_dense2 = tf.keras.layers.Dense(hidden_size // 2, activation='relu')
        self.sentiment_output = tf.keras.layers.Dense(num_sentiments, name='sentiment')
    
    def call(self, inputs, training=False):
        # Ensure input_ids are int32
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

"""### Context-Aware DialogueRNN Model"""
class DialogueRNN(tf.keras.Model):
    """DialogueRNN for context-aware emotion recognition"""
    def __init__(self, encoder_model, hidden_dim=128, num_emotions=7,
                 num_sentiments=3, num_speakers=6, dropout_rate=0.3):
        super(DialogueRNN, self).__init__()

        self.encoder = encoder_model
        self.hidden_dim = hidden_dim

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
        self.emotion_gru = layers.GRU(hidden_dim, return_sequences=True)

        # Attention mechanism
        self.attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=hidden_dim
        )

        # Dropout
        self.dropout = layers.Dropout(dropout_rate)

        # Classification heads
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')

    def call(self, inputs, speaker_ids=None, training=False):
        # Encode utterances
        encoded = self.encoder(inputs, training=training)

        # Get sequence of hidden states
        if hasattr(encoded, 'last_hidden_state'):
            utterance_features = encoded.last_hidden_state
        else:
            utterance_features = encoded[0]  # For some models

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

        # Get final hidden state for classification
        final_features = emotion_features[:, -1, :]

        # Predictions
        emotion_logits = self.emotion_output(final_features)
        sentiment_logits = self.sentiment_output(final_features)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

"""### COSMIC-style Model with Attention"""
class COSMICModel(tf.keras.Model):
    """COSMIC-style model with commonsense reasoning"""
    def __init__(self, model_name='roberta-base', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.3):
        super(COSMICModel, self).__init__()

        # Main encoder
        self.encoder = TFAutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Commonsense encoder (can be same or different model)
        self.commonsense_encoder = TFAutoModel.from_pretrained('bert-base-uncased')

        # Cross-attention for fusion
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=hidden_size // 8
        )

        # Fusion layers
        self.fusion_dense1 = layers.Dense(hidden_size, activation='relu')
        self.fusion_dropout = layers.Dropout(dropout_rate)
        self.fusion_dense2 = layers.Dense(hidden_size // 2, activation='relu')

        # Self-attention
        self.self_attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=hidden_size // 16
        )

        # Classification heads
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')

        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, commonsense_inputs=None, training=False):
        # Encode utterance
        utterance_outputs = self.encoder(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            training=training
        )
        utterance_features = utterance_outputs.pooler_output

        if commonsense_inputs is not None:
            # Encode commonsense
            commonsense_outputs = self.commonsense_encoder(
                commonsense_inputs['input_ids'],
                attention_mask=commonsense_inputs['attention_mask'],
                training=training
            )
            commonsense_features = commonsense_outputs.pooler_output

            # Cross-attention fusion
            utterance_features = tf.expand_dims(utterance_features, axis=1)
            commonsense_features = tf.expand_dims(commonsense_features, axis=1)

            fused_features = self.cross_attention(
                utterance_features,
                commonsense_features,
                training=training
            )
            fused_features = tf.squeeze(fused_features, axis=1)

            # Further fusion
            fused_features = self.fusion_dense1(fused_features)
            fused_features = self.fusion_dropout(fused_features, training=training)
            final_features = self.fusion_dense2(fused_features)
        else:
            final_features = utterance_features

        final_features = self.dropout(final_features, training=training)

        # Predictions
        emotion_logits = self.emotion_output(final_features)
        sentiment_logits = self.sentiment_output(final_features)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

"""## Loss Functions and Metrics"""
class MultiTaskLoss(tf.keras.losses.Loss):
    """Multi-task loss with task weighting"""
    def __init__(self, emotion_weight=1.0, sentiment_weight=0.5,
                 emotion_class_weights=None, sentiment_class_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.emotion_weight = emotion_weight
        self.sentiment_weight = sentiment_weight
        self.emotion_class_weights = emotion_class_weights
        self.sentiment_class_weights = sentiment_class_weights

    def call(self, y_true, y_pred):
        # Separate emotion and sentiment losses
        emotion_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true['emotion'], y_pred['emotion'], from_logits=True
        )
        sentiment_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true['sentiment'], y_pred['sentiment'], from_logits=True
        )

        # Apply class weights if provided
        if self.emotion_class_weights is not None:
            emotion_weights = tf.gather(self.emotion_class_weights, y_true['emotion'])
            emotion_loss = emotion_loss * emotion_weights

        if self.sentiment_class_weights is not None:
            sentiment_weights = tf.gather(self.sentiment_class_weights, y_true['sentiment'])
            sentiment_loss = sentiment_loss * sentiment_weights

        # Compute weighted sum
        total_loss = (self.emotion_weight * tf.reduce_mean(emotion_loss) +
                     self.sentiment_weight * tf.reduce_mean(sentiment_loss))

        return total_loss

class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss for handling class imbalance"""
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Convert to one-hot
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32),
                                    depth=tf.shape(y_pred)[-1])

        # Compute softmax
        y_pred_softmax = tf.nn.softmax(y_pred)

        # Compute cross entropy
        ce = -y_true_one_hot * tf.math.log(y_pred_softmax + 1e-7)

        # Compute focal term
        focal_term = tf.pow(1.0 - y_pred_softmax, self.gamma)

        # Compute focal loss
        focal_loss = focal_term * ce

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = tf.gather(self.alpha, tf.cast(y_true, tf.int32))
            focal_loss = alpha_t * focal_loss

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

# Custom metrics
class MacroF1Score(tf.keras.metrics.Metric):
    """Macro F1 Score metric"""
    def __init__(self, num_classes, name='macro_f1', **kwargs):
        super(MacroF1Score, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

"""## Training Functions"""
def create_callbacks(model_name, patience=5):
    """Create training callbacks"""
    callbacks = [
        ModelCheckpoint(
            f'best_{model_name}_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience//2,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            histogram_freq=1
        )
    ]
    return callbacks

def train_model(model, train_data, val_data, epochs=10, callbacks=None):
    """Train the model"""
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, test_data, mappings):
    """Evaluate model and generate classification reports"""
    emotion_preds = []
    sentiment_preds = []
    emotion_labels = []
    sentiment_labels = []

    for batch in tqdm(test_data, desc='Evaluating'):
        inputs, labels = batch
        predictions = model(inputs, training=False)

        # Handle predictions (these are tensors)
        if isinstance(predictions['emotion'], tf.Tensor):
            emotion_preds.extend(tf.argmax(predictions['emotion'], axis=-1).numpy())
            sentiment_preds.extend(tf.argmax(predictions['sentiment'], axis=-1).numpy())
        else:
            emotion_preds.extend(np.argmax(predictions['emotion'], axis=-1))
            sentiment_preds.extend(np.argmax(predictions['sentiment'], axis=-1))

        # Handle labels (these might be numpy arrays or tensors)
        if isinstance(labels['emotion'], tf.Tensor):
            emotion_labels.extend(labels['emotion'].numpy())
            sentiment_labels.extend(labels['sentiment'].numpy())
        else:
            emotion_labels.extend(labels['emotion'])
            sentiment_labels.extend(labels['sentiment'])

    # Convert to numpy arrays
    emotion_preds = np.array(emotion_preds)
    sentiment_preds = np.array(sentiment_preds)
    emotion_labels = np.array(emotion_labels)
    sentiment_labels = np.array(sentiment_labels)

    # Generate reports
    emotion_report = classification_report(
        emotion_labels, emotion_preds,
        target_names=list(mappings['idx_to_emotion'].values()),
        output_dict=True
    )

    sentiment_report = classification_report(
        sentiment_labels, sentiment_preds,
        target_names=list(mappings['idx_to_sentiment'].values()),
        output_dict=True
    )

    return {
        'emotion_report': emotion_report,
        'sentiment_report': sentiment_report,
        'emotion_preds': emotion_preds,
        'sentiment_preds': sentiment_preds,
        'emotion_labels': emotion_labels,
        'sentiment_labels': sentiment_labels
    }

"""## Training Pipeline"""
def main(model_type = 'bert'):
    # Configuration
    config = {
        'model_type': model_type,  # 'bert', 'roberta', 'lstm', 'dialoguernn', 'cosmic'
        'model_name': 'bert-base-uncased',
        'batch_size': 16,
        'learning_rate': 2e-5,
        'num_epochs': 10,
        'max_length': 128,
        'context_window': 3,
        'emotion_weight': 1.0,
        'sentiment_weight': 0.5
    }

    # Load data
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Load label mappings
    with open('data/label_mappings.json', 'r') as f:
        mappings = json.load(f)

    # Prepare dialogues
    train_dialogues, _, _ = prepare_for_transformers(train_df)
    dev_dialogues, _, _ = prepare_for_transformers(dev_df)
    test_dialogues, _, _ = prepare_for_transformers(test_df)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Set tokenizer properties
    tokenizer.model_max_length = config['max_length']
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Ensure padding is set correctly
    tokenizer.padding_side = 'right'

    # Create data generators
    print("Creating data generators...")
    train_gen = MELDDataGenerator(
        train_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=True
    )

    val_gen = MELDDataGenerator(
        dev_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=False
    )

    test_gen = MELDDataGenerator(
        test_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=False
    )

    test_data_generator(train_dialogues[:5], tokenizer,
                   batch_size=config['batch_size'],
                   max_length=config['max_length'])

    # Compute class weights
    print("Computing class weights...")
    emotion_labels = train_df['emotion_label'].values
    sentiment_labels = train_df['sentiment_label'].values

    emotion_weights = compute_class_weight(
        'balanced',
        classes=np.unique(emotion_labels),
        y=emotion_labels
    )
    sentiment_weights = compute_class_weight(
        'balanced',
        classes=np.unique(sentiment_labels),
        y=sentiment_labels
    )

    emotion_weights = tf.constant(emotion_weights, dtype=tf.float32)
    sentiment_weights = tf.constant(sentiment_weights, dtype=tf.float32)

    # Initialize model
    print(f"Initializing {config['model_type']} model...")
    if config['model_type'] == 'bert':
        model = BERTMultiTaskModel(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'roberta':
        model = BERTMultiTaskModel(
            model_name='roberta-base',
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'lstm':
        # For LSTM, we need vocab size
        vocab_size = tokenizer.vocab_size
        model = LSTMEmotionModel(
            vocab_size=vocab_size,
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'cosmic':
        model = COSMICModel(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )

    # Build model
    dummy_input = next(iter(train_gen))[0]
    _ = model(dummy_input)

    # Compile model
    loss = MultiTaskLoss(
        emotion_weight=config['emotion_weight'],
        sentiment_weight=config['sentiment_weight'],
        emotion_class_weights=emotion_weights,
        sentiment_class_weights=sentiment_weights
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss={
            'emotion': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'sentiment': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        metrics={
            'emotion': ['accuracy', MacroF1Score(len(mappings['emotion_to_idx']))],
            'sentiment': ['accuracy', MacroF1Score(len(mappings['sentiment_to_idx']))]
        },
        loss_weights={
            'emotion': config['emotion_weight'],
            'sentiment': config['sentiment_weight']
        }
    )

    # Print model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(config['model_type'])

    # Train model
    print("Starting training...")
    history = train_model(
        model,
        train_gen,
        val_gen,
        epochs=config['num_epochs'],
        callbacks=callbacks
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_gen, mappings)

    print("\n=== Emotion Classification Report ===")
    print(classification_report(
        results['emotion_labels'],
        results['emotion_preds'],
        target_names=list(mappings['idx_to_emotion'].values())
    ))

    print("\n=== Sentiment Classification Report ===")
    print(classification_report(
        results['sentiment_labels'],
        results['sentiment_preds'],
        target_names=list(mappings['idx_to_sentiment'].values())
    ))

    # Plot results
    plot_training_history(history)
    plot_confusion_matrices(results, mappings)

    return model, history, results

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Emotion metrics
    axes[0, 0].plot(history.history['emotion_loss'], label='Train')
    axes[0, 0].plot(history.history['val_emotion_loss'], label='Val')
    axes[0, 0].set_title('Emotion Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()

    axes[0, 1].plot(history.history['emotion_accuracy'], label='Train')
    axes[0, 1].plot(history.history['val_emotion_accuracy'], label='Val')
    axes[0, 1].set_title('Emotion Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()

    # Sentiment metrics
    axes[1, 0].plot(history.history['sentiment_loss'], label='Train')
    axes[1, 0].plot(history.history['val_sentiment_loss'], label='Val')
    axes[1, 0].set_title('Sentiment Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].legend()

    axes[1, 1].plot(history.history['sentiment_accuracy'], label='Train')
    axes[1, 1].plot(history.history['val_sentiment_accuracy'], label='Val')
    axes[1, 1].set_title('Sentiment Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_history_tf.png')
    plt.show()

def plot_confusion_matrices(results, mappings):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Emotion confusion matrix
    emotion_cm = confusion_matrix(
        results['emotion_labels'],
        results['emotion_preds']
    )
    sns.heatmap(
        emotion_cm, annot=True, fmt='d',
        xticklabels=list(mappings['idx_to_emotion'].values()),
        yticklabels=list(mappings['idx_to_emotion'].values()),
        ax=axes[0]
    )
    axes[0].set_title('Emotion Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Sentiment confusion matrix
    sentiment_cm = confusion_matrix(
        results['sentiment_labels'],
        results['sentiment_preds']
    )
    sns.heatmap(
        sentiment_cm, annot=True, fmt='d',
        xticklabels=list(mappings['idx_to_sentiment'].values()),
        yticklabels=list(mappings['idx_to_sentiment'].values()),
        ax=axes[1]
    )
    axes[1].set_title('Sentiment Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig('confusion_matrices_tf.png')
    plt.show()

if __name__ == "__main__":
    train_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv'
    dev_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv'
    test_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv'
    
    model_types = ['bert', 'roberta', 'lstm', 'dialoguernn', 'cosmic']
    for model_type in model_types:
        model, history, results = main(model_type)