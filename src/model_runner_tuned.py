from data_loader import *
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
import json
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

###
# BERT Model Implementation
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import numpy as np

class EnhancedBERTModel(tf.keras.Model):
    """Enhanced BERT with dialogue understanding"""
    def __init__(self, model_name='bert-base-uncased', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.2):
        super(EnhancedBERTModel, self).__init__()

        # BERT encoder - freeze first few layers for stability
        self.bert = TFAutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Dialogue context processing
        self.context_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size // 4, return_sequences=True, dropout=0.1)
        )

        # Multi-head attention for context integration
        self.context_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=12, key_dim=hidden_size // 12, dropout=0.1
        )

        # Layer normalization
        self.layer_norm1 = tf.keras.layers.LayerNormalization()
        self.layer_norm2 = tf.keras.layers.LayerNormalization()

        # Enhanced feature extraction
        self.feature_dense = tf.keras.layers.Dense(hidden_size, activation='gelu')
        self.feature_dropout = tf.keras.layers.Dropout(dropout_rate)

        # Emotion classification head (deeper for complex emotions)
        self.emotion_dense1 = tf.keras.layers.Dense(hidden_size, activation='gelu')
        self.emotion_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.emotion_dense2 = tf.keras.layers.Dense(hidden_size // 2, activation='gelu')
        self.emotion_dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.emotion_dense3 = tf.keras.layers.Dense(hidden_size // 4, activation='gelu')
        self.emotion_output = tf.keras.layers.Dense(num_emotions, name='emotion')

        # Sentiment classification head (simpler for sentiment)
        self.sentiment_dense1 = tf.keras.layers.Dense(hidden_size // 2, activation='gelu')
        self.sentiment_dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.sentiment_dense2 = tf.keras.layers.Dense(hidden_size // 4, activation='gelu')
        self.sentiment_output = tf.keras.layers.Dense(num_sentiments, name='sentiment')

    def call(self, inputs, training=False):
        input_ids = tf.cast(inputs['input_ids'], tf.int32)
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)

        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )

        # Get outputs
        sequence_output = bert_outputs.last_hidden_state
        pooled_output = bert_outputs.pooler_output

        # Apply context processing
        context_features = self.context_lstm(sequence_output, training=training)
        context_features = self.layer_norm1(context_features, training=training)

        # Apply attention to integrate context
        pooled_expanded = tf.expand_dims(pooled_output, axis=1)
        attended_features = self.context_attention(
            pooled_expanded, context_features, training=training
        )
        final_features = tf.squeeze(attended_features, axis=1)
        final_features = self.layer_norm2(final_features, training=training)

        # Enhanced feature extraction
        enhanced_features = self.feature_dense(final_features)
        enhanced_features = self.feature_dropout(enhanced_features, training=training)

        # Emotion classification (deeper processing)
        emotion_x = self.emotion_dense1(enhanced_features)
        emotion_x = self.emotion_dropout1(emotion_x, training=training)
        emotion_x = self.emotion_dense2(emotion_x)
        emotion_x = self.emotion_dropout2(emotion_x, training=training)
        emotion_x = self.emotion_dense3(emotion_x)
        emotion_logits = self.emotion_output(emotion_x)

        # Sentiment classification
        sentiment_x = self.sentiment_dense1(enhanced_features)
        sentiment_x = self.sentiment_dropout1(sentiment_x, training=training)
        sentiment_x = self.sentiment_dense2(sentiment_x)
        sentiment_logits = self.sentiment_output(sentiment_x)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

####
# LSTM Model Implementation
class EnhancedLSTMEmotionModel(tf.keras.Model):
    """LSTM model for emotion and sentiment classification"""
    def __init__(self, vocab_size, embedding_dim=300, lstm_units=256,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3):
        super(EnhancedLSTMEmotionModel, self).__init__()

        # Embedding layer
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)

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

####
# DialogueRNN model Implementation
class EnhancedDialogueRNN(tf.keras.Model):
    """DialogueRNN for context-aware emotion recognition"""
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3):
        super(EnhancedDialogueRNN, self).__init__()

        # Base encoder (BERT or RoBERTa)
        self.encoder = TFAutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        # Utterance-level GRU
        self.utterance_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True, dropout=0.2)
        )

        # Dialogue-level GRU
        self.dialogue_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=False, dropout=0.2)
        )

        # Self-attention mechanism
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
        utterance_features = encoded.last_hidden_state

        # Utterance-level GRU
        utterance_context = self.utterance_gru(utterance_features, training=training)

        # Apply self-attention
        attended_features = self.attention(
            utterance_context, utterance_context,
            training=training
        )

        # Dialogue-level GRU
        dialogue_context = self.dialogue_gru(attended_features, training=training)

        # Fusion
        final_features = self.fusion(dialogue_context)
        final_features = self.dropout(final_features, training=training)


        # Predictions
        emotion_logits = self.emotion_output(final_features)
        sentiment_logits = self.sentiment_output(final_features)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

####
# COSMIC Model Implementation
class EnhancedCOSMICModel(tf.keras.Model):
    """Enhanced COSMIC-style model for emotion recognition"""
    def __init__(self, model_name='roberta-base', num_emotions=7,
                 num_sentiments=3, dropout_rate=0.3, hidden_dim=256):
        super(EnhancedCOSMICModel, self).__init__()

        # Main encoder
        self.encoder = TFAutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        # Contextual GRU layer to capture sequential information
        self.context_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True, dropout=0.2)
        )

        # Cross-attention to fuse global and contextual features
        self.cross_attention = layers.MultiHeadAttention(
            num_heads=8, key_dim=encoder_hidden_size // 8, dropout=0.1
        )
        self.attention_norm = layers.LayerNormalization()
        self.attention_dropout = layers.Dropout(dropout_rate)


        # Deeper fusion layers
        self.fusion_dense1 = layers.Dense(encoder_hidden_size, activation='gelu')
        self.fusion_dropout = layers.Dropout(dropout_rate)
        self.fusion_dense2 = layers.Dense(hidden_dim, activation='gelu')

        # Final classification heads
        self.emotion_output = layers.Dense(num_emotions, name='emotion')
        self.sentiment_output = layers.Dense(num_sentiments, name='sentiment')

    def call(self, inputs, training=False):
        input_ids = tf.cast(inputs['input_ids'], tf.int32)
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)

        # Encode utterance with main encoder
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            training=training
        )
        sequence_output = encoder_outputs.last_hidden_state # (batch, seq_len, hidden_size)
        pooled_output = encoder_outputs.pooler_output     # (batch, hidden_size)

        # Process sequence with GRU to get contextual features
        contextual_output = self.context_gru(sequence_output, training=training)

        # Fuse global (pooled) and contextual features using cross-attention
        # Query: global info, Key/Value: contextual info
        pooled_output_expanded = tf.expand_dims(pooled_output, axis=1)
        attention_output = self.cross_attention(
            query=pooled_output_expanded,
            value=contextual_output,
            key=contextual_output,
            attention_mask=None, # We can attend to the full context
            training=training
        )

        # Residual connection and normalization
        attention_output = tf.squeeze(attention_output, axis=1)
        fused_features = self.attention_norm(pooled_output + attention_output, training=training)
        fused_features = self.attention_dropout(fused_features, training=training)


        # Deeper fusion
        final_features = self.fusion_dense1(fused_features)
        final_features = self.fusion_dropout(final_features, training=training)
        final_features = self.fusion_dense2(final_features)


        # Classification
        emotion_logits = self.emotion_output(final_features)
        sentiment_logits = self.sentiment_output(final_features)

        return {
            'emotion': emotion_logits,
            'sentiment': sentiment_logits
        }

####
# Data Generator and Augmentor
class AdvancedMELDDataGenerator(tf.keras.utils.Sequence):
    """Fixed advanced data generator"""

    def __init__(self, dialogues, tokenizer, batch_size=16, max_length=256,
                 context_window=7, shuffle=True):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_window = context_window
        self.shuffle = shuffle

        # Prepare data
        self.utterances = []
        self.emotions = []
        self.sentiments = []
        self._prepare_data()

        self.indices = np.arange(len(self.utterances))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _prepare_data(self):
        """Enhanced data preparation"""
        for dialogue in self.dialogues:
            utterances = dialogue['utterances']
            emotions = dialogue['emotions']
            sentiments = dialogue['sentiments']
            speakers = dialogue['speakers']

            for i in range(len(utterances)):
                # Build context with speaker information
                context_parts = []
                for j in range(max(0, i - self.context_window), i):
                    speaker = speakers[j] if j < len(speakers) else "Unknown"
                    context_parts.append(f"{speaker}: {utterances[j]}")

                # Current utterance with speaker
                current_speaker = speakers[i] if i < len(speakers) else "Unknown"
                current_utterance = f"{current_speaker}: {utterances[i]}"

                # Combine context and current utterance
                if context_parts:
                    full_text = " [SEP] ".join(context_parts + [current_utterance])
                else:
                    full_text = current_utterance

                self.utterances.append(full_text)
                self.emotions.append(int(emotions[i]))
                self.sentiments.append(int(sentiments[i]))

    def __len__(self):
        return int(np.ceil(len(self.utterances) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_texts = [self.utterances[i] for i in batch_indices]
        batch_emotions = [self.emotions[i] for i in batch_indices]
        batch_sentiments = [self.sentiments[i] for i in batch_indices]

        # Tokenize with proper padding
        encoded = self.tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }, {
            'emotion': tf.convert_to_tensor(batch_emotions, dtype=tf.int32),
            'sentiment': tf.convert_to_tensor(batch_sentiments, dtype=tf.int32)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

####
# Data Augmentor
class AdvancedEmotionAugmenter:
    """Fixed emotion augmentation with proper logic"""

    def __init__(self):
        # Emotion-specific enhancement patterns
        self.emotion_enhancers = {
            0: ["I'm really angry", "This makes me so mad", "I'm furious"],  # anger
            1: ["This is disgusting", "I hate this", "This is gross"],  # disgust
            2: ["I'm scared", "This worries me", "I'm afraid"],  # fear
            3: ["I'm so happy", "This is great", "I love this"],  # joy
            4: ["I think", "Maybe", "Perhaps"],  # neutral
            5: ["I'm sad", "This hurts", "I feel terrible"],  # sadness
            6: ["Wow", "Amazing", "I can't believe it"]  # surprise
        }

    def augment_text_simple(self, text, emotion_label):
        """Simple but effective text augmentation"""
        if emotion_label not in self.emotion_enhancers:
            return text

        words = text.split()
        if len(words) < 2:
            return text

        # Get emotion-appropriate enhancer
        enhancers = self.emotion_enhancers[emotion_label]
        enhancer = np.random.choice(enhancers)

        # Simple strategies
        strategies = []

        # Strategy 1: Add enhancer at the beginning
        strategies.append(f"{enhancer}. {text}")

        # Strategy 2: Add emotional modifier
        emotional_modifiers = ["really", "very", "quite", "so", "absolutely"]
        if len(words) > 1:
            modifier = np.random.choice(emotional_modifiers)
            new_words = [words[0]] + [modifier] + words[1:]
            strategies.append(" ".join(new_words))

        # Strategy 3: Simple word reordering (for longer sentences)
        if len(words) > 3:
            # Swap first two words
            reordered = [words[1], words[0]] + words[2:]
            strategies.append(" ".join(reordered))

        # Return a random strategy
        return np.random.choice(strategies)

    def create_balanced_dataset(self, dialogues, target_ratio=0.35):
        """Fixed balanced dataset creation"""
        print("Starting data augmentation...")

        # Extract all emotion labels
        all_emotions = []
        all_utterances = []
        all_sentiments = []

        for d_idx, dialogue in enumerate(dialogues):
            for u_idx, (emo, sent, utt) in enumerate(zip(
                dialogue['emotions'],
                dialogue['sentiments'],
                dialogue['utterances']
            )):
                all_emotions.append(int(emo))
                all_sentiments.append(int(sent))
                all_utterances.append(utt)

        # Calculate distribution
        emotion_counts = np.bincount(all_emotions)
        max_count = np.max(emotion_counts)
        target_count = int(max_count * target_ratio)

        print(f"Original emotion distribution: {emotion_counts}")
        print(f"Target count for minority classes: {target_count}")

        augmented_dialogues = dialogues.copy()

        # Augment each emotion class that needs it
        for emotion_class in range(len(emotion_counts)):
            current_count = emotion_counts[emotion_class]

            if current_count < target_count:
                needed = target_count - current_count
                print(f"Augmenting emotion {emotion_class}: +{needed} samples")

                # Find all samples of this emotion
                emotion_samples = []
                for i, emo in enumerate(all_emotions):
                    if emo == emotion_class:
                        emotion_samples.append(i)

                # Generate needed augmented samples
                for _ in range(needed):
                    if emotion_samples:  # Make sure we have samples to augment
                        # FIXED: Randomly select a sample to augment
                        sample_idx = np.random.choice(emotion_samples)

                        # Get original data
                        original_text = all_utterances[sample_idx]
                        original_emotion = all_emotions[sample_idx]
                        original_sentiment = all_sentiments[sample_idx]

                        # Augment the text
                        augmented_text = self.augment_text_simple(original_text, original_emotion)

                        # Create new dialogue entry
                        new_dialogue = {
                            'dialogue_id': f"aug_{len(augmented_dialogues)}",
                            'utterances': [augmented_text],
                            'emotions': [original_emotion],
                            'sentiments': [original_sentiment],
                            'speakers': ['Speaker']
                        }

                        augmented_dialogues.append(new_dialogue)

        print(f"Augmentation complete. New dataset size: {len(augmented_dialogues)}")
        return augmented_dialogues

####
# Focal Loss function
class FocalLossTF(tf.keras.losses.Loss):
    """Fixed Focal Loss for TensorFlow"""
    def __init__(self, alpha=None, gamma=2.0, from_logits=True):
        super(FocalLossTF, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        # Ensure y_true is integer type
        y_true = tf.cast(y_true, tf.int32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Convert to one-hot
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # Calculate cross entropy
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true_one_hot * tf.math.log(y_pred)

        # Calculate focal term
        pt = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
        focal_term = tf.pow(1.0 - pt, self.gamma)

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = tf.gather(self.alpha, y_true)
            focal_loss = alpha_t * focal_term * tf.reduce_sum(ce, axis=-1)
        else:
            focal_loss = focal_term * tf.reduce_sum(ce, axis=-1)

        return tf.reduce_mean(focal_loss)

class LabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Custom label smoothing loss"""
    def __init__(self, smoothing=0.1, from_logits=True):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        num_classes = tf.shape(y_pred)[-1]

        # Convert to one-hot
        y_true_one_hot = tf.one_hot(y_true, depth=num_classes, dtype=tf.float32)

        # Apply label smoothing
        y_true_smooth = y_true_one_hot * (1 - self.smoothing) + \
                       (self.smoothing / tf.cast(num_classes, tf.float32))

        # Calculate loss
        if self.from_logits:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_true_smooth, logits=y_pred
            )
        else:
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
            loss = -tf.reduce_sum(y_true_smooth * tf.math.log(y_pred), axis=-1)

        return tf.reduce_mean(loss)

def create_balanced_weights_tf2(labels):
    """Create balanced class weights for TensorFlow"""
    labels = np.array(labels, dtype=int)
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)

    weight_tensor = tf.constant(class_weights, dtype=tf.float32)
    return weight_tensor


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

####
# Training main function
###
from sklearn.metrics import balanced_accuracy_score
def train_enhanced_model(model_type='bert'):
    """Fixed enhanced training function"""

    # Optimized configuration
    config = {
        'model_type': model_type,
        'model_name': 'bert-base-uncased',
        'batch_size': 8,
        'learning_rate': 2e-5,  # Better learning rate
        'num_epochs': 10,
        'max_length': 256,
        'context_window': 7,
        'emotion_weight': 2.5,
        'sentiment_weight': 1.0,
        'warmup_ratio': 0.1
    }

    print("Loading and preparing data...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Prepare data
    train_dialogues, emotion_to_idx, sentiment_to_idx = prepare_for_transformers(train_df)
    dev_dialogues, _, _ = prepare_for_transformers(dev_df)
    test_dialogues, _, _ = prepare_for_transformers(test_df)

    mappings = {
        'emotion_to_idx': emotion_to_idx,
        'sentiment_to_idx': sentiment_to_idx,
        'idx_to_emotion': {v: k for k, v in emotion_to_idx.items()},
        'idx_to_sentiment': {v: k for k, v in sentiment_to_idx.items()}
    }

    # Data augmentation
    print("Applying advanced data augmentation...")
    augmenter = AdvancedEmotionAugmenter()
    train_dialogues = augmenter.create_balanced_dataset(train_dialogues)
    print(f"Final training set size: {len(train_dialogues)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Create data generators
    train_gen = AdvancedMELDDataGenerator(
        train_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=True
    )

    val_gen = AdvancedMELDDataGenerator(
        dev_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=False
    )

    test_gen = AdvancedMELDDataGenerator(
        test_dialogues, tokenizer,
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=False
    )

    # Calculate class weights
    all_emotion_labels = []
    all_sentiment_labels = []
    for dialogue in train_dialogues:
        all_emotion_labels.extend([int(x) for x in dialogue['emotions']])
        all_sentiment_labels.extend([int(x) for x in dialogue['sentiments']])

    emotion_weights = create_balanced_weights_tf2(all_emotion_labels)
    sentiment_weights = create_balanced_weights_tf2(all_sentiment_labels)

    print(f"Final emotion distribution: {np.bincount(all_emotion_labels)}")

    # Initialize enhanced model
    print(f"Initializing enhanced {config['model_type']} model...")
    if config['model_type'] == 'bert':
        model = EnhancedBERTModel(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'roberta':
        model = EnhancedBERTModel(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'lstm':
        # For LSTM, we need vocab size
        vocab_size = tokenizer.vocab_size
        model = EnhancedLSTMEmotionModel(
            vocab_size=vocab_size,
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'cosmic':
        model = EnhancedCOSMICModel(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )
    elif config['model_type'] == 'dialoguernn':
        model = EnhancedDialogueRNN(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )

    # Build model
    dummy_input = next(iter(train_gen))[0]
    _ = model(dummy_input)

    # Enhanced loss functions (fixed)
    emotion_loss = FocalLossTF(alpha=emotion_weights, gamma=2.0)
    sentiment_loss = LabelSmoothingCrossEntropy(smoothing=0.1)  # Custom implementation

    # Learning rate schedule with warmup
    total_steps = len(train_gen) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])

    def lr_schedule(epoch, lr):
        if epoch < config['num_epochs'] * config['warmup_ratio']:
            return config['learning_rate'] * (epoch + 1) / (config['num_epochs'] * config['warmup_ratio'])
        else:
            return config['learning_rate'] * 0.95 ** (epoch - config['num_epochs'] * config['warmup_ratio'])

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss={
            'emotion': emotion_loss,
            'sentiment': sentiment_loss
        },
        metrics={
            'emotion': ['accuracy'],
            'sentiment': ['accuracy']
        },
        loss_weights={
            'emotion': config['emotion_weight'],
            'sentiment': config['sentiment_weight']
        }
    )

    print("Enhanced model summary:")
    try:
        model.summary()
    except:
        print("Model built successfully")

    # Enhanced callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_emotion_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_{model_type}_balanced_model.weights.h5',
            monitor='val_emotion_accuracy',
            save_best_only=True,
            save_weights_only=True,
            mode='max',
            verbose=1
        )
    ]

    # Train model
    print("Starting enhanced training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['num_epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # Load best weights
    try:
        model.load_weights(f'best_{model_type}_balanced_model.weights.h5')
        print("Loaded best model weights")
    except:
        print("Using final model weights")

    # Comprehensive evaluation
    print("\n" + "="*50)
    print("ENHANCED MODEL EVALUATION")
    print("="*50)

    # Get predictions
    emotion_preds = []
    sentiment_preds = []
    emotion_labels = []
    sentiment_labels = []

    print("Generating predictions...")
    for i, batch in enumerate(test_gen):
        inputs, labels = batch
        predictions = model(inputs, training=False)

        emotion_preds.extend(tf.argmax(predictions['emotion'], axis=-1).numpy())
        sentiment_preds.extend(tf.argmax(predictions['sentiment'], axis=-1).numpy())
        emotion_labels.extend(labels['emotion'].numpy())
        sentiment_labels.extend(labels['sentiment'].numpy())

        if i % 10 == 0:
            print(f"Processed {i+1}/{len(test_gen)} batches")

    # Calculate enhanced metrics
    emotion_balanced_acc = balanced_accuracy_score(emotion_labels, emotion_preds)
    sentiment_balanced_acc = balanced_accuracy_score(sentiment_labels, sentiment_preds)

    print(f"\n📊 FINAL PERFORMANCE METRICS")
    print(f"Emotion Balanced Accuracy: {emotion_balanced_acc:.4f}")
    print(f"Sentiment Balanced Accuracy: {sentiment_balanced_acc:.4f}")

    # Detailed emotion analysis
    from sklearn.metrics import classification_report
    emotion_report = classification_report(
        emotion_labels, emotion_preds,
        target_names=list(mappings['idx_to_emotion'].values()),
        output_dict=True,
        zero_division=0
    )

    print(f"\n🎭 DETAILED EMOTION PERFORMANCE")
    for emotion, metrics in emotion_report.items():
        if emotion not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{emotion:>10}: F1={metrics['f1-score']:.3f} | "
                  f"Precision={metrics['precision']:.3f} | "
                  f"Recall={metrics['recall']:.3f} | "
                  f"Support={int(metrics['support'])}")

    print(f"\n📈 SUMMARY METRICS")
    print(f"Emotion Macro F1: {emotion_report['macro avg']['f1-score']:.4f}")
    print(f"Emotion Weighted F1: {emotion_report['weighted avg']['f1-score']:.4f}")

    # Count non-zero performance classes
    non_zero_emotions = sum(1 for emotion, metrics in emotion_report.items()
                           if emotion not in ['accuracy', 'macro avg', 'weighted avg']
                           and metrics['f1-score'] > 0.0)

    print(f"Emotions with non-zero F1: {non_zero_emotions}/7")

    # Plot results
    plot_training_history(history)

    return model, history, emotion_report, mappings

#####
# Main function
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score

train_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv'
dev_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv'
test_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv'
if __name__ == "__main__":
    print("🚀 Starting Enhanced Model Training...")
    try:
        model_types = ['bert', 'dialoguernn', 'lstm'] #['bert', 'lstm', 'dialoguernn', 'cosmic']
        for model_type in model_types:
            model, history, results, mappings = train_enhanced_model(model_type)
        print("✅ Enhanced training completed successfully!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()