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

class MELDDataGenerator(tf.keras.utils.Sequence):
    """Fixed data generator with proper max_length padding"""

    def __init__(self, dialogues, tokenizer, batch_size=32, max_length=128,
                 context_window=3, shuffle=True):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_window = context_window
        self.shuffle = shuffle

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
                # Current utterance
                self.utterances.append(utterances[i])
                self.emotions.append(emotions[i])
                self.sentiments.append(sentiments[i])

                # Context (previous utterances)
                context = []
                for j in range(max(0, i - self.context_window), i):
                    context.append(f"{speakers[j]}: {utterances[j]}")
                self.contexts.append(" [SEP] ".join(context) if context else "")

    def __len__(self):
        return int(np.ceil(len(self.utterances) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

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

        # Tokenize batch with FIXED padding to max_length
        encoded = self.tokenizer(
            batch_texts,
            padding='max_length',  # Changed from padding=True
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }, {
            'emotion': tf.convert_to_tensor(batch_emotions),
            'sentiment': tf.convert_to_tensor(batch_sentiments)
        }

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Test function to verify the generator works correctly
def test_data_generator(dialogues, tokenizer, batch_size=16, max_length=128):
    """Test the data generator to ensure it works correctly"""

    print("Testing data generator...")

    # Create generator
    generator = MELDDataGenerator(
        dialogues,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=False
    )

    print(f"Total batches: {len(generator)}")
    print(f"Total samples: {len(generator.utterances)}")

    # Test first batch
    batch_x, batch_y = generator[0]

    print(f"\nFirst batch shapes:")
    print(f"  input_ids: {batch_x['input_ids'].shape}")
    print(f"  attention_mask: {batch_x['attention_mask'].shape}")
    print(f"  emotions: {batch_y['emotion'].shape}")
    print(f"  sentiments: {batch_y['sentiment'].shape}")

    # Verify shapes
    actual_batch_size = batch_x['input_ids'].shape[0]
    assert actual_batch_size <= batch_size, f"Batch size too large! Expected <= {batch_size}, got {actual_batch_size}"
    assert batch_x['input_ids'].shape[1] == max_length, f"Max length mismatch! Expected {max_length}, got {batch_x['input_ids'].shape[1]}"
    assert batch_x['attention_mask'].shape == batch_x['input_ids'].shape, "Attention mask shape mismatch!"
    assert batch_y['emotion'].shape[0] == actual_batch_size, "Emotion batch size mismatch!"
    assert batch_y['sentiment'].shape[0] == actual_batch_size, "Sentiment batch size mismatch!"

    print("\n✓ Data generator test passed!")

    # Print sample data
    print("\nSample from first batch:")
    print(f"  Text (decoded): {tokenizer.decode(batch_x['input_ids'][0], skip_special_tokens=True)[:100]}...")
    print(f"  Emotion label: {batch_y['emotion'][0]}")
    print(f"  Sentiment label: {batch_y['sentiment'][0]}")

    return generator


# LSTM Model Implementation
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

# BERT Model Implementation
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

# DialogueRNN Model Implementation
class DialogueRNN(tf.keras.Model):
    """DialogueRNN for context-aware emotion recognition"""
    def __init__(self, model_name='bert-base-uncased', hidden_dim=128,
                 num_emotions=7, num_sentiments=3, dropout_rate=0.3):
        super(DialogueRNN, self).__init__()

        # Base encoder (BERT or RoBERTa)
        self.encoder = TFAutoModel.from_pretrained(model_name)
        encoder_hidden_size = self.encoder.config.hidden_size

        # Global GRU for context
        self.global_gru = layers.Bidirectional(
            layers.GRU(hidden_dim, return_sequences=True)
        )

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

        # Global context (simplified for single utterance inference)
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

# COSMIC Model Implementation
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
        input_ids = tf.cast(inputs['input_ids'], tf.int32)
        attention_mask = tf.cast(inputs['attention_mask'], tf.int32)
        # Encode utterance
        utterance_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
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

# Create balanced weights for tensor flow dataset
def create_balanced_weights_tf(labels):
    """Create balanced class weights for TensorFlow"""
    # Ensure labels are integers
    labels = np.array(labels, dtype=int)

    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )

    # Create weight tensor
    weight_tensor = tf.constant(class_weights, dtype=tf.float32)

    # Create weight dictionary for class_weight parameter
    weight_dict = {}
    for i, weight in enumerate(class_weights):
        weight_dict[int(unique_labels[i])] = float(weight)

    return weight_tensor, weight_dict

# Focal Loss function for balanced class
class FocalLossBalancedTF(tf.keras.losses.Loss):
    """Fixed Focal Loss for TensorFlow with proper data type handling"""
    def __init__(self, alpha=None, gamma=2.0, from_logits=True):
        super(FocalLossBalancedTF, self).__init__()
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

class EmotionDataAugmenterTF:
    """TensorFlow version of data augmentation"""

    def __init__(self, minority_threshold=0.1):
        self.minority_threshold = minority_threshold

    def identify_minority_classes(self, labels):
        """Identify classes that need augmentation"""
        unique, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        minority_classes = []

        for cls, count in zip(unique, counts):
            if count / total_samples < self.minority_threshold:
                minority_classes.append(cls)

        return minority_classes, dict(zip(unique, counts))

    def simple_augment_text(self, text):
        """Simple text augmentation methods"""
        import random

        augmented = []
        words = text.split()

        # Method 1: Random word deletion (keep at least 2 words)
        if len(words) > 2:
            n_delete = max(1, len(words) // 4)
            indices_to_keep = random.sample(range(len(words)), len(words) - n_delete)
            indices_to_keep.sort()
            aug1 = ' '.join([words[i] for i in indices_to_keep])
            augmented.append(aug1)

        # Method 2: Random word swapping
        if len(words) > 1:
            words_copy = words.copy()
            n_swaps = max(1, len(words) // 4)
            for _ in range(n_swaps):
                if len(words_copy) > 1:
                    idx1, idx2 = random.sample(range(len(words_copy)), 2)
                    words_copy[idx1], words_copy[idx2] = words_copy[idx2], words_copy[idx1]
            augmented.append(' '.join(words_copy))

        return augmented if augmented else [text]

    def create_balanced_dataset(self, dialogues, target_samples_per_class=None):
        """Create balanced dataset through augmentation"""
        # Extract all utterances with labels
        all_utterances = []
        all_emotions = []
        all_sentiments = []

        for dialogue in dialogues:
            for utt, emo, sent in zip(dialogue['utterances'],
                                     dialogue['emotions'],
                                     dialogue['sentiments']):
                all_utterances.append(utt)
                all_emotions.append(emo)
                all_sentiments.append(sent)

        # Identify minority classes
        minority_classes, class_counts = self.identify_minority_classes(all_emotions)

        if target_samples_per_class is None:
            max_count = max(class_counts.values())
            target_samples_per_class = max_count // 3  # Target 1/3 of majority class

        print(f"Original class distribution: {class_counts}")
        print(f"Minority classes to augment: {minority_classes}")

        augmented_dialogues = dialogues.copy()

        for minority_class in minority_classes:
            # Find samples of this class
            minority_indices = [i for i, emo in enumerate(all_emotions)
                              if emo == minority_class]

            current_count = len(minority_indices)
            needed_samples = max(0, target_samples_per_class - current_count)

            print(f"Augmenting class {minority_class}: {current_count} -> {target_samples_per_class}")

            # Generate augmented samples
            import random
            for _ in range(needed_samples):
                # Randomly select existing sample
                source_idx = random.choice(minority_indices)
                source_text = all_utterances[source_idx]

                # Augment text
                augmented_texts = self.simple_augment_text(source_text)

                # Create new dialogue
                new_dialogue = {
                    'dialogue_id': f"aug_{len(augmented_dialogues)}",
                    'utterances': [augmented_texts[0]],
                    'emotions': [all_emotions[source_idx]],
                    'sentiments': [all_sentiments[source_idx]],
                    'speakers': ['Augmented']
                }
                augmented_dialogues.append(new_dialogue)

        return augmented_dialogues

####
# Training main function
###
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import json

def main_with_class_balancing_tf(model_type='bert'):
    """TensorFlow version with class balancing"""

    config = {
        'model_type': model_type,
        'model_name': 'bert-base-uncased', #'distilbert-base-uncased', 
        'batch_size': 16,
        'learning_rate': 1e-5,
        'num_epochs': 15,
        'max_length': 128,
        'context_window': 5,
        'emotion_weight': 2.0,
        'sentiment_weight': 1.0,
        'use_augmentation': True,
        'use_focal_loss': True,
        'early_stopping_patience': 5
    }

    print("Loading data...")
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    # Create label mappings if they don't exist
    try:
        with open('data/label_mappings.json', 'r') as f:
            mappings = json.load(f)
    except:
        print("Creating label mappings...")
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(train_df['Emotion'].unique()))}
        sentiment_to_idx = {sentiment: idx for idx, sentiment in enumerate(sorted(train_df['Sentiment'].unique()))}

        mappings = {
            'emotion_to_idx': emotion_to_idx,
            'sentiment_to_idx': sentiment_to_idx,
            'idx_to_emotion': {v: k for k, v in emotion_to_idx.items()},
            'idx_to_sentiment': {v: k for k, v in sentiment_to_idx.items()}
        }

        # Add label columns
        train_df['emotion_label'] = train_df['Emotion'].map(emotion_to_idx)
        train_df['sentiment_label'] = train_df['Sentiment'].map(sentiment_to_idx)
        dev_df['emotion_label'] = dev_df['Emotion'].map(emotion_to_idx)
        dev_df['sentiment_label'] = dev_df['Sentiment'].map(sentiment_to_idx)
        test_df['emotion_label'] = test_df['Emotion'].map(emotion_to_idx)
        test_df['sentiment_label'] = test_df['Sentiment'].map(sentiment_to_idx)

    # Prepare dialogues
    train_dialogues, _, _ = prepare_for_transformers(train_df)
    dev_dialogues, _, _ = prepare_for_transformers(dev_df)
    test_dialogues, _, _ = prepare_for_transformers(test_df)

    # Apply data augmentation
    if config['use_augmentation']:
        print("Applying data augmentation...")
        augmenter = EmotionDataAugmenterTF()
        train_dialogues = augmenter.create_balanced_dataset(
            train_dialogues, target_samples_per_class=1500
        )
        print(f"Augmented training set size: {len(train_dialogues)}")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    # Create data generators
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

    # Calculate class weights
    all_emotion_labels = []
    all_sentiment_labels = []
    for dialogue in train_dialogues:
        all_emotion_labels.extend(dialogue['emotions'])
        all_sentiment_labels.extend(dialogue['sentiments'])

    emotion_weights_tensor, emotion_weights_dict = create_balanced_weights_tf(all_emotion_labels)
    sentiment_weights_tensor, sentiment_weights_dict = create_balanced_weights_tf(all_sentiment_labels)

    print(f"Emotion class distribution: {np.bincount(all_emotion_labels)}")
    print(f"Emotion weights: {emotion_weights_tensor.numpy()}")

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
    elif config['model_type'] == 'dialoguernn':
        model = DialogueRNN(
            model_name=config['model_name'],
            num_emotions=len(mappings['emotion_to_idx']),
            num_sentiments=len(mappings['sentiment_to_idx'])
        )

    # Build model
    dummy_input = next(iter(train_gen))[0]
    _ = model(dummy_input)

    # Compile with class weights and focal loss
    if config['use_focal_loss']:
        emotion_loss = FocalLossBalancedTF(alpha=emotion_weights_tensor, gamma=2.0)
        sentiment_loss = FocalLossBalancedTF(alpha=sentiment_weights_tensor, gamma=2.0)
    else:
        emotion_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        sentiment_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

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

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            f'best_{model_type}_balanced_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train model
    print("Starting training with class balancing...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['num_epochs'],
        callbacks=callbacks,
        verbose=1
    )

    # Plot results
    plot_training_history(history)

    # Evaluate on test set
    print("\n=== FINAL TEST EVALUATION ===")
    test_results = evaluate_model(model, test_gen, mappings)

    # Additional balanced evaluation
    emotion_preds = []
    sentiment_preds = []
    emotion_labels = []
    sentiment_labels = []

    for batch in test_gen:
        inputs, labels = batch
        predictions = model(inputs, training=False)

        emotion_preds.extend(tf.argmax(predictions['emotion'], axis=-1).numpy())
        sentiment_preds.extend(tf.argmax(predictions['sentiment'], axis=-1).numpy())
        emotion_labels.extend(labels['emotion'].numpy())
        sentiment_labels.extend(labels['sentiment'].numpy())

    # Calculate balanced accuracy
    emotion_balanced_acc = balanced_accuracy_score(emotion_labels, emotion_preds)
    sentiment_balanced_acc = balanced_accuracy_score(sentiment_labels, sentiment_preds)

    print(f"\n=== BALANCED METRICS ===")
    print(f"Emotion Balanced Accuracy: {emotion_balanced_acc:.4f}")
    print(f"Sentiment Balanced Accuracy: {sentiment_balanced_acc:.4f}")

    # Per-class performance for emotions
    from sklearn.metrics import classification_report
    emotion_report = classification_report(
        emotion_labels, emotion_preds,
        target_names=list(mappings['idx_to_emotion'].values()),
        output_dict=True
    )

    print(f"\n=== PER-CLASS EMOTION PERFORMANCE ===")
    for emotion, metrics in emotion_report.items():
        if emotion not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"{emotion}: F1={metrics['f1-score']:.3f}, "
                  f"Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}")

    return model, history, test_results, mappings


#####
# Main function
train_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/train_sent_emo.csv'
dev_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/dev_sent_emo.csv'
test_path = 'https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/test_sent_emo.csv'    

if __name__ == "__main__":
    # Run model
    model_types = ['lstm', 'bert', 'dialoguernn']
    for model_type in model_types:
      model, history, results, mappings = main_with_class_balancing_tf(model_type)
      #plot_confusion_matrices(results, mappings)