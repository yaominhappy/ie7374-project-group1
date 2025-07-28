# implementation for Multimodal MELD and Data Augmentation

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, AutoTokenizer
import pickle
import json
from tqdm import tqdm
import librosa
import cv2
import os

# ============= Multimodal Feature Extraction =============

class MultimodalFeatureExtractor:
    """Extract audio and visual features from MELD dataset"""
    
    def __init__(self, data_path='data/MELD.Raw'):
        self.data_path = data_path
        self.audio_path = os.path.join(data_path, 'audio')
        self.video_path = os.path.join(data_path, 'video')
        
    def extract_audio_features(self, audio_file, feature_type='mfcc'):
        """Extract audio features using librosa"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=16000)
            
            if feature_type == 'mfcc':
                # Extract MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                # Concatenate features
                features = np.concatenate([
                    mfcc.mean(axis=1),
                    mfcc_delta.mean(axis=1),
                    mfcc_delta2.mean(axis=1)
                ])
                
            elif feature_type == 'mel_spectrogram':
                # Extract mel-spectrogram
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                features = mel_db.mean(axis=1)
                
            elif feature_type == 'combined':
                # Combine multiple features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
                
                features = np.concatenate([
                    mfcc.mean(axis=1),
                    spectral_centroid.mean(axis=1),
                    spectral_rolloff.mean(axis=1),
                    zero_crossing_rate.mean(axis=1)
                ])
            
            # Pad or truncate to fixed size
            fixed_size = 768  # Match BERT hidden size
            if len(features) < fixed_size:
                features = np.pad(features, (0, fixed_size - len(features)))
            else:
                features = features[:fixed_size]
                
            return features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return np.zeros(768)  # Return zero vector on error
    
    def extract_visual_features(self, video_file, frame_sampling='middle'):
        """Extract visual features from video using OpenCV"""
        try:
            # Open video
            cap = cv2.VideoCapture(video_file)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if frame_sampling == 'middle':
                # Extract middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                frames = [frame] if ret else []
                
            elif frame_sampling == 'uniform':
                # Extract uniformly sampled frames
                frames = []
                sample_indices = np.linspace(0, total_frames-1, 5, dtype=int)
                
                for idx in sample_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            
            cap.release()
            
            # Extract features from frames
            if frames:
                # Simple approach: Use pre-trained CNN features
                # In practice, you'd use a proper vision model
                features = []
                
                for frame in frames:
                    # Resize frame
                    frame = cv2.resize(frame, (224, 224))
                    
                    # Extract basic features (color histograms, etc.)
                    # In practice, use ResNet, VGG, etc.
                    hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
                    hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
                    hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
                    
                    frame_features = np.concatenate([
                        hist_b.flatten(),
                        hist_g.flatten(),
                        hist_r.flatten()
                    ])
                    
                    features.append(frame_features)
                
                # Average features across frames
                features = np.mean(features, axis=0)
                
                # Reduce dimensionality to fixed size
                # In practice, use proper dimensionality reduction
                fixed_size = 2048
                if len(features) > fixed_size:
                    features = features[:fixed_size]
                else:
                    features = np.pad(features, (0, fixed_size - len(features)))
                
                return features
            
            else:
                return np.zeros(2048)
                
        except Exception as e:
            print(f"Error extracting visual features: {e}")
            return np.zeros(2048)
    
    def extract_all_features(self, dialogue_df, save_path='multimodal_features.pkl'):
        """Extract all multimodal features for the dataset"""
        
        audio_features = []
        visual_features = []
        
        print("Extracting multimodal features...")
        
        for idx, row in tqdm(dialogue_df.iterrows(), total=len(dialogue_df)):
            # Get file paths (assuming standard MELD structure)
            dialogue_id = row['Dialogue_ID']
            utterance_id = row['Utterance_ID']
            
            # Construct file paths
            audio_file = os.path.join(self.audio_path, f'dia{dialogue_id}_utt{utterance_id}.wav')
            video_file = os.path.join(self.video_path, f'dia{dialogue_id}_utt{utterance_id}.mp4')
            
            # Extract features
            audio_feat = self.extract_audio_features(audio_file, feature_type='combined')
            visual_feat = self.extract_visual_features(video_file, frame_sampling='uniform')
            
            audio_features.append(audio_feat)
            visual_features.append(visual_feat)
        
        # Save features
        features = {
            'audio': np.array(audio_features),
            'visual': np.array(visual_features)
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(features, f)
        
        print(f"Features saved to {save_path}")
        return features

# ============= Multimodal Dataset =============

class MultimodalMELDDataGenerator(tf.keras.utils.Sequence):
    """Data generator with multimodal features"""
    
    def __init__(self, dialogues, tokenizer, audio_features, visual_features,
                 batch_size=32, max_length=128, context_window=3, shuffle=True):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_window = context_window
        self.shuffle = shuffle
        
        # Prepare data
        self.utterances = []
        self.contexts = []
        self.emotions = []
        self.sentiments = []
        self.indices_map = []  # Map to original indices for features
        self._prepare_data()
        
        self.indices = np.arange(len(self.utterances))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _prepare_data(self):
        """Flatten dialogues and create index mapping"""
        global_idx = 0
        
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
                self.indices_map.append(global_idx)
                global_idx += 1
                
                # Context
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
        batch_audio = []
        batch_visual = []
        
        for i in batch_indices:
            # Text
            if self.contexts[i]:
                text = f"{self.contexts[i]} [SEP] {self.utterances[i]}"
            else:
                text = self.utterances[i]
            
            batch_texts.append(text)
            batch_emotions.append(self.emotions[i])
            batch_sentiments.append(self.sentiments[i])
            
            # Multimodal features
            feature_idx = self.indices_map[i]
            batch_audio.append(self.audio_features[feature_idx])
            batch_visual.append(self.visual_features[feature_idx])
        
        # Tokenize batch
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'audio_features': tf.convert_to_tensor(batch_audio, dtype=tf.float32),
            'visual_features': tf.convert_to_tensor(batch_visual, dtype=tf.float32)
        }, {
            'emotion': tf.convert_to_tensor(batch_emotions),
            'sentiment': tf.convert_to_tensor(batch_sentiments)
        }

# ============= Training Multimodal Model =============

def train_multimodal_model(train_df, dev_df, config=None):
    """Train the multimodal model"""
    
    if config is None:
        config = {
            'batch_size': 8,
            'learning_rate': 1e-5,
            'num_epochs': 10,
            'max_length': 128,
            'context_window': 3,
            'emotion_weight': 1.0,
            'sentiment_weight': 0.5
        }
    
    # Extract or load multimodal features
    feature_extractor = MultimodalFeatureExtractor()
    
    # Check if features already exist
    if os.path.exists('train_multimodal_features.pkl'):
        print("Loading existing features...")
        with open('train_multimodal_features.pkl', 'rb') as f:
            train_features = pickle.load(f)
        with open('dev_multimodal_features.pkl', 'rb') as f:
            dev_features = pickle.load(f)
    else:
        print("Extracting features (this will take a while)...")
        train_features = feature_extractor.extract_all_features(train_df, 'train_multimodal_features.pkl')
        dev_features = feature_extractor.extract_all_features(dev_df, 'dev_multimodal_features.pkl')
    
    # Prepare dialogues
    from prepare_data import prepare_for_transformers
    train_dialogues, _, _ = prepare_for_transformers(train_df)
    dev_dialogues, _, _ = prepare_for_transformers(dev_df)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Load mappings
    with open('data/label_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    # Create data generators
    train_gen = MultimodalMELDDataGenerator(
        train_dialogues, tokenizer,
        train_features['audio'], train_features['visual'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=True
    )
    
    dev_gen = MultimodalMELDDataGenerator(
        dev_dialogues, tokenizer,
        dev_features['audio'], dev_features['visual'],
        batch_size=config['batch_size'],
        max_length=config['max_length'],
        context_window=config['context_window'],
        shuffle=False
    )
    
    # Create model
    model = MultimodalMELDModel(
        text_model_name='bert-base-uncased',
        audio_dim=768,
        visual_dim=2048,
        num_emotions=len(mappings['emotion_to_idx']),
        num_sentiments=len(mappings['sentiment_to_idx'])
    )
    
    # Build model
    dummy_input = next(iter(train_gen))[0]
    _ = model(dummy_input)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])
    
    model.compile(
        optimizer=optimizer,
        loss={
            'emotion': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'sentiment': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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
        tf.keras.callbacks.ModelCheckpoint(
            'best_multimodal_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print("Training multimodal model...")
    history = model.fit(
        train_gen,
        validation_data=dev_gen,
        epochs=config['num_epochs'],
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ============= Data Augmentation Implementation =============

class TextAugmenter:
    """Advanced text augmentation techniques"""
    
    def __init__(self):
        self.augmentation_techniques = [
            'synonym_replacement',
            'random_swap',
            'random_deletion',
            'random_insertion',
            'back_translation',
            'paraphrase'
        ]
        
        # Simple synonym dictionary (in practice, use WordNet or similar)
        self.synonyms = {
            'happy': ['glad', 'joyful', 'pleased', 'delighted'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'melancholy'],
            'angry': ['mad', 'furious', 'irritated', 'annoyed'],
            'good': ['great', 'excellent', 'fine', 'wonderful'],
            'bad': ['terrible', 'awful', 'poor', 'horrible']
        }
    
    def augment(self, text, num_augmented=2, techniques=None):
        """Generate augmented versions of text"""
        if techniques is None:
            techniques = ['synonym_replacement', 'random_swap', 'random_deletion']
        
        augmented_texts = []
        
        for _ in range(num_augmented):
            technique = np.random.choice(techniques)
            
            if technique == 'synonym_replacement':
                augmented = self._synonym_replacement(text)
            elif technique == 'random_swap':
                augmented = self._random_swap(text)
            elif technique == 'random_deletion':
                augmented = self._random_deletion(text)
            elif technique == 'random_insertion':
                augmented = self._random_insertion(text)
            elif technique == 'back_translation':
                augmented = self._back_translation(text)
            elif technique == 'paraphrase':
                augmented = self._paraphrase(text)
            else:
                augmented = text
            
            augmented_texts.append(augmented)
        
        return augmented_texts
    
    def _synonym_replacement(self, text, n=1):
        """Replace n words with synonyms"""
        words = text.split()
        new_words = words.copy()
        
        # Find words that have synonyms
        replaceable_words = []
        for i, word in enumerate(words):
            if word.lower() in self.synonyms:
                replaceable_words.append(i)
        
        # Replace up to n words
        if replaceable_words:
            num_replacements = min(n, len(replaceable_words))
            indices_to_replace = np.random.choice(replaceable_words, num_replacements, replace=False)
            
            for idx in indices_to_replace:
                word = words[idx].lower()
                synonym = np.random.choice(self.synonyms[word])
                # Maintain capitalization
                if words[idx][0].isupper():
                    synonym = synonym.capitalize()
                new_words[idx] = synonym
        
        return ' '.join(new_words)
    
    def _random_swap(self, text, n=1):
        """Randomly swap n pairs of words"""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def _random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p"""
        words = text.split()
        
        # Don't delete if only one word
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if np.random.random() > p:
                new_words.append(word)
        
        # If all words deleted, return original
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)
    
    def _random_insertion(self, text, n=1):
        """Insert n random words"""
        words = text.split()
        new_words = words.copy()
        
        # Simple insertion words
        insertion_words = ['really', 'very', 'quite', 'actually', 'just']
        
        for _ in range(n):
            insertion_word = np.random.choice(insertion_words)
            insertion_position = np.random.randint(0, len(new_words) + 1)
            new_words.insert(insertion_position, insertion_word)
        
        return ' '.join(new_words)
    
    def _back_translation(self, text):
        """Simulate back-translation (in practice, use translation API)"""
        # This is a placeholder - real implementation would use Google Translate API
        # For now, just return a slightly modified version
        return self._synonym_replacement(text, n=2)
    
    def _paraphrase(self, text):
        """Simulate paraphrasing (in practice, use paraphrasing model)"""
        # This is a placeholder - real implementation would use T5 or similar
        # For now, combine multiple techniques
        text = self._synonym_replacement(text)
        text = self._random_swap(text)
        return text

# ============= Augmented Data Generator =============

class AugmentedMELDDataGenerator(tf.keras.utils.Sequence):
    """Data generator with augmentation for minority classes"""
    
    def __init__(self, dialogues, tokenizer, batch_size=32, max_length=128,
                 context_window=3, shuffle=True, augment_minority_classes=True,
                 augmentation_factor=2):
        self.dialogues = dialogues
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_window = context_window
        self.shuffle = shuffle
        self.augment_minority_classes = augment_minority_classes
        self.augmentation_factor = augmentation_factor
        self.augmenter = TextAugmenter()
        
        # Prepare data
        self.utterances = []
        self.contexts = []
        self.emotions = []
        self.sentiments = []
        self._prepare_data()
        
        # Augment if requested
        if self.augment_minority_classes:
            self._augment_data()
        
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
                self.utterances.append(utterances[i])
                self.emotions.append(emotions[i])
                self.sentiments.append(sentiments[i])
                
                # Context
                context = []
                for j in range(max(0, i - self.context_window), i):
                    context.append(f"{speakers[j]}: {utterances[j]}")
                self.contexts.append(" [SEP] ".join(context) if context else "")
    
    def _augment_data(self):
        """Augment minority class samples"""
        print("Augmenting minority classes...")
        
        # Count emotion frequencies
        emotion_counts = {}
        for emotion in self.emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        max_count = max(emotion_counts.values())
        print(f"Class distribution before augmentation: {emotion_counts}")
        
        # Augment minority classes
        augmented_utterances = []
        augmented_contexts = []
        augmented_emotions = []
        augmented_sentiments = []
        
        for emotion_idx, count in emotion_counts.items():
            if count < max_count * 0.5:  # Minority class threshold
                print(f"Augmenting emotion class {emotion_idx} ({count} samples)")
                
                # Find samples with this emotion
                indices = [i for i, e in enumerate(self.emotions) if e == emotion_idx]
                
                # Calculate how many augmentations needed
                target_count = int(max_count * 0.7)  # Don't fully balance
                augmentations_needed = target_count - count
                augmentations_per_sample = max(1, augmentations_needed // len(indices))
                
                # Augment samples
                for idx in indices:
                    aug_texts = self.augmenter.augment(
                        self.utterances[idx],
                        num_augmented=min(augmentations_per_sample, self.augmentation_factor)
                    )
                    
                    for aug_text in aug_texts:
                        augmented_utterances.append(aug_text)
                        augmented_contexts.append(self.contexts[idx])
                        augmented_emotions.append(self.emotions[idx])
                        augmented_sentiments.append(self.sentiments[idx])
        
        # Add augmented data
        print(f"Added {len(augmented_utterances)} augmented samples")
        self.utterances.extend(augmented_utterances)
        self.contexts.extend(augmented_contexts)
        self.emotions.extend(augmented_emotions)
        self.sentiments.extend(augmented_sentiments)
        
        # Print new distribution
        new_emotion_counts = {}
        for emotion in self.emotions:
            new_emotion_counts[emotion] = new_emotion_counts.get(emotion, 0) + 1
        print(f"Class distribution after augmentation: {new_emotion_counts}")
    
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
        
        # Tokenize batch
        encoded = self.tokenizer(
            batch_texts,
            padding=True,
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

# ============= Training with Augmentation =============

def train_with_augmentation(model_type='bert', use_augmentation=True):
    """Train a model with data augmentation"""
    
    # Load data
    train_df = pd.read_csv('data/MELD.Raw/train_sent_emo.csv')
    dev_df = pd.read_csv('data/MELD.Raw/dev_sent_emo.csv')
    
    # Prepare dialogues
    from prepare_data import prepare_for_transformers
    train_dialogues, _, _ = prepare_for_transformers(train_df)
    dev_dialogues, _, _ = prepare_for_transformers(dev_df)
    
    # Initialize tokenizer
    if model_type in ['bert', 'dialoguernn', 'lstm']:
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    
    # Load mappings
    with open('data/label_mappings.json', 'r') as f:
        mappings = json.load(f)
    
    # Create data generators
    if use_augmentation:
        print("Creating augmented data generator...")
        train_gen = AugmentedMELDDataGenerator(
            train_dialogues, tokenizer,
            batch_size=16,
            max_length=128,
            context_window=3,
            shuffle=True,
            augment_minority_classes=True,
            augmentation_factor=3
        )
    else:
        print("Creating standard data generator...")
        from MELDDataGenerator import MELDDataGenerator
        train_gen = MELDDataGenerator(
            train_dialogues, tokenizer,
            batch_size=16,
            max_length=128,
            context_window=3,
            shuffle=True
        )
    
    # Validation generator (no augmentation)
    from MELDDataGenerator import MELDDataGenerator
    dev_gen = MELDDataGenerator(
        dev_dialogues, tokenizer,
        batch_size=16,
        max_length=128,
        context_window=3,
        shuffle=False
    )
    
    # Create model
    from UnifiedEmotionSentimentPredictor import create_model
    model = create_model(
        model_type=model_type,
        num_emotions=len(mappings['emotion_to_idx']),
        num_sentiments=len(mappings['sentiment_to_idx']),
        tokenizer=tokenizer
    )
    
    # Build model
    dummy_input = next(iter(train_gen))[0]
    _ = model(dummy_input)
    
    # Compile
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    
    model.compile(
        optimizer=optimizer,
        loss={
            'emotion': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'sentiment': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        },
        metrics={
            'emotion': ['accuracy'],
            'sentiment': ['accuracy']
        },
        loss_weights={
            'emotion': 1.0,
            'sentiment': 0.5
        }
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f'best_{model_type}_augmented_model.weights.h5' if use_augmentation 
            else f'best_{model_type}_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"Training {model_type} model {'with' if use_augmentation else 'without'} augmentation...")
    history = model.fit(
        train_gen,
        validation_data=dev_gen,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ============= Usage Examples =============

def multimodal_usage_example():
    """Example of using multimodal model"""
    
    print("="*50)
    print("Multimodal Model Usage Example")
    print("="*50)
    
    # Load data
    train_df = pd.read_csv('data/MELD.Raw/train_sent_emo.csv')
    dev_df = pd.read_csv('data/MELD.Raw/dev_sent_emo.csv')
    
    # Train multimodal model
    model, history = train_multimodal_model(train_df, dev_df)
    
    print("Multimodal model training complete!")
    
    # For inference with multimodal features
    class MultimodalPredictor:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.feature_extractor = MultimodalFeatureExtractor()
        
        def predict(self, text, audio_file=None, video_file=None):
            # Tokenize text
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='tf'
            )
            
            # Extract features if files provided
            if audio_file:
                audio_features = self.feature_extractor.extract_audio_features(audio_file)
            else:
                audio_features = np.zeros(768)
            
            if video_file:
                visual_features = self.feature_extractor.extract_visual_features(video_file)
            else:
                visual_features = np.zeros(2048)
            
            # Prepare input
            inputs = {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'audio_features': tf.expand_dims(audio_features, 0),
                'visual_features': tf.expand_dims(visual_features, 0)
            }
            
            # Predict
            predictions = self.model(inputs, training=False)
            
            emotion_probs = tf.nn.softmax(predictions['emotion'])
            sentiment_probs = tf.nn.softmax(predictions['sentiment'])
            
            return {
                'emotion': tf.argmax(emotion_probs, axis=-1).numpy()[0],
                'sentiment': tf.argmax(sentiment_probs, axis=-1).numpy()[0],
                'emotion_probs': emotion_probs.numpy()[0],
                'sentiment_probs': sentiment_probs.numpy()[0]
            }
    
    # Example usage
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    predictor = MultimodalPredictor(model, tokenizer)
    
    result = predictor.predict(
        text="I'm really excited about this!",
        audio_file="path/to/audio.wav",  # Optional
        video_file="path/to/video.mp4"   # Optional
    )
    
    print(f"Prediction: Emotion={result['emotion']}, Sentiment={result['sentiment']}")

def augmentation_usage_example():
    """Example of using data augmentation"""
    
    print("="*50)
    print("Data Augmentation Usage Example")
    print("="*50)
    
    # Example 1: Augment individual texts
    augmenter = TextAugmenter()
    
    original_text = "I am really happy about this amazing news!"
    print(f"Original: {original_text}")
    
    augmented_texts = augmenter.augment(original_text, num_augmented=3)
    print("Augmented versions:")
    for i, text in enumerate(augmented_texts):
        print(f"  {i+1}: {text}")
    
    # Example 2: Train with augmentation
    print("\nTraining BERT with augmentation...")
    model_aug, history_aug = train_with_augmentation(
        model_type='bert',
        use_augmentation=True
    )
    
    # Example 3: Compare with and without augmentation
    print("\nTraining BERT without augmentation...")
    model_no_aug, history_no_aug = train_with_augmentation(
        model_type='bert',
        use_augmentation=False
    )
    
    # Compare results
    print("\nComparison:")
    print(f"With augmentation - Final val acc: {max(history_aug.history['val_emotion_accuracy']):.3f}")
    print(f"Without augmentation - Final val acc: {max(history_no_aug.history['val_emotion_accuracy']):.3f}")

def compare_augmentation_techniques():
    """Compare different augmentation techniques"""
    
    augmenter = TextAugmenter()
    test_sentences = [
        "I am very happy today!",
        "This is terrible and I hate it.",
        "The movie was quite interesting.",
        "I can't believe this happened to me.",
        "Everything is going great!"
    ]
    
    techniques = ['synonym_replacement', 'random_swap', 'random_deletion', 
                  'random_insertion', 'paraphrase']
    
    print("Augmentation Technique Comparison")
    print("="*50)
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        for technique in techniques:
            augmented = augmenter.augment(sentence, num_augmented=1, techniques=[technique])[0]
            print(f"  {technique}: {augmented}")

if __name__ == "__main__":
    # Choose which example to run
    print("Select example to run:")
    print("1. Multimodal Model Usage")
    print("2. Data Augmentation Usage")
    print("3. Compare Augmentation Techniques")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        multimodal_usage_example()
    elif choice == "2":
        augmentation_usage_example()
    elif choice == "3":
        compare_augmentation_techniques()
    else:
        print("Invalid choice")