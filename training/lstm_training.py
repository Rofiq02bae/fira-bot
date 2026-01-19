import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from datetime import datetime
import json
import os
from pathlib import Path

# Get absolute paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_FILE = PROJECT_ROOT / "data" / "dataset" / "dataset_training.csv"
MODEL_DIR = PROJECT_ROOT / "data" / "lstm_models"
LOG_DIR = PROJECT_ROOT / "logs"

# Buat folder jika belum ada
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# === MUAT DATASET ===
print("📂 Memuat dataset...")
try:
    df = pd.read_csv(DATASET_FILE)
    print(f"✅ Dataset berhasil dimuat! Shape: {df.shape}")
    print(f"📊 Distribusi intent: {df['intent'].value_counts()}")
except FileNotFoundError:
    print(f"❌ Error: File {DATASET_FILE} tidak ditemukan!")
    exit()

import sys
import os

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from core.processors.text_normalizer import text_normalizer

# === PREPROCESSING DENGAN VALIDASI ===
# Apply global cleaner with lstm mode
patterns = [text_normalizer.global_cleaner(str(p), model_type='lstm') for p in df['pattern'].tolist()] 
# Filter empty
df['cleaned_pattern'] = patterns
df = df[df['cleaned_pattern'] != ""]
patterns = df['cleaned_pattern'].tolist()
intents = df['intent'].tolist()

# Cek distribusi kelas

# Cek distribusi kelas
print("📈 Distribusi kelas:")
print(df['intent'].value_counts())

le = LabelEncoder()
encoded_labels = le.fit_transform(intents)
num_classes = len(le.classes_)

print(f"🎯 Jumlah kelas: {num_classes}")
print(f"🏷️ Daftar intent: {le.classes_}")

tokenizer = Tokenizer(oov_token="<oov>", num_words=5000)  # Batasi vocabulary
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)

# Analisis panjang sequence
seq_lengths = [len(seq) for seq in sequences]
max_len = min(50, int(np.percentile(seq_lengths, 95)))  # Gunakan percentile 95
print(f"📏 Panjang sequence: max={max(seq_lengths)}, avg={np.mean(seq_lengths):.1f}")
print(f"📐 Menggunakan max_len: {max_len}")

padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

print(f"🔢 Vocabulary size: {len(tokenizer.word_index)}")
print("✅ Preprocessing lstm selesai!")

# metode nlp untuk text classification
def bert_setup():
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        bert_model_name = "cahya/bert-base-indonesian-522M"

        print(f"📦 Memuat model BERT: {bert_model_name}...")
        bert_info = {
            "model_name": bert_model_name,
            "purpose": "hybrid nlp dengan lstm",
            "integration_ready": True,
            "version": "1.0.0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return bert_info
    except ImportError as e:
        print("❌ Error: Library transformers tidak ditemukan. Silakan install dengan 'pip install transformers'.")
        return None

# Panggil fungsi bert_setup dan simpan hasilnya
bert_info = bert_setup()

# === SPLIT DATA YANG LEBIH BAIK ===
# Cek apakah semua kelas memiliki minimal 2 sampel untuk stratify
unique, counts = np.unique(encoded_labels, return_counts=True)
min_samples_per_class = np.min(counts)
print(f"📊 Sampel minimum per kelas: {min_samples_per_class}")

# Jika ada kelas dengan hanya 1 sampel, jangan gunakan stratify
if min_samples_per_class < 2:
    print("⚠️  Ada kelas dengan sampel < 2, tidak menggunakan stratify")
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, 
        np.array(encoded_labels), 
        test_size=0.2, 
        random_state=42
    )
else:
    print("✅ Semua kelas memiliki ≥ 2 sampel, menggunakan stratify")
    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, 
        np.array(encoded_labels), 
        test_size=0.2, 
        random_state=42,
        stratify=encoded_labels  # PERBAIKAN: gunakan encoded_labels langsung
    )

print(f"📊 Train shape: {X_train.shape}, Validation shape: {X_val.shape}")

# === BANGUN MODEL DENGAN REGULARISASI ===
vocab_size = min(5000, len(tokenizer.word_index) + 1)  # Batasi vocabulary

model = Sequential([
    Embedding(
        input_dim=vocab_size, 
        output_dim=64, 
        input_length=max_len,
        mask_zero=True
    ),
    LSTM(32, dropout=0.3, recurrent_dropout=0.3),  # Kurangi units, tambah dropout
    Dropout(0.5),
    Dense(16, activation='relu', kernel_regularizer=l2(0.01)),  # Tambah regularisasi L2
    BatchNormalization(),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Gunakan optimizer dengan learning rate lebih rendah
from tensorflow.keras.optimizers import Adam  # PERBAIKAN: import dari tensorflow.keras
optimizer = Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer, 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()

# === CALLBACKS UNTUK MENCEGAH OVERFITTING ===
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.0001,
    verbose=1
)

# === TRAIN MODEL ===
print("\n🚀 Memulai training model dengan regularisasi...")
history = model.fit(
    X_train,
    y_train,
    epochs=100,  # Biarkan early stopping yang menentukan
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

print("✅ Model lstm selesai dilatih!")

# === EVALUASI ===
train_accuracy = model.evaluate(X_train, y_train, verbose=0)
val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print(f"\n📊 Hasil Final:")
print(f"Training - Loss: {train_accuracy[0]:.4f}, Accuracy: {train_accuracy[1]:.4f}")
print(f"Validation - Loss: {val_accuracy[0]:.4f}, Accuracy: {val_accuracy[1]:.4f}")

# === CLASSIFICATION REPORT ===
print("\n📈 Generating Classification Report...")
y_pred = model.predict(X_val, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Generate classification report
report = classification_report(
    y_val, 
    y_pred_classes, 
    labels=range(num_classes),
    target_names=le.classes_,
    digits=4,
    zero_division=0
)

print("\n" + "="*80)
print("📊 CLASSIFICATION REPORT")
print("="*80)
print(report)
print("="*80)

# Simpan classification report ke file
report_file = LOG_DIR / f'classification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("LSTM Model Classification Report\n")
    f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Dataset: {DATASET_FILE}\n")
    f.write(f"Training Accuracy: {train_accuracy[1]:.4f}\n")
    f.write(f"Validation Accuracy: {val_accuracy[1]:.4f}\n")
    f.write("\n" + "="*80 + "\n")
    f.write(report)
    f.write("\n" + "="*80 + "\n")

print(f"💾 Classification report disimpan ke: {report_file}")

# === SIMPAN MODEL ===
model.save(MODEL_DIR / 'chatbot_model.h5')
with open(MODEL_DIR / 'tokenizer.pkl', 'wb') as f:
    pickle.dump({'tokenizer': tokenizer, 'max_len': max_len, 'vocab_size': vocab_size}, f)

with open(MODEL_DIR / 'label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("💾 Model dan tokenizer berhasil disimpan!")

# === SIMPAN INFORMASI HYBRID ===
hybrid_config = {
    "model_type": "lstm",
    "vocab_size": vocab_size,
    "max_sequence_length": max_len,
    "embedding_dim": 64,
    "lstm_units": 32,
    "num_classes": num_classes,
    "tokenizer_info": {
        "oov_token": "<oov>",
        "num_words": 5000
    },
    "training_info": {
        "epochs_trained": len(history.history['accuracy']),
        "final_train_accuracy": float(train_accuracy[1]),
        "final_val_accuracy": float(val_accuracy[1]),
        "final_train_loss": float(train_accuracy[0]),
        "final_val_loss": float(val_accuracy[0])
    },
    "lstm_model": "data/lstm_models/chatbot_model.h5",
    "label_encoder": "data/lstm_models/label_encoder.pkl",
    "tokenizer": "data/lstm_models/tokenizer.pkl",
    "bert_model": bert_info["model_name"] if bert_info else "not_configured",  # PERBAIKAN: gunakan bert_info yang sudah dipanggil
    "training_strategy": "LSTM + IndoBERT Hybrid",
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(MODEL_DIR / 'hybrid_config.json', 'w') as f:
    json.dump(hybrid_config, f, indent=2)

print("🔧 Konfigurasi hybrid system disimpan!")

# === LOGGING ===
log_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "dataset_shape": df.shape,
    "num_classes": num_classes,
    "vocab_size": vocab_size,
    "max_sequence_length": max_len,
    "final_train_accuracy": float(train_accuracy[1]),
    "final_val_accuracy": float(val_accuracy[1]),
    "final_train_loss": float(train_accuracy[0]),
    "final_val_loss": float(val_accuracy[0]),
    "epochs_trained": len(history.history['accuracy']),
    "early_stopping_triggered": len(history.history['accuracy']) < 50,
    "bert_available": bert_info is not None,
    "classification_report_file": str(report_file)
}

with open(LOG_DIR / 'training_log.json', 'a') as f:
    json.dump(log_data, f)
    f.write('\n')

print(f"🧾 Hasil training dicatat ke {LOG_DIR / 'training_log.json'}")

# === INFORMASI NEXT STEPS ===
print("\n" + "="*60)
print("🎯 TRAINING SELESAI - NEXT STEPS:")
print("="*60)
if bert_info:
    print("✅ LSTM + IndoBERT Hybrid System siap!")
    print("   - LSTM Model: ✅ Trained")
    print("   - IndoBERT: ✅ Configured") 
    print("   - Next: Jalankan hybrid_nlu_service.py")
else:
    print("✅ LSTM Model siap! (IndoBERT bisa ditambahkan nanti)")
    print("   - LSTM Model: ✅ Trained")
    print("   - IndoBERT: ❌ Not installed")
    print("   - Install: pip install transformers torch")
print("="*60)