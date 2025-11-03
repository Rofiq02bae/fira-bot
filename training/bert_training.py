
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from transformers.optimization import AdamW # Import AdamW from optimization module
from tqdm import tqdm
import json
from datetime import datetime
import os
import re

# ==================== TEXT NORMALIZER ====================
class SPBETextNormalizer:
    """Text normalizer khusus untuk istilah SPBE"""

    def __init__(self):
        self.spelling_corrections = {
            # Double letter corrections
            'bapenda': 'bappenda',
            'dinsoss': 'dinsos',
            'disoss': 'dinsos',
            'dukcapill': 'dukcapil',
            'capill': 'capil',

            # Short forms & typos
            'kt': 'ktp',
            'kartu keluarga': 'kk',
            'kartu kel': 'kk',
            'k k': 'kk',

            # Common typos
            'gimana': 'bagaimana',
            'gmn': 'bagaimana',
            'bgmn': 'bagaimana',
            'gmana': 'bagaimana',

            # Time related
            'pukul': 'jam',
            'pkl': 'jam',
            'pukl': 'jam',
        }

        self.regex_patterns = {
            r'bap+enda': 'bappenda',
            r'dinsos+': 'dinsos',
            r'ktp?p?': 'ktp',
            r'k+k': 'kk',
        }

    def normalize(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = text.lower().strip()

        # Simple replacements
        for wrong, correct in self.spelling_corrections.items():
            text = text.replace(wrong, correct)

        # Regex replacements
        for pattern, replacement in self.regex_patterns.items():
            text = re.sub(pattern, replacement, text)

        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

# Global normalizer instance
text_normalizer = SPBETextNormalizer()

# ==================== MAIN TRAINING CODE ====================
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):  # Reduced max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,  # Use reduced length
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)

    return avg_loss, accuracy

def eval_model(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(actual_labels, predictions)

    return avg_loss, accuracy, predictions, actual_labels

def main():
    print("🚀 Memulai Fine-Tuning BERT Optimized...")

    # Load dataset
    print("📂 Memuat dataset...")
    df = pd.read_csv('dataset_training.csv')
    print(f"✅ Dataset loaded: {len(df)} rows")

    # Apply text normalization to patterns
    print("🔄 Applying text normalization...")
    df['pattern_normalized'] = df['pattern'].apply(
        lambda x: text_normalizer.normalize(str(x))
    )

    # Show normalization examples
    print("🔍 Normalization examples:")
    for i in range(min(3, len(df))):
        original = df.iloc[i]['pattern']
        normalized = df.iloc[i]['pattern_normalized']
        if original != normalized:
            print(f"   '{original}' -> '{normalized}'")

    # Filter kelas dengan minimal 2 sampel
    intent_counts = df['intent'].value_counts()
    classes_to_keep = intent_counts[intent_counts >= 2].index
    df_filtered = df[df['intent'].isin(classes_to_keep)].copy()

    print(f"📊 Setelah filtering: {len(df_filtered)} rows")
    print(f"🎯 Kelas: {df_filtered['intent'].nunique()}")

    # Encode labels
    le = LabelEncoder()
    df_filtered['label'] = le.fit_transform(df_filtered['intent'])
    num_classes = len(le.classes_)

    # Split data dengan data yang sudah dinormalisasi
    pattern_array = df_filtered['pattern_normalized'].to_numpy()  # Use normalized patterns!
    label_array = df_filtered['label'].to_numpy()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        pattern_array,
        label_array,
        test_size=0.2,
        random_state=42,
        stratify=label_array
    )

    print(f"📊 Train: {len(train_texts)}, Val: {len(val_texts)}")

    # ==================== MODEL SELECTION ====================
    # UNCOMMENT SALAH SATU MODEL BERDASARKAN KEBUTUHAN:

    # OPTION 1: Fast Training (Recommended for Laptop)
    #model_name = "indobenchmark/indobert-lite-base-p1"  # ~200MB, Training: 1-2 jam
    model_name = "cahya/bert-base-indonesian-522M"
    print("⚡ Using FAST model: indobert-lite-base-p1")

    # OPTION 2: Balanced (Good accuracy + reasonable time)
    # model_name = "indobenchmark/indobert-base-p1"  # ~400MB, Training: 2-3 jam
    # print("⚖️ Using BALANCED model: indobert-base-p1")

    # OPTION 3: Best Accuracy (Heavy - use in Colab)
    # model_name = "cahya/bert-base-indonesian-522M"  # ~500MB, Training: 4-5 jam
    # print("🎯 Using BEST model: bert-base-indonesian-522M")

    print(f"📦 Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes
    )

    model.to(device)

    # Create datasets
    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer, max_length=64)
    val_dataset = ChatbotDataset(val_texts, val_labels, tokenizer, max_length=64)

    # Create data loaders dengan batch size lebih kecil
    batch_size = 16  # Reduced from 8 untuk menghemat memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Scheduler untuk training lebih stabil
    num_training_steps = len(train_loader) * 2  # epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Training loop dengan epochs lebih sedikit
    num_epochs = 3  # Reduced from 3
    best_accuracy = 0

    print(f"\n🔥 Memulai training untuk {num_epochs} epochs...")
    print(f"📊 Config: batch_size={batch_size}, max_length=64")

    training_history = []

    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler)

        # Evaluate
        val_loss, val_acc, val_preds, val_labels = eval_model(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'bert_optimized_best.pth')
            print(f"💾 Saved best model with accuracy: {val_acc:.4f}")

        training_history.append({
            'epoch': epoch + 1,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

    # Save final results
    print("\n💾 Menyimpan hasil akhir...")

    # Save tokenizer dan model
    output_dir = 'bert_optimized_finetuned'
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    # Save label encoder
    import pickle
    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # Save text normalizer
    with open(f'{output_dir}/text_normalizer.pkl', 'wb') as f:
        pickle.dump(text_normalizer, f)

    # Save training info
    info = {
        "model_name": model_name,
        "num_classes": num_classes,
        "classes": le.classes_.tolist(),
        "best_accuracy": best_accuracy,
        "training_history": training_history,
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(df_filtered),
        "text_normalization": True,
        "optimized_config": {
            "batch_size": batch_size,
            "max_length": 64,
            "epochs": num_epochs
        }
    }

    with open(f'{output_dir}/info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Classification report
    print("\n📈 Classification Report:")
    print(classification_report(val_labels, val_preds, target_names=le.classes_))

    print("✅ Fine-tuning selesai!")
    print(f"🎯 Best accuracy: {best_accuracy:.4f}")
    print(f"📁 Model disimpan di: {output_dir}/")
    print(f"⚡ Config: {num_epochs} epochs, batch_size={batch_size}")

if __name__ == "__main__":
    main()