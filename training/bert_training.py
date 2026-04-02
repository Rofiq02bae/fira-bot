import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight  # Tambahan untuk handle imbalance
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import json
from datetime import datetime
import os
import pickle
import sys

# Add parent directory to path for imports
# import text_normalizer
# from text_normalizer import TextNormalizer # This line caused the ModuleNotFoundError
import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Text normalizer dengan improved handling untuk avoid data leakage"""
    
    def __init__(self, aggressive_normalization: bool = False):
        """
        Args:
            aggressive_normalization: Jika True, lakukan normalization lebih aggressive
                                     Jika False, pertahankan beberapa variasi untuk avoid data leakage
        """
        self.aggressive = aggressive_normalization
        
        # Core normalization untuk typo critical
        self.critical_corrections = {
            # Hanya fix yang benar-benar critical
            'bapenda': 'bappenda',
            'dinsoss': 'dinsos',
            'disoss': 'dinsos',
            'kt': 'ktp',
        }
        
        # Optional corrections - hanya jika aggressive mode
        self.optional_corrections = {
            'gimana': 'bagaimana',
            'gmn': 'bagaimana', 
            'bgmn': 'bagaimana',
            'pukul': 'jam',
            'pkl': 'jam',
        }
        
        # Regex patterns dengan caution
        self.regex_patterns = {
            r'bap+enda': 'bappenda',  # bapenda, bappenda -> bappenda
            r'dinsos+': 'dinsos',     # dinsos, dinsoss -> dinsos
        }
        
        # Words to preserve (jangan di-normalize untuk maintain diversity)
        self.preserve_words = {
            'dimana', 'kemana', 'kapan', 'berapa', 'bagaimana',
            'apa', 'kenapa', 'siapa', 'lokasi', 'alamat'
        }
    
    def normalize(self, text: str, preserve_variations: bool = True) -> str:
        """
        Normalize text dengan opsi untuk preserve variations
        
        Args:
            text: Input text
            preserve_variations: Jika True, pertahankan beberapa variasi untuk avoid data leakage
        """
        if not text or not isinstance(text, str):
            return ""
        
        original_text = text
        text = text.lower().strip()
        
        # Step 1: Critical fixes only (typo yang mengubah makna)
        for wrong, correct in self.critical_corrections.items():
            text = text.replace(wrong, correct)
        
        # Step 2: Optional fixes hanya jika aggressive mode
        if self.aggressive:
            for wrong, correct in self.optional_corrections.items():
                text = text.replace(wrong, correct)
        
        # Step 3: Regex patterns dengan word boundary preservation
        for pattern, replacement in self.regex_patterns.items():
            text = re.sub(pattern, replacement, text)
        
        # Step 4: Clean text - tapi preserve some variations
        if preserve_variations:
            # Jangan remove punctuation sepenuhnya, maintain some diversity
            text = re.sub(r'[^\w\s?]', ' ', text)  # Keep question marks
        else:
            text = re.sub(r'[^\w\s]', ' ', text)   # Remove all punctuation
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Log jika normalization membuat perubahan signifikan
        if original_text != text and len(original_text) - len(text) > 5:
            logger.debug(f"Normalization: '{original_text}' -> '{text}'")
        
        return text
    
    def normalize_batch(self, texts: List[str], preserve_variations: bool = True) -> List[str]:
        """Normalize batch of texts"""
        return [self.normalize(text, preserve_variations) for text in texts]
    
    def analyze_normalization_impact(self, patterns: List[str]) -> Dict:
        """Analyze how normalization affects pattern diversity"""
        original_set = set(patterns)
        normalized_set = set(self.normalize_batch(patterns))
        
        return {
            'original_unique': len(original_set),
            'normalized_unique': len(normalized_set),
            'patterns_lost': len(original_set) - len(normalized_set),
            'diversity_preserved': len(normalized_set) / len(original_set) if original_set else 1.0
        }

    def filter_gibberish(self, text: str) -> bool:
        """
        Check if text is likely gibberish (e.g., "asdfghjkl", "123123").
        Returns True if text is VALID, False if GIBBERISH.
        """
        if not text or len(text) < 2:
            return False
            
        # Check repetitive characters (e.g. "aaaaa")
        if re.search(r'(.)\1{4,}', text):
            return False
            
        # Check consonant clusters usually impossible in Indonesian (simplistic)
        if re.search(r'[bcdfghjklmnpqrstvwxyz]{5,}', text):
            return False
            
        # Check if text is just JSON or special chars
        if text.startswith('{') or text.startswith('['):
            return False
            
        return True

    def global_cleaner(self, text: str, model_type: str = 'general') -> str:
        """
        Unified preprocessing function for both BERT and LSTM.
        
        Args:
            text: Input text
            model_type: 'bert' (preserves more context) or 'lstm' (more aggressive normalization)
        """
        if not self.filter_gibberish(text):
            return ""

        text = text.lower().strip()
        
        # 1. Critical Fixes (Always applied)
        for wrong, correct in self.critical_corrections.items():
            text = text.replace(wrong, correct)
            
        # 2. Regex Patterns
        for pattern, replacement in self.regex_patterns.items():
            text = re.sub(pattern, replacement, text)
            
        # 3. Model-specific cleaning
        if model_type == 'lstm':
            # Aggressive: remove punctuation, standardizing
            text = re.sub(r'[^\w\s]', ' ', text)
            for wrong, correct in self.optional_corrections.items():
                text = text.replace(wrong, correct)
        else:
            # BERT: Preserve punctuation useful for context (?, !, etc.)
            text = re.sub(r'[^\w\s?!.,]', ' ', text)
            
        text = re.sub(r'\s+', ' ', text).strip()
        return text

# Global instance
text_normalizer = TextNormalizer(aggressive_normalization=True)


# ==================== MAIN TRAINING CODE ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Using device: {device}")

# Initialize text normalizer for BERT preprocessing
text_normalizer = TextNormalizer(aggressive_normalization=False)

class ChatbotDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
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
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# UPDATE: Menambahkan loss_fn dengan class weights
def train_epoch(model, data_loader, optimizer, device, scheduler=None, loss_fn=None):
    model.train()
    total_loss = 0
    predictions = []
    actual_labels = []

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Gunakan loss_fn custom (dengan weights) jika ada
        if loss_fn:
            loss = loss_fn(outputs.logits, labels)
        else:
            loss = outputs.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    print("🚀 Memulai Fine-Tuning BERT (Balanced Mode)...")

    # Load dataset hasil ekspansi (dataset_training_expanded.csv)
    df = pd.read_csv('dataset_training.csv')

    # Preprocessing menggunakan TextNormalizer dengan mode BERT
    df['cleaned_pattern'] = df['pattern'].apply(lambda x: text_normalizer.global_cleaner(str(x), model_type='bert'))
    df = df[df['cleaned_pattern'] != ""].copy()

    print(f"📊 Total patterns after cleaning: {len(df)}")

    # Identify and filter out intents with only one sample
    intent_counts = df['intent'].value_counts()
    single_sample_intents = intent_counts[intent_counts == 1].index

    if not single_sample_intents.empty:
        print(f"⚠️ Removing {len(single_sample_intents)} intents with only one sample for stratified split: {list(single_sample_intents)}")
        initial_samples = len(df)
        df = df[~df['intent'].isin(single_sample_intents)].copy()
        print(f"   Removed {initial_samples - len(df)} samples.")

    patterns = df['cleaned_pattern'].tolist()
    intents = df['intent'].tolist()

    # Check if there are still enough samples after filtering
    if len(np.unique(intents)) < 2 or len(patterns) < 2:
        raise ValueError("Not enough unique intents or samples after filtering for stratified split. Consider reviewing your dataset or preprocessing steps.")

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(intents)
    num_classes = len(le.classes_)

    # HITUNG CLASS WEIGHTS (Sangat penting untuk mengatasi bias sls_info)
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(weights, dtype=torch.float).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) # Memberi beban lebih besar pada kelas kecil

    # Stratified Split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        patterns, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model_name = "cahya/bert-base-indonesian-522M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    model.to(device)

    train_dataset = ChatbotDataset(train_texts, train_labels, tokenizer)
    val_dataset = ChatbotDataset(val_texts, val_labels, tokenizer)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # EPOCH ditingkatkan ke 15 agar model sempat belajar kelas-kelas kecil
    num_epochs = 15
    num_training_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    best_accuracy = 0
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scheduler, loss_fn=loss_fn)
        val_loss, val_acc, val_preds, val_labels_eval = eval_model(model, val_loader, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'bert_optimized_best.pth')

    # SAVE RESULTS
    output_dir = 'bert_optimized_finetuned'
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with open(f'{output_dir}/label_encoder.pkl', 'wb') as f: pickle.dump(le, f)

    print("\n📈 Final Classification Report:")
    # Gunakan val_labels_eval dari loop terakhir
    print(classification_report(val_labels_eval, val_preds, labels=range(num_classes), target_names=le.classes_))

if __name__ == "__main__":
    main()