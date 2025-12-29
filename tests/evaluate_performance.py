import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import logging
import sys
import os

# Tambahkan project root ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import get_hybrid_nlu

# Setup logging minimal agar tidak mengganggu output evaluasi
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def evaluate_chatbot(test_data_path: str):
    """
    Menjalankan evaluasi performa chatbot hybrid.
    """
    if not os.path.exists(test_data_path):
        print(f"❌ Error: File {test_data_path} tidak ditemukan.")
        return

    print(f"📂 Loading data uji dari: {test_data_path}")
    df = pd.read_csv(test_data_path)
    
    # Pastikan kolom yang dibutuhkan ada
    if 'response' not in df.columns or 'intent' not in df.columns:
        print("❌ Error: Dataset harus memiliki kolom 'response' dan 'intent'.")
        return

    # Inisialisasi NLU Service
    nlu = get_hybrid_nlu()
    
    y_true = []
    y_pred = []
    
    print(f"🧪 Mengevaluasi {len(df)} baris data...")
    
    # Iterasi data
    for index, row in df.iterrows():
        text = str(row['response'])
        true_intent = str(row['intent'])
        
        # Prediksi menggunakan hybrid flow
        result = nlu.process_query(text)
        pred_intent = result['intent']
        
        y_true.append(true_intent)
        y_pred.append(pred_intent)
        
        if (index + 1) % 50 == 0:
            print(f"   Processed {index + 1}/{len(df)}...")

    # 1. Classification Report
    print("\n" + "="*60)
    print("📊 CLASSIFICATION REPORT")
    print("="*60)
    report = classification_report(y_true, y_pred)
    print(report)
    
    # 2. Confusion Matrix
    print("\n🎨 Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Hybrid Chatbot Prediction')
    plt.ylabel('Actual Intent')
    plt.xlabel('Predicted Intent')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save visualization
    plt.savefig('evaluation_results.png')
    print("✅ Visualisasi disimpan sebagai 'evaluation_results.png'")
    
    plt.show()

if __name__ == "__main__":
    # Silahkan ganti dataset_path ke file test Anda
    # Jika belum ada file test, kita gunakan subset dari training untuk demo
    dataset_path = "data/dataset/dataset_training.csv" 
    
    try:
        evaluate_chatbot(dataset_path)
    except Exception as e:
        print(f"❌ Terjadi kesalahan: {e}")
        import traceback
        traceback.print_exc()
