import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from config.settings import ModelConfig
from core.nlu_service import HybridNLUService
from tqdm import tqdm

# Paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_FILE = PROJECT_ROOT / "data" / "dataset" / "dataset_training.csv"
LOG_DIR = PROJECT_ROOT / "logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_hybrid_model():
    """
    Evaluasi model hybrid (LSTM + BERT) dan generate classification report
    """
    print("="*80)
    print("🔬 HYBRID MODEL EVALUATION - Classification Report")
    print("="*80)
    
    # 1. Initialize service
    print("\n📦 Initializing Hybrid NLU Service...")
    config = ModelConfig()
    service = HybridNLUService(config)
    
    # 2. Load dataset
    print(f"\n📂 Loading dataset from: {DATASET_FILE}")
    df = pd.read_csv(DATASET_FILE)
    print(f"✅ Dataset loaded: {len(df)} samples")
    
    # 3. Prepare data
    patterns = df['pattern'].tolist()
    true_intents = df['intent'].tolist()
    
    # 4. Get predictions dari hybrid system
    print("\n🚀 Running predictions on all samples...")
    predicted_intents = []
    confidences = []
    methods_used = []
    
    for pattern in tqdm(patterns, desc="Predicting"):
        try:
            result = service.process_query(str(pattern))
            predicted_intents.append(result.get('intent', 'unknown'))
            confidences.append(result.get('confidence', 0.0))
            methods_used.append(result.get('method', 'unknown'))
        except Exception as e:
            print(f"Error processing '{pattern}': {e}")
            predicted_intents.append('error')
            confidences.append(0.0)
            methods_used.append('error')
    
    # 5. Encode labels untuk classification report
    le = LabelEncoder()
    all_intents = list(set(true_intents + predicted_intents))
    le.fit(all_intents)
    
    y_true = le.transform(true_intents)
    y_pred = le.transform(predicted_intents)
    
    # 6. Generate classification report
    print("\n📊 Generating Classification Report...")
    report = classification_report(
        y_true,
        y_pred,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        digits=4,
        zero_division=0
    )
    
    # 7. Calculate overall metrics
    accuracy = np.mean(np.array(true_intents) == np.array(predicted_intents))
    avg_confidence = np.mean(confidences)
    
    # Count methods used
    method_counts = pd.Series(methods_used).value_counts().to_dict()
    
    # 8. Display results
    print("\n" + "="*80)
    print("📊 HYBRID MODEL CLASSIFICATION REPORT")
    print("="*80)
    print(f"\n📈 Overall Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Average Confidence: {avg_confidence:.4f}")
    print(f"   Total Samples: {len(patterns)}")
    print(f"\n🔧 Methods Distribution:")
    for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(methods_used)) * 100
        print(f"   {method}: {count} ({percentage:.2f}%)")
    
    print("\n" + "="*80)
    print("📋 Detailed Classification Report by Intent:")
    print("="*80)
    print(report)
    print("="*80)
    
    # 9. Save report to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = LOG_DIR / f'hybrid_classification_report_{timestamp}.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("HYBRID MODEL (LSTM + BERT) CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {DATASET_FILE}\n")
        f.write(f"Total Samples: {len(patterns)}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n")
        f.write(f"Average Confidence: {avg_confidence:.4f}\n\n")
        
        f.write("Methods Distribution:\n")
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(methods_used)) * 100
            f.write(f"  {method}: {count} ({percentage:.2f}%)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Detailed Classification Report:\n")
        f.write("="*80 + "\n\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")
    
    print(f"\n💾 Classification report saved to: {report_file}")
    
    # 10. Save detailed results to CSV
    results_df = pd.DataFrame({
        'pattern': patterns,
        'true_intent': true_intents,
        'predicted_intent': predicted_intents,
        'confidence': confidences,
        'method': methods_used,
        'correct': np.array(true_intents) == np.array(predicted_intents)
    })
    
    csv_file = LOG_DIR / f'hybrid_evaluation_results_{timestamp}.csv'
    results_df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"💾 Detailed results saved to: {csv_file}")
    
    # 11. Generate confusion matrix summary
    print("\n📊 Generating confusion matrix summary...")
    conf_matrix = confusion_matrix(y_true, y_pred, labels=range(len(le.classes_)))
    
    # Find most confused pairs
    confused_pairs = []
    for i in range(len(le.classes_)):
        for j in range(len(le.classes_)):
            if i != j and conf_matrix[i][j] > 0:
                confused_pairs.append({
                    'true_intent': le.classes_[i],
                    'predicted_as': le.classes_[j],
                    'count': int(conf_matrix[i][j])
                })
    
    confused_pairs = sorted(confused_pairs, key=lambda x: x['count'], reverse=True)[:10]
    
    print("\n🔴 Top 10 Most Confused Intent Pairs:")
    for idx, pair in enumerate(confused_pairs, 1):
        print(f"   {idx}. {pair['true_intent']} → {pair['predicted_as']}: {pair['count']} times")
    
    # Save confusion pairs
    confusion_file = LOG_DIR / f'hybrid_confusion_analysis_{timestamp}.json'
    with open(confusion_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'top_confused_pairs': confused_pairs,
            'total_samples': len(patterns),
            'accuracy': float(accuracy),
            'avg_confidence': float(avg_confidence)
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Confusion analysis saved to: {confusion_file}")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    
    return {
        'accuracy': accuracy,
        'report': report,
        'report_file': str(report_file),
        'results_file': str(csv_file),
        'confusion_file': str(confusion_file)
    }

if __name__ == "__main__":
    try:
        results = evaluate_hybrid_model()
        print(f"\n🎯 Final Accuracy: {results['accuracy']:.4f}")
        print(f"📄 Reports saved in: {LOG_DIR}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
