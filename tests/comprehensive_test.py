import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import ModelConfig
from core.nlu_service import HybridNLUService

def run_test():
    print("🔄 Initializing NLU Service...")
    config = ModelConfig()
    service = HybridNLUService(config)
    
    # Load dataset
    data_path = os.path.join("data", "dataset", "data_mentah.csv")
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}")
        return

    print(f"📖 Reading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    results = []
    
    print("🚀 Starting comprehensive testing...")
    
    # Iterate through all intents
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Testing Patterns"):
        expected_intent = row['intent']
        patterns = str(row['pattern']).split('|')
        
        # Check if master (handle boolean or string 'true'/'false')
        is_master_val = row.get('is_master', False)
        if isinstance(is_master_val, str):
            is_master = is_master_val.lower() == 'true'
        else:
            is_master = bool(is_master_val)
        
        for p in patterns:
            p = p.strip()
            if not p: continue
            
            # Run prediction
            res = service.process_query(p)
            
            pred_intent = res['intent']
            confidence = res['confidence']
            method = res.get('method', 'unknown')
            
            # Determine correctness
            correct = False
            # 1. Exact match
            if pred_intent == expected_intent:
                correct = True
            # 2. Master intent triggers clarification (Valid behavior)
            elif is_master and pred_intent == 'clarification_needed':
                correct = True
            # 3. Master intent detected as master intent (also valid if logic decides no ambiguity is present? depends on implementation)
            # Actually, our logic forces clarification for master intents if ambiguous.
            # But let's count 'clarification_needed' as the success state for ambiguous master queries.
                
            results.append({
                'Pattern': p,
                'Expected': expected_intent,
                'Predicted': pred_intent,
                'Confidence': confidence,
                'IsCorrect': correct,
                'Method': method,
                'IsMaster': is_master
            })
            
    # Create Results DataFrame
    res_df = pd.DataFrame(results)
    
    # Save Results
    output_csv = os.path.join("tests", "comprehensive_test_results.csv")
    res_df.to_csv(output_csv, index=False)
    print(f"\n✅ Detailed results saved to: {output_csv}")
    
    # --- Statistics ---
    total = len(res_df)
    correct_count = res_df['IsCorrect'].sum()
    accuracy = correct_count / total if total > 0 else 0
    
    avg_conf_correct = res_df[res_df['IsCorrect'] == True]['Confidence'].mean()
    avg_conf_wrong = res_df[res_df['IsCorrect'] == False]['Confidence'].mean()
    
    print("\n" + "="*40)
    print("📊 PERFORMANCE STATISTICS")
    print("="*40)
    print(f"Total Patterns Tested : {total}")
    print(f"Correct Predictions   : {correct_count}")
    print(f"Accuracy              : {accuracy:.2%}")
    print(f"Avg Confidence (✅)   : {avg_conf_correct:.4f}")
    print(f"Avg Confidence (❌)   : {avg_conf_wrong:.4f}")
    print("="*40)
    
    # --- Visualization ---
    try:
        output_plot = os.path.join("tests", "accuracy_visualization.png")
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Count of Correct vs Incorrect
        plt.subplot(1, 2, 1)
        sns.countplot(data=res_df, x='IsCorrect', palette='viridis')
        plt.title('Prediction Correctness Count')
        plt.ylabel('Count')
        plt.xlabel('Is Correct?')
        
        # Subplot 2: Confidence Distribution
        plt.subplot(1, 2, 2)
        sns.histplot(data=res_df, x='Confidence', hue='IsCorrect', bins=20, multiple='stack', palette='viridis')
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_plot)
        print(f"\n🖼️ Visualization saved to: {output_plot}")
        
    except Exception as e:
        print(f"\n⚠️ Visualization failed: {e}")
        print("Make sure matplotlib and seaborn are installed.")

if __name__ == "__main__":
    run_test()
