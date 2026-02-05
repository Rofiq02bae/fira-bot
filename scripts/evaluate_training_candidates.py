#!/usr/bin/env python3
"""
Script untuk evaluate dan add training candidates ke dataset
"""

import pandas as pd
import json
import os
import sys

# Add parent directory to path untuk import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.query_logger import query_logger

def evaluate_candidates():
    """Evaluate training candidates dan generate dataset update"""
    
    print("🎯 Evaluating Training Candidates...")
    
    # Get candidates
    candidates = query_logger.get_training_candidates(limit=100)
    
    if candidates.empty:
        print("✅ No training candidates to evaluate!")
        return
    
    print(f"📋 Found {len(candidates)} candidates for evaluation")
    
    evaluated_count = 0
    for index, candidate in candidates.iterrows():
        print(f"\n--- Candidate {evaluated_count + 1} ---")
        print(f"Query: '{candidate['text']}'")
        print(f"Predicted: {candidate['predicted_intent']} (conf: {candidate['confidence']:.3f})")
        print(f"Method: {candidate['method_used']}")
        
        # Manual evaluation (bisa diotomasi lebih lanjut)
        print("\nOptions:")
        print("1. Add to existing intent")
        print("2. Create new intent") 
        print("3. Skip")
        print("4. Mark as noise")
        print("5. Exit evaluation")
        
        choice = input("Choose option (1-5): ").strip()
        
        if choice == "1":
            # Add to existing intent
            intent = input("Enter existing intent name: ").strip()
            response = input("Enter response text: ").strip()
            
            query_logger.mark_as_evaluated(
                candidate['text'], 
                suggested_intent=intent,
                suggested_response=response
            )
            evaluated_count += 1
            print(f"✅ Added to intent: {intent}")
            
        elif choice == "2":
            # Create new intent
            intent = input("Enter new intent name: ").strip()
            response = input("Enter response text: ").strip()
            
            query_logger.mark_as_evaluated(
                candidate['text'],
                suggested_intent=intent, 
                suggested_response=response
            )
            evaluated_count += 1
            print(f"✅ Created new intent: {intent}")
            
        elif choice == "3":
            # Skip
            query_logger.mark_as_evaluated(candidate['text'])
            evaluated_count += 1
            print("⏭️  Skipped")
            
        elif choice == "4":
            # Mark as noise
            query_logger.mark_as_evaluated(candidate['text'])
            evaluated_count += 1
            print("🗑️  Marked as noise")
            
        elif choice == "5":
            print("👋 Exiting evaluation...")
            break
            
        else:
            print("❌ Invalid choice, skipping...")
            continue
    
    print(f"\n✅ Evaluated {evaluated_count} candidates!")
    
    # Generate dataset update report
    generate_update_report()

def generate_update_report():
    """Generate report untuk dataset updates"""
    try:
        csv_file = query_logger.csv_file
        
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            evaluated = df[df['evaluated'] == True]
            suggested_updates = evaluated[evaluated['suggested_intent'] != '']
            
            if not suggested_updates.empty:
                print(f"\n📝 SUGGESTED DATASET UPDATES ({len(suggested_updates)}):")
                print("=" * 60)
                
                for _, update in suggested_updates.iterrows():
                    print(f"Text: '{update['text']}'")
                    print(f"Intent: {update['suggested_intent']}")
                    print(f"Response: {update['suggested_response']}")
                    print("-" * 40)
                
                # Export to CSV untuk easy import
                update_file = "../data/training_candidates/dataset_updates.csv"
                os.makedirs(os.path.dirname(update_file), exist_ok=True)
                
                suggested_updates[['text', 'suggested_intent', 'suggested_response']].to_csv(
                    update_file, index=False
                )
                print(f"💾 Updates exported to: {update_file}")
                
                # Generate import commands
                print(f"\n🔧 IMPORT COMMANDS:")
                print(f"# Untuk menambah ke dataset utama:")
                print(f"python scripts/add_to_dataset.py {update_file}")
                
            else:
                print("ℹ️  No suggested updates found.")
                
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

def show_statistics():
    """Show statistics about training candidates"""
    try:
        csv_file = query_logger.csv_file
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            
            print(f"\n📊 TRAINING CANDIDATES STATISTICS:")
            print(f"Total candidates: {len(df)}")
            print(f"Evaluated: {len(df[df['evaluated'] == True])}")
            print(f"Pending evaluation: {len(df[df['evaluated'] == False])}")
            print(f"With suggested intents: {len(df[df['suggested_intent'] != ''])}")
            
            # Show top predicted intents
            intent_counts = df['predicted_intent'].value_counts().head(10)
            print(f"\n🔝 Top predicted intents:")
            for intent, count in intent_counts.items():
                print(f"  {intent}: {count}")
                
    except Exception as e:
        print(f"Error showing statistics: {e}")

if __name__ == "__main__":
    print("🤖 NLU Training Candidates Evaluator")
    print("=" * 50)
    
    show_statistics()
    evaluate_candidates()