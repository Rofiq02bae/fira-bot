#!/usr/bin/env python3
"""
Script untuk menambah training candidates ke dataset utama
"""

import pandas as pd
import sys
import os

def add_to_dataset(updates_file: str, dataset_file: str = "../data/dataset/data_mentah.csv"):
    """Add evaluated candidates ke dataset utama"""
    
    if not os.path.exists(updates_file):
        print(f"❌ Updates file not found: {updates_file}")
        return
    
    if not os.path.exists(dataset_file):
        print(f"❌ Dataset file not found: {dataset_file}")
        return
    
    try:
        # Load updates
        updates_df = pd.read_csv(updates_file)
        
        # Load existing dataset
        dataset_df = pd.read_csv(dataset_file)
        
        # Prepare new rows
        new_rows = []
        for _, row in updates_df.iterrows():
            if row['suggested_intent'] and row['suggested_response']:
                new_row = {
                    'intent': row['suggested_intent'],
                    'pattern': row['text'],
                    'response_type': 'static',
                    'response': row['suggested_response']
                }
                new_rows.append(new_row)
        
        if new_rows:
            # Add to dataset
            new_df = pd.DataFrame(new_rows)
            updated_dataset = pd.concat([dataset_df, new_df], ignore_index=True)
            
            # Save backup
            backup_file = dataset_file.replace('.csv', '_backup.csv')
            dataset_df.to_csv(backup_file, index=False)
            print(f"💾 Backup saved: {backup_file}")
            
            # Save updated dataset
            updated_dataset.to_csv(dataset_file, index=False)
            print(f"✅ Added {len(new_rows)} new patterns to dataset!")
            print(f"📁 Updated: {dataset_file}")
            
            # Show summary
            print(f"\n📊 Dataset now has:")
            print(f"   Total rows: {len(updated_dataset)}")
            print(f"   Unique intents: {updated_dataset['intent'].nunique()}")
            
        else:
            print("ℹ️  No valid updates to add.")
            
    except Exception as e:
        print(f"❌ Error adding to dataset: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python add_to_dataset.py <updates_file>")
        print("Example: python add_to_dataset.py data/training_candidates/dataset_updates.csv")
        sys.exit(1)
    
    updates_file = sys.argv[1]
    add_to_dataset(updates_file)