#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk menghapus duplikasi pattern dalam dataset.
"""

# Fix encoding HARUS di awal sebelum import lain
import encoding_fix
from encoding_fix import get_data_path

import pandas as pd
import numpy as np
from collections import defaultdict


def remove_duplicate_patterns(input_file, output_file):
    """
    Menghapus duplikasi pattern dalam kolom pattern yang dipisahkan oleh delimiter |
    Delimiter kolom: koma (,)
    Delimiter array pattern: pipe (|)
    """
    print("🔄 Memulai proses penghapusan duplikasi pattern...")
    
    # Baca file CSV dengan delimiter koma
    try:
        df = pd.read_csv(input_file, delimiter=',', header=None, 
                        names=['intent', 'patterns', 'response_type', 'response'],
                        encoding='utf-8')
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return None
    
    print(f"📊 Data awal: {len(df)} baris")
    
    cleaned_data = []
    total_patterns_before = 0
    total_patterns_after = 0
    rows_with_duplicates = 0
    
    for index, row in df.iterrows():
        intent = str(row['intent'])
        patterns_str = str(row['patterns'])
        response_type = str(row['response_type'])
        response = str(row['response'])
        
        # Split patterns by delimiter |
        patterns = patterns_str.split('|')
        total_patterns_before += len(patterns)
        
        # Clean each pattern: remove whitespace, convert to lowercase
        cleaned_patterns = []
        seen_patterns = set()
        
        for pattern in patterns:
            # Clean the pattern
            clean_pattern = pattern.strip()
            
            # Skip empty patterns
            if not clean_pattern:
                continue
                
            # Skip duplicate patterns (case insensitive)
            pattern_lower = clean_pattern.lower()
            if pattern_lower not in seen_patterns:
                cleaned_patterns.append(clean_pattern)  # Simpan original case
                seen_patterns.add(pattern_lower)
        
        total_patterns_after += len(cleaned_patterns)
        
        # Hitung jika ada duplikasi yang dihapus
        if len(patterns) != len(cleaned_patterns):
            rows_with_duplicates += 1
        
        # Jika masih ada pattern setelah cleaning, simpan data
        if cleaned_patterns:
            # Gabungkan pattern yang sudah dibersihkan dengan delimiter |
            cleaned_patterns_str = '|'.join(cleaned_patterns)
            
            cleaned_row = {
                'intent': intent,
                'patterns': cleaned_patterns_str,
                'response_type': response_type,
                'response': response
            }
            cleaned_data.append(cleaned_row)
    
    # Buat DataFrame dari data yang sudah dibersihkan
    cleaned_df = pd.DataFrame(cleaned_data)
    
    print(f"✅ Data setelah dibersihkan: {len(cleaned_df)} baris")
    print(f"📊 Total pattern sebelum: {total_patterns_before}")
    print(f"📊 Total pattern setelah: {total_patterns_after}")
    print(f"🗑️  Pattern duplikat dihapus: {total_patterns_before - total_patterns_after}")
    print(f"📝 Baris dengan duplikasi: {rows_with_duplicates}")
    
    # Simpan ke file CSV baru dengan delimiter koma
    cleaned_df.to_csv(output_file, sep=',', index=False, header=False, encoding='utf-8')
    
    # Tampilkan sample hasil
    print("\n📋 Sample hasil (3 baris pertama):")
    print("-" * 80)
    for i in range(min(3, len(cleaned_df))):
        row = cleaned_df.iloc[i]
        patterns = row['patterns'].split('|')
        print(f"Baris {i+1}:")
        print(f"  Intent: {row['intent']}")
        print(f"  Response Type: {row['response_type']}")
        print(f"  Response: {row['response'][:50]}...")
        print(f"  Patterns ({len(patterns)}):")
        for j, pattern in enumerate(patterns[:5]):  # Tampilkan 5 pattern pertama
            print(f"    {j+1}. {pattern}")
        if len(patterns) > 5:
            print(f"    ... dan {len(patterns) - 5} pattern lainnya")
        print("-" * 80)
    
    return cleaned_df

def analyze_duplicates_per_intent(input_file):
    """
    Analisis detail duplikasi pattern per intent
    """
    print("\n🔍 ANALISIS DETAIL DUPLIKASI PER INTENT:")
    
    # Baca file asli untuk analisis
    try:
        df = pd.read_csv(input_file, delimiter=',', header=None, 
                        names=['intent', 'patterns', 'response_type', 'response'],
                        encoding='utf-8')
    except Exception as e:
        print(f"❌ Error membaca file: {e}")
        return
    
    # Analisis duplikasi per intent
    intent_analysis = {}
    
    for index, row in df.iterrows():
        intent = str(row['intent'])
        patterns = str(row['patterns']).split('|')
        
        # Hitung duplikasi (case insensitive)
        original_patterns = [p.strip() for p in patterns if p.strip()]
        unique_patterns_lower = set()
        unique_patterns_original = []
        
        for pattern in original_patterns:
            pattern_lower = pattern.lower()
            if pattern_lower not in unique_patterns_lower:
                unique_patterns_lower.add(pattern_lower)
                unique_patterns_original.append(pattern)
        
        duplicates_count = len(original_patterns) - len(unique_patterns_original)
        
        if duplicates_count > 0:
            intent_analysis[intent] = {
                'total_patterns': len(original_patterns),
                'unique_patterns': len(unique_patterns_original),
                'duplicates_removed': duplicates_count,
                'duplicate_percentage': (duplicates_count / len(original_patterns)) * 100
            }
    
    # Tampilkan hasil analisis
    if intent_analysis:
        print(f"🎯 Ditemukan {len(intent_analysis)} intent dengan pattern duplikat:")
        print("=" * 90)
        print(f"{'Intent':<30} {'Total':<8} {'Unique':<8} {'Duplikat':<10} {'% Duplikat':<12}")
        print("-" * 90)
        
        for intent, stats in sorted(intent_analysis.items(), 
                                  key=lambda x: x[1]['duplicates_removed'], reverse=True)[:15]:
            print(f"{intent[:28]:<30} {stats['total_patterns']:<8} {stats['unique_patterns']:<8} "
                  f"{stats['duplicates_removed']:<10} {stats['duplicate_percentage']:<12.1f}%")
        
        if len(intent_analysis) > 15:
            print(f"... dan {len(intent_analysis) - 15} intent lainnya")
        print("=" * 90)
        
        # Statistik summary
        total_duplicates = sum(stats['duplicates_removed'] for stats in intent_analysis.values())
        avg_duplicate_percentage = np.mean([stats['duplicate_percentage'] for stats in intent_analysis.values()])
        
        print(f"\n📈 Statistik Summary:")
        print(f"   ├─ Total pattern duplikat: {total_duplicates}")
        print(f"   ├─ Rata-rata duplikasi: {avg_duplicate_percentage:.1f}%")
        print(f"   └─ Intent dengan duplikasi terbanyak: {max(intent_analysis.items(), key=lambda x: x[1]['duplicates_removed'])[0]}")
    
    else:
        print("✅ Tidak ditemukan pattern duplikat")
    
    return intent_analysis

def validate_output_file(output_file):
    """
    Validasi file output untuk memastikan format benar
    """
    print(f"\n🔍 VALIDASI FILE OUTPUT: {output_file}")
    
    try:
        # Baca file output
        df_output = pd.read_csv(output_file, delimiter=',', header=None,
                               names=['intent', 'patterns', 'response_type', 'response'],
                               encoding='utf-8')
        
        print(f"✅ File output berhasil dibaca: {len(df_output)} baris")
        
        # Cek sample pattern
        print("📋 Sample pattern dari file output:")
        for i in range(min(2, len(df_output))):
            row = df_output.iloc[i]
            patterns = str(row['patterns']).split('|')
            print(f"  Baris {i+1}: {len(patterns)} pattern")
            if patterns:
                print(f"    Contoh: '{patterns[0][:30]}...'")
        
        # Cek delimiter pattern
        sample_patterns = str(df_output.iloc[0]['patterns'])
        if '|' in sample_patterns:
            print("✅ Delimiter pattern '|' terdeteksi dengan benar")
        else:
            print("❌ Delimiter pattern '|' tidak terdeteksi")
            
    except Exception as e:
        print(f"❌ Error validasi file output: {e}")

if __name__ == "__main__":
    # Konfigurasi file
    input_file = get_data_path("data_mentah.csv")
    output_file = get_data_path("data_tanpa_duplikat.csv")
    
    print("=" * 80)
    print("🛠️  TOOL PENGHAPUS DUPLIKASI PATTERN")
    print("=" * 80)
    print("📝 Format:")
    print("   - Delimiter kolom: koma (,)")
    print("   - Delimiter array pattern: pipe (|)")
    print("=" * 80)
    
    try:
        # Analisis data sebelum cleaning
        intent_analysis = analyze_duplicates_per_intent(input_file)
        
        print("\n" + "=" * 80)
        # Jalankan pembersihan data
        cleaned_df = remove_duplicate_patterns(input_file, output_file)
        
        if cleaned_df is not None:
            print("\n" + "=" * 80)
            # Validasi file output
            validate_output_file(output_file)
            
            print("\n" + "=" * 80)
            print("✅ PROSES SELESAI!")
            print(f"📁 File input: {input_file}")
            print(f"📁 File output: {output_file}")
            print(f"📊 Statistik akhir:")
            print(f"   ├─ Baris data: {len(cleaned_df)}")
            print(f"   ├─ Intent unik: {cleaned_df['intent'].nunique()}")
            
            # Hitung total pattern setelah cleaning
            total_patterns = sum(len(str(p).split('|')) for p in cleaned_df['patterns'])
            print(f"   └─ Total pattern unik: {total_patterns}")
            
    except FileNotFoundError:
        print(f"❌ File {input_file} tidak ditemukan!")
        print("Pastikan file berada di direktori yang sama dengan script ini.")
    except Exception as e:
        print(f"❌ Terjadi error: {e}")
        import traceback
        traceback.print_exc()