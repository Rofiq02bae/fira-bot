#!/bin/bash
# Train the model
# Usage: bash train.sh
echo "🧹 Membersihkan duplikasi data CSV..."
D:/bot/chatbot/chatbot/.venv/Scripts/python.exe ../deduplicate.py --mode advanced --threshold 0.85 --input ../dataset/data_mentah.csv --output ../dataset/dataset_training.csv
echo "✅ Deduplication selesai"

echo "🔧 Memvalidasi dan memperbaiki format CSV..."
D:/bot/chatbot/chatbot/.venv/Scripts/python.exe validate_csv.py

echo "📊 Membagi data untuk training dan validation..."
D:/bot/chatbot/chatbot/.venv/Scripts/python.exe data_splitter.py

echo "🚀 Memulai training model..."
cd ../train_model
D:/bot/chatbot/chatbot/.venv/Scripts/python.exe chatbot_training.py
cd ../scripts

echo "🎉 Training selesai! Model berhasil disimpan di direktori model/"