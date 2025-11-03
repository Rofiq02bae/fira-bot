# Telegram Bot Setup Guide

## 🤖 Telegram Bot Chatbot Bappenda Tegal

Bot Telegram untuk menjawab pertanyaan tentang layanan pemerintah Kabupaten Tegal menggunakan AI (LSTM + BERT).

## 📋 Prerequisites

1. **Python Environment**: Virtual environment sudah aktif
2. **Dependencies**: Sudah terinstall (lihat requirements.txt)
3. **Models**: LSTM dan BERT model sudah ditrain
4. **Telegram Bot Token**: Token dari @BotFather

## 🔧 Setup

### 1. Dapatkan Token Bot

1. Buka Telegram dan cari `@BotFather`
2. Kirim `/newbot` untuk membuat bot baru
3. Ikuti instruksi dan catat token yang diberikan
4. Format token: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

### 2. Setup Environment Variable

Edit file `.env` di root project:

```env
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE
```

**Contoh:**
```env
TELEGRAM_BOT_TOKEN=8227931536:AAGeGZzS_1XFEZk-YOv446fqW7KZAKq554M
```

### 3. Install Dependencies

```powershell
# Aktifkan virtual environment (jika belum)
.\.venv\Scripts\Activate.ps1

# Install telegram bot dependencies
pip install python-telegram-bot==20.7 python-dotenv
```

### 4. Verify Token

Test token Anda dengan curl atau browser:

```powershell
# PowerShell
$token = "YOUR_TOKEN_HERE"
Invoke-WebRequest "https://api.telegram.org/bot$token/getMe" | Select-Object -ExpandProperty Content
```

Atau buka di browser:
```
https://api.telegram.org/botYOUR_TOKEN_HERE/getMe
```

Jika valid, akan menampilkan info bot Anda.

## 🚀 Menjalankan Bot

### Method 1: Langsung dari Terminal

```powershell
# Dari root directory project
python .\interfaces\telegram_bot.py
```

### Method 2: Menggunakan Virtual Environment secara Explicit

```powershell
& "D:/bot/New folder/.venv/Scripts/python.exe" .\interfaces\telegram_bot.py
```

### Output yang Diharapkan

```
2025-10-22 20:30:24,429 - __main__ - INFO - 🔄 Initializing Telegram Bot...
2025-10-22 20:30:37,893 - main - INFO - 🎯 Initializing Modular Hybrid NLU Service...
2025-10-22 20:30:38,148 - core.models.lstm_model - INFO - ✅ LSTM model loaded successfully!
2025-10-22 20:30:39,694 - core.models.bert_model - INFO - ✅ Fine-tuned BERT loaded successfully!
2025-10-22 20:30:39,788 - core.nlu_service - INFO - ✅ HybridNLUService initialized successfully!
2025-10-22 20:30:39,788 - main - INFO - 🚀 Modular Hybrid NLU Service initialized successfully!
2025-10-22 20:30:40,096 - __main__ - INFO - ✅ Telegram Bot initialized: @your_bot_name
2025-10-22 20:30:40,096 - __main__ - INFO - 🚀 Starting Telegram Bot...
```

## 📱 Menggunakan Bot

### Commands yang Tersedia

- `/start` - Memulai bot dan menampilkan welcome message
- `/help` - Menampilkan bantuan penggunaan
- `/status` - Cek status sistem bot (LSTM, BERT, intents)
- `/intents` - Lihat daftar intent yang dikenali bot

### Contoh Pertanyaan

```
- jam buka bappenda
- alamat dinsos tegal
- cara membuat ktp
- prosedur pembuatan kk
- syarat nikah
- info layanan dukcapil
```

## 🐛 Troubleshooting

### Error: `Telegram Bot Token tidak ditemukan`

**Penyebab:** File `.env` tidak ada atau token tidak di-set

**Solusi:**
1. Pastikan file `.env` ada di root directory
2. Isi dengan token yang valid:
   ```env
   TELEGRAM_BOT_TOKEN=your_actual_token_here
   ```

### Error: `ModuleNotFoundError: No module named 'telegram'`

**Penyebab:** Package `python-telegram-bot` belum terinstall

**Solusi:**
```powershell
pip install python-telegram-bot==20.7 python-dotenv
```

### Error: Invalid Token / Unauthorized

**Penyebab:** Token salah atau expired

**Solusi:**
1. Verifikasi token dengan mengakses:
   ```
   https://api.telegram.org/botYOUR_TOKEN/getMe
   ```
2. Jika invalid, dapatkan token baru dari @BotFather
3. Update file `.env`

### Error: `No module named 'main'`

**Penyebab:** Import path tidak benar

**Solusi:** Pastikan menjalankan bot dari root directory:
```powershell
cd "d:\bot\New folder"
python .\interfaces\telegram_bot.py
```

### Error: Models tidak ditemukan

**Penyebab:** LSTM/BERT model belum ditrain

**Solusi:**
```powershell
# Train LSTM model
python .\training\lstm_trainng.py

# Train BERT model (optional)
python .\training\bert_training.py
```

### Bot terhenti saat loading

**Penyebab:** Loading model memakan waktu (normal)

**Catatan:** 
- LSTM loading: ~2-5 detik
- BERT loading: ~5-10 detik
- Total initialization: ~15-20 detik
- Bersabarlah, ini normal untuk first run

## 📊 Monitoring

### Logs

Bot mencatat semua aktivitas:
- Interaksi user
- Intent predictions
- Confidence scores
- Errors

Log format:
```
2025-10-22 20:30:40 - __main__ - INFO - 💬 Message from User: jam buka bappenda
2025-10-22 20:30:41 - __main__ - INFO - 📝 Interaction: User -> bappenda_info (conf: 0.956)
```

### Status Check

Gunakan command `/status` di Telegram untuk cek:
- LSTM Model status
- BERT Model status
- Jumlah intent yang tersedia
- Health status system

## 🔐 Security Notes

1. **Jangan commit token** ke git repository
2. Gunakan `.gitignore` untuk exclude `.env`
3. Generate token baru jika ter-expose
4. Limit access ke file `.env` (chmod 600 di Linux)

## 📝 File Structure

```
interfaces/
├── telegram_bot.py          # Main bot implementation
└── __ini__.py

.env                          # Environment variables (TOKEN disini)
requirements.txt              # Dependencies
```

## 🎯 Next Steps

1. **Test bot** dengan berbagai pertanyaan
2. **Monitor performance** dengan `/status`
3. **Collect feedback** dari users
4. **Improve responses** berdasarkan log
5. **Add features** seperti:
   - Inline keyboards
   - Rich media responses
   - Analytics dashboard
   - Multi-language support

## 📞 Support

Jika masih ada masalah:
1. Check logs di terminal
2. Verify `.env` file
3. Test token validity
4. Ensure models are trained
5. Check internet connection

## ✅ Checklist Setup

- [ ] Virtual environment activated
- [ ] Dependencies installed
- [ ] `.env` file created with valid token
- [ ] LSTM model trained
- [ ] BERT model trained (optional)
- [ ] Bot tested with `/start` command
- [ ] Bot responds to questions

Happy chatting! 🤖✨
