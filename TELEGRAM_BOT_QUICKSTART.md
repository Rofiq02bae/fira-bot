# Quick Start - Telegram Bot

## 🚀 Menjalankan Bot

### Step 1: Pastikan FastAPI Server Running

```powershell
# Terminal 1 - Run FastAPI Server
uvicorn app:app --reload
```

Output yang diharapkan:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Application startup complete.
```

### Step 2: Jalankan Telegram Bot

```powershell
# Terminal 2 - Run Telegram Bot (setelah FastAPI running)
python .\interfaces\telegram_bot.py
```

Output yang diharapkan:
```
2025-10-22 20:47:13 - __main__ - INFO - 🔄 Initializing Telegram Bot (API Client)...
2025-10-22 20:47:14 - __main__ - INFO - ✅ Connected to API server
2025-10-22 20:47:15 - __main__ - INFO - 🔑 Verifying Telegram token...
2025-10-22 20:47:16 - __main__ - INFO - ✅ Token valid: @your_bot_name
2025-10-22 20:47:16 - __main__ - INFO - ✅ Telegram Bot initialized: @your_bot_name
2025-10-22 20:47:16 - __main__ - INFO - 🚀 Starting Telegram Bot (API Client)...
```

## ⚙️ Environment Variables

Edit file `.env`:

```env
# Telegram Bot Token dari @BotFather
TELEGRAM_BOT_TOKEN=your_telegram_token_here

# API Server URL (optional, default: http://localhost:8000)
API_BASE_URL=http://localhost:8000
```

## 🔍 Troubleshooting

### Error: "Timed out" saat initialize

**Penyebab:** Token Telegram invalid atau tidak bisa connect ke Telegram API

**Solusi:**
1. Verify token dengan curl:
   ```powershell
   curl https://api.telegram.org/botYOUR_TOKEN/getMe
   ```
2. Check internet connection
3. Generate token baru dari @BotFather jika perlu

### Error: "API server not available"

**Penyebab:** FastAPI server belum running

**Solusi:**
1. Jalankan FastAPI server dulu:
   ```powershell
   uvicorn app:app --reload
   ```
2. Pastikan server berjalan di `http://localhost:8000`
3. Test dengan browser: http://localhost:8000/health

### Error: "Cannot connect to API server"

**Penyebab:** API URL salah atau server tidak accessible

**Solusi:**
1. Check API_BASE_URL di `.env`
2. Test API health endpoint:
   ```powershell
   curl http://localhost:8000/health
   ```
3. Pastikan port 8000 tidak digunakan aplikasi lain

## 📱 Menggunakan Bot

1. Buka Telegram
2. Cari bot Anda (sesuai username di @BotFather)
3. Klik `/start`
4. Coba tanyakan:
   - "jam buka bappenda"
   - "alamat dinsos tegal"
   - "cara buat ktp"

## 🎯 Commands

- `/start` - Mulai bot
- `/help` - Bantuan
- `/status` - Status sistem
- `/intents` - Daftar intent

## 📊 Monitoring

Log akan muncul di terminal:
```
2025-10-22 20:50:30 - __main__ - INFO - 💬 Message from User: jam buka bappenda
2025-10-22 20:50:31 - __main__ - INFO - 📝 ✅ User -> bappenda_info (conf: 0.956)
```

## 🛑 Stop Bot

Press `Ctrl+C` di terminal untuk stop bot dengan graceful shutdown.

## ✅ Checklist

- [ ] File `.env` sudah ada dengan token valid
- [ ] FastAPI server running di terminal 1
- [ ] Bot running di terminal 2
- [ ] Bot respond di Telegram
- [ ] `/status` menunjukkan sistem online

## 🔧 Architecture

```
[User] -> [Telegram] -> [telegram_bot.py] -> HTTP -> [app.py FastAPI] -> [NLU Service] -> [LSTM/BERT]
                                                                                  |
                                                                            [Response]
                                                                                  |
[User] <- [Telegram] <- [telegram_bot.py] <- HTTP <- [app.py FastAPI] <----------
```

Bot = Client (ringan, tidak load model)
API Server = Heavy processing (load model, predict, dll)
