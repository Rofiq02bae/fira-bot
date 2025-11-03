import os
import logging
import asyncio
import aiohttp
import json
import signal
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

class TelegramBot:
    def __init__(self, token: str, api_base_url: str = "http://localhost:8000"):
        self.token = token
        self.api_base_url = api_base_url.rstrip('/')
        self.application = None
        self.bot_info = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Bot configuration
        self.config = {
            'max_message_length': 4096,
            'typing_delay': 0.5,
            'api_timeout': 30,
            'welcome_message': """
🤖 **Selamat datang di Chatbot Asisten**

Saya adalah asisten virtual yang siap membantu menjawab pertanyaan tentang berbagai layanan

**Cara menggunakan:**
Tanyakan langsung seperti:
- "Bappenda buka jam berapa?"
- "Alamat dinsos tegal dimana?"
- "Cara buat KTP?"
- "cara buat kk?"

Tekan /help untuk bantuan lebih lanjut.
            """,
            'help_message': """
🆘 **Bantuan Penggunaan Bot**

**Perintah yang tersedia:**
/start - Memulai bot
/help - Menampilkan bantuan ini
/status - Status sistem bot
/intents - Daftar pertanyaan terkait yang dikenali

**Contoh pertanyaan:**
• "jam buka bappenda"
• "alamat dinsos tegal" 
• "prosedur pembuatan ktp"
• "cara buat kk?"

Bot akan memberikan respon terbaik berdasarkan pemahaman AI.
            """,
            'error_messages': {
                'api_unavailable': "❌ Layanan sedang tidak tersedia. Silakan coba lagi nanti.",
                'timeout': "⏰ Waktu tunggu terlalu lama. Silakan coba lagi.",
                'general_error': "❌ Terjadi kesalahan sistem. Silakan coba lagi."
            }
        }
    
    async def initialize(self):
        """Initialize bot dan HTTP session"""
        try:
            logger.info("🔄 Initializing Telegram Bot (API Client)...")
            
            # Create HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config['api_timeout'])
            )
            
            # Test API connection first (non-blocking)
            try:
                await self._test_api_connection()
            except Exception as e:
                logger.warning(f"⚠️ API server not available yet: {e}")
                logger.info("Bot will continue, but may not work until API server is started")
            
            # Create application
            self.application = Application.builder().token(self.token).build()
            
            # Get bot info (verify token)
            logger.info("🔑 Verifying Telegram token...")
            bot = Bot(self.token)
            try:
                self.bot_info = await bot.get_me()
                logger.info(f"✅ Token valid: @{self.bot_info.username}")
            except Exception as e:
                logger.error(f"❌ Invalid Telegram token or network issue: {e}")
                raise ValueError(
                    f"Cannot verify Telegram token. "
                    f"Please check: 1) Token is correct, 2) Internet connection is working. "
                    f"Error: {e}"
                )
            
            # Register handlers
            self._register_handlers()
            
            logger.info(f"✅ Telegram Bot initialized: @{self.bot_info.username}")
            logger.info(f"🌐 API Server: {self.api_base_url}")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Telegram Bot: {e}")
            if self.session:
                await self.session.close()
            raise
    
    async def _test_api_connection(self):
        """Test koneksi ke API server"""
        try:
            async with self.session.get(f"{self.api_base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"✅ API Connection successful: {data.get('status', 'unknown')}")
                    return True
                else:
                    raise Exception(f"API returned status {response.status}")
        except Exception as e:
            logger.error(f"❌ API Connection failed: {e}")
            raise Exception(f"Cannot connect to API server: {e}")
    
    def _register_handlers(self):
        """Register semua message handlers"""
        # Command handlers
        self.application.add_handler(CommandHandler("start", self._start_handler))
        self.application.add_handler(CommandHandler("help", self._help_handler))
        self.application.add_handler(CommandHandler("status", self._status_handler))
        self.application.add_handler(CommandHandler("intents", self._intents_handler))
        
        # Message handler untuk semua text messages
        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND, 
            self._message_handler
        ))
        
        # Error handler (note: error handler tidak async dalam python-telegram-bot)
        # Skip error handler untuk sekarang, gunakan logging saja
    
    async def _start_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        await self._send_typing_action(update)
        
        user = update.effective_user
        welcome_msg = f"Halo {user.first_name}! {self.config['welcome_message']}"
        
        await update.message.reply_text(
            welcome_msg,
            parse_mode='Markdown'
        )
    
    async def _help_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        await self._send_typing_action(update)
        await update.message.reply_text(
            self.config['help_message'],
            parse_mode='Markdown'
        )
    
    async def _status_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        try:
            await self._send_typing_action(update)
            
            async with self.session.get(f"{self.api_base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    
                    status_msg = f"""
🖥️ **Status Sistem**

**API Status:** ✅ Online
**NLU Service:** {data.get('status', 'unknown')}
**LSTM Model:** {'✅ Ready' if data.get('model_loaded', False) else '❌ Offline'}
**BERT Model:** {'✅ Ready' if data.get('bert_available', False) else '❌ Offline'}
**Intents:** {data.get('intents_count', 0)} intent

Bot berjalan dengan normal 🚀
                    """
                    
                    await update.message.reply_text(status_msg, parse_mode='Markdown')
                else:
                    await update.message.reply_text("❌ Tidak bisa mendapatkan status server")
                    
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text("❌ Error mendapatkan status sistem")
    
    async def _intents_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /intents command"""
        try:
            await self._send_typing_action(update)
            
            async with self.session.get(f"{self.api_base_url}/intents") as response:
                if response.status == 200:
                    data = await response.json()
                    intents = data.get('available_intents', [])
                    
                    if not intents:
                        await update.message.reply_text("❌ Tidak ada intent yang tersedia")
                        return
                    
                    intent_list = "\n".join([f"• {intent}" for intent in intents])
                    intent_msg = f"""
🎯 <b>Daftar Intent yang Dikenali</b>

{intent_list}

Total: {len(intents)} intent
                    """
                    
                    await update.message.reply_text(intent_msg, parse_mode='HTML')
                else:
                    await update.message.reply_text("❌ Tidak bisa mendapatkan daftar intent")
                    
        except Exception as e:
            logger.error(f"Intents command error: {e}")
            await update.message.reply_text("❌ Error mendapatkan daftar intent")
    
    async def _message_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle semua text messages"""
        user_message = update.message.text
        user = update.effective_user
        
        logger.info(f"💬 Message from {user.first_name}: {user_message}")
        
        try:
            # Show typing indicator
            await self._send_typing_action(update)
            
            # Process message via API
            result = await self._call_chat_api(user_message)
            
            # Send response
            await self._send_response(update, result)
            
            # Log interaction
            self._log_interaction(user, user_message, result)
            
        except asyncio.TimeoutError:
            logger.error(f"API timeout for user {user.first_name}")
            await update.message.reply_text(self.config['error_messages']['timeout'])
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await update.message.reply_text(self.config['error_messages']['general_error'])
    
    async def _call_chat_api(self, message: str) -> Dict[str, Any]:
        """Call chat API endpoint"""
        try:
            payload = {"text": message}
            
            async with self.session.post(
                f"{self.api_base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'success': True,
                        'response': data.get('response', 'Tidak ada response'),
                        'intent': data.get('predicted_intent', 'unknown'),
                        'confidence': data.get('confidence', 0.0),
                        'method_used': data.get('method_used', 'unknown'),
                        'processing_time': data.get('processing_time', 0)
                    }
                elif response.status == 503:
                    return {
                        'success': False,
                        'response': self.config['error_messages']['api_unavailable'],
                        'intent': 'error',
                        'confidence': 0.0,
                        'method_used': 'api_error'
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return {
                        'success': False,
                        'response': self.config['error_messages']['general_error'],
                        'intent': 'error',
                        'confidence': 0.0,
                        'method_used': 'api_error'
                    }
                    
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            logger.error(f"API call error: {e}")
            return {
                'success': False,
                'response': self.config['error_messages']['general_error'],
                'intent': 'error',
                'confidence': 0.0,
                'method_used': 'api_error'
            }
    
    async def _send_response(self, update: Update, result: Dict[str, Any]):
        """Send formatted response ke user"""
        response_text = result['response']
        
        # Potong message jika terlalu panjang untuk Telegram
        if len(response_text) > self.config['max_message_length']:
            response_text = response_text[:self.config['max_message_length'] - 100] + "...\n\n📝 _Response dipotong karena terlalu panjang_"
        
        await update.message.reply_text(
            response_text,
            parse_mode='Markdown'
        )
    
    async def _send_typing_action(self, update: Update):
        """Send typing action indicator"""
        try:
            await update.message.chat.send_action(action="typing")
            await asyncio.sleep(self.config['typing_delay'])
        except Exception as e:
            logger.warning(f"Failed to send typing action: {e}")
    
    def _log_interaction(self, user, user_message: str, result: Dict[str, Any]):
        """Log user interaction untuk analytics"""
        log_data = {
            'user_id': user.id,
            'user_name': user.first_name,
            'message': user_message,
            'intent': result['intent'],
            'confidence': result.get('confidence', 0),
            'method': result.get('method_used', 'unknown'),
            'success': result.get('success', False),
            'timestamp': asyncio.get_event_loop().time()
        }
        
        status_icon = "✅" if result.get('success') else "❌"
        logger.info(f"📝 {status_icon} {user.first_name} -> {result['intent']} (conf: {result.get('confidence', 0):.3f})")
    
    async def _error_handler(self, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors from handlers"""
        logger.error(f"Bot error: {context.error}")
        if context.error:
            logger.error(f"Error traceback: {context.error}")
    
    async def start(self):
        """Start the bot"""
        if not self.application:
            await self.initialize()
        
        logger.info("🚀 Starting Telegram Bot (API Client)...")
        
        # Initialize and start application
        await self.application.initialize()
        await self.application.start()
        
        # Start polling (this is the correct async way)
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
        logger.info("✅ Bot is running! Press Ctrl+C to stop.")
        
        # Keep running until interrupted
        stop_event = asyncio.Event()
        
        # Set up signal handling
        def signal_handler():
            logger.info("📡 Received stop signal...")
            stop_event.set()
        
        try:
            # Wait for stop signal
            await stop_event.wait()
        except asyncio.CancelledError:
            logger.info("📡 Polling cancelled")
    
    async def stop(self):
        """Stop the bot"""
        logger.info("🛑 Stopping Telegram Bot...")
        
        try:
            # Stop polling
            if self.application and self.application.updater:
                await self.application.updater.stop()
        except Exception as e:
            logger.warning(f"Error stopping updater: {e}")
        
        try:
            # Stop and shutdown application
            if self.application:
                await self.application.stop()
                await self.application.shutdown()
        except Exception as e:
            logger.warning(f"Error stopping application: {e}")
        
        try:
            # Close aiohttp session
            if self.session and not self.session.closed:
                await self.session.close()
        except Exception as e:
            logger.warning(f"Error closing session: {e}")
            
        logger.info("✅ Telegram Bot stopped")

# Factory function
def create_telegram_bot(
    token: str = None, 
    api_url: str = None
) -> TelegramBot:
    """Create Telegram Bot instance"""
    if token is None:
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        
    if not token:
        raise ValueError("Telegram Bot Token tidak ditemukan. Set TELEGRAM_BOT_TOKEN environment variable.")
    
    if api_url is None:
        api_url = os.getenv('API_BASE_URL', 'http://localhost:8000')
    
    return TelegramBot(token, api_url)

async def main():
    """Main function untuk run bot standalone"""
    bot = None
    try:
        logger.info("🎯 Starting Telegram Bot Client...")
        bot = create_telegram_bot()
        await bot.start()
    except KeyboardInterrupt:
        logger.info("⚠️ Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Bot failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if bot:
            await bot.stop()

if __name__ == "__main__":
    asyncio.run(main())