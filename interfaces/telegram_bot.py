import os
import json
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import Dict, Any, Optional
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler, CallbackQueryHandler,
    filters, ContextTypes
)

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class TelegramBot:
    def __init__(self, token: str, api_base_url: str = "http://localhost:8000"):
        self.token = token
        self.api_base_url = api_base_url.rstrip('/')
        self.application = None
        self.bot_info = None
        self.session: Optional[aiohttp.ClientSession] = None

        self.config = {
            'max_message_length': 4096,
            'typing_delay': 0.5,
            'api_timeout': 30,
            'welcome_message': (
                "🤖 **Selamat datang di Chatbot Asisten**\n\n"
                "Tanyakan langsung seperti:\n"
                "- \"cara buat kk\"\n"
                "- \"siapa kamu?\"\n"
                "- \"cara buat ktp\"\n\n"
                "Tekan /help untuk bantuan lebih lanjut."
            ),
            'help_message': (
                "🆘 **Bantuan Penggunaan Bot**\n\n"
                "/start - Memulai bot\n"
                "/help - Bantuan\n"
                "/status - Status bot\n"
                "/intents - Daftar intent atau pertanyaan terkait"
            ),
            'error_messages': {
                'api_unavailable': "❌ Layanan sedang tidak tersedia.",
                'timeout': "⏰ Request timeout.",
                'general_error': "❌ Terjadi kesalahan sistem."
            }
        }

    # ---------------------------------------------------
    # INITIALIZATION
    # ---------------------------------------------------
    async def initialize(self):
        logger.info("🔄 Initializing bot...")

        # Create session
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config['api_timeout'])
        )

        # Test API (optional)
        await self._safe_test_api()

        # Create Telegram application
        self.application = Application.builder().token(self.token).build()

        # Verify token
        await self._verify_token()

        # Register all handlers
        self._register_handlers()

        logger.info("✅ Initialization complete")

    async def _safe_test_api(self):
        try:
            await self._test_api_connection()
        except Exception as e:
            logger.warning(f"API not ready: {e}")

    async def _verify_token(self):
        try:
            bot = Bot(self.token)
            self.bot_info = await bot.get_me()
            logger.info(f"🔑 Token OK → @{self.bot_info.username}")
        except Exception as e:
            raise ValueError(f"Invalid Telegram token: {e}")

    async def _test_api_connection(self):
        async with self.session.get(f"{self.api_base_url}/health") as r:
            if r.status != 200:
                raise Exception(f"API status: {r.status}")
            data = await r.json()
            logger.info(f"✅ API Ready: {data}")

    # ---------------------------------------------------
    # HANDLERS
    # ---------------------------------------------------
    def _register_handlers(self):
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("help", self._help))
        self.application.add_handler(CommandHandler("status", self._status))
        self.application.add_handler(CommandHandler("intents", self._intents))

        self.application.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message
        ))
        
        # Callback query handler for buttons
        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        await update.message.reply_text(
            f"Halo {update.effective_user.first_name}!\n"
            f"{self.config['welcome_message']}",
            parse_mode="Markdown"
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        await update.message.reply_text(self.config['help_message'], parse_mode="Markdown")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        try:
            data = await self._fetch("/health")
            msg = (
                "🖥️ **Status Sistem**\n\n"
                f"API: {'✅ Online' if data else '❌ Error'}\n"
                f"Model: {'Ready' if data.get('model_loaded') else 'Offline'}"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text("❌ Tidak bisa cek status")

    async def _intents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        try:
            data = await self._fetch("/intents")
            intents = data.get("available_intents", [])
            if not intents:
                return await update.message.reply_text("Tidak ada intent.")
            msg = "<b>Intent Tersedia:</b>\n" + "\n".join(f"• {i}" for i in intents)
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception:
            await update.message.reply_text("❌ Error mengambil intent")

    # ---------------------------------------------------
    # MESSAGE PROCESSING
    # ---------------------------------------------------
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        # Safety check
        if not update.message or not update.message.text:
            logger.error("⚠ Message text is empty or not supported")
            return await update.message.reply_text(
                "⚠ Aku cuma bisa baca pesan teks yaa 😊"
            )

        text = update.message.text
        user = update.effective_user

        logger.info(f"💬 {user.first_name}: {text}")

        try:
            await self._typing(update)
            result = await self._call_chat_api(text)
            await self._send_response(update, result)
            self._log_interaction(user, text, result)

        except asyncio.TimeoutError:
            await update.message.reply_text(self.config['error_messages']['timeout'])

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button clicks"""
        query = update.callback_query
        await query.answer() # Acknowledge intera                                                                                                    ction

        text = query.data
        user = update.effective_user
        
        logger.info(f"🔄 Callback {user.first_name}: {text}")
        
        # Send a message to show what was clicked (optional, but good UX)
        # await query.message.reply_text(f"👉 Anda memilih: {text}")
        
        # Process as if user typed it
        try:
            await self._typing(update)
            result = await self._call_chat_api(text)
            await self._send_response(update, result)
            self._log_interaction(user, text, result)
            
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.message.reply_text(self.config['error_messages']['general_error'])

    async def _call_chat_api(self, message: str) -> Dict[str, Any]:
        try:
            return await self._post_json("/api/chat", {"text": message})
        except asyncio.TimeoutError:
            raise
        except Exception:
            return {
                "success": False,
                "response": self.config["error_messages"]["general_error"],
                "intent": "error",
                "confidence": 0.0
            }

    # ---------------------------------------------------
    # HTTP UTILS
    # ---------------------------------------------------
    async def _fetch(self, path: str):
        async with self.session.get(f"{self.api_base_url}{path}") as r:
            return await r.json()

    async def _post_json(self, path: str, payload: dict):
        async with self.session.post(
            f"{self.api_base_url}{path}",
            json=payload
        ) as r:
            if r.status == 200:
                return await r.json()
            if r.status == 503:
                return {"success": False, "response": self.config["error_messages"]["api_unavailable"]}

            return {"success": False, "response": self.config["error_messages"]["general_error"]}

    # ---------------------------------------------------
    # JSON RESPONSE RENDER
    # ---------------------------------------------------
    async def _render_json_response(self, data: Dict[str, Any]) -> str:
        r_type = data.get("type", "text")

        # TEXT RESPONSE
        if r_type == "text":
            return data.get("body", "")

        # LIST RESPONSE WITH TITLE + ITEMS
        if r_type == "list":
            title = data.get("title", "")
            items = data.get("items", data.get("body", []))

            formatted_items = "\n".join(f"• {item}" for item in items)

            if title:
                return f"{title}\n\n{formatted_items}"
            return formatted_items

        # fallback
        return str(data)

    async def _send_response(self, update: Update, result: Dict[str, Any]):
        api_response = result.get("response", "")

        # Jika API mengirim JSON string → parse
        if isinstance(api_response, str):
            try:
                api_response = json.loads(api_response)
            except Exception:
                pass

        # Render JSON → text
        if isinstance(api_response, dict):
            text = await self._render_json_response(api_response)
        else:
            text = str(api_response)

        # Batasi panjang pesan
        if len(text) > self.config['max_message_length']:
            text = text[:4000] + "\n\n📝 _Pesan dipotong_"

        # Check for options to build Keyboard
        reply_markup = None
        options = result.get("options", [])
        if options and isinstance(options, list):
            keyboard = []
            for opt in options:
                label = opt.get("label", "Option")
                # Use label as value sent to server
                keyboard.append([InlineKeyboardButton(label, callback_data=label)])
            
            reply_markup = InlineKeyboardMarkup(keyboard)

        # Send info if there is reply_markup
        if reply_markup:
             await update.effective_message.reply_text(text, reply_markup=reply_markup)
        else:
             await update.effective_message.reply_text(text)

    # ---------------------------------------------------
    # UTILITIES
    # ---------------------------------------------------
    async def _typing(self, update: Update):
        try:
            await update.effective_chat.send_action("typing")
            await asyncio.sleep(self.config['typing_delay'])
        except Exception:
            pass

    def _log_interaction(self, user, message, result):
        logger.info(
            f"📝 {user.first_name} -> {result.get('intent')} "
            f"(conf: {result.get('confidence', 0):.3f})"
        )

    # ---------------------------------------------------
    # LIFECYCLE
    # ---------------------------------------------------
    async def start(self):
        if not self.application:
            await self.initialize()

        logger.info("🚀 Starting bot...")
        await self.application.initialize()
        await self.application.start()

        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )

        logger.info("✅ Bot is running!")
        stop_event = asyncio.Event()
        await stop_event.wait()

    async def stop(self):
        logger.info("🛑 Stopping bot...")

        if self.application and self.application.updater:
            await self.application.updater.stop()

        if self.application:
            await self.application.stop()
            await self.application.shutdown()

        if self.session and not self.session.closed:
            await self.session.close()

        logger.info("✅ Bot stopped")


# FACTORY
def create_telegram_bot(token=None, api_url=None):
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Missing Telegram Bot Token.")
    api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    return TelegramBot(token, api_url)


# MAIN
async def main():
    bot = None
    try:
        bot = create_telegram_bot()
        await bot.start()
    except KeyboardInterrupt:
        logger.info("⛔ CTRL+C detected, stopping...")
    finally:
        if bot:
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
