import os
import json
import html
import re
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()


class TelegramBot:
    def __init__(self, token: str, api_base_url: str = "http://localhost:8000", skip_api_test: bool = False):
        self.token = token
        self.api_base_url = api_base_url.rstrip('/')
        self.skip_api_test = skip_api_test
        self.application = None
        self.bot_info = None
        self.session: Optional[aiohttp.ClientSession] = None

        self.config = {
            'max_message_length': 4096,
            'typing_delay': 0.5,
            'api_timeout': 30,
            'welcome_message': (
                "🤖 <b>Selamat datang di Chatbot Asisten</b>\n\n"
                "Tanyakan langsung seperti:\n"
                "- \"cara buat kk\"\n"
                "- \"siapa kamu?\"\n"
                "- \"cara buat ktp\"\n\n"
                "Tekan /help untuk bantuan lebih lanjut."
            ),
            'help_message': (
                "🆘 <b>Bantuan Penggunaan Bot</b>\n\n"
                "/start - Memulai bot\n"
                "/help - Bantuan\n"
                "/status - Status bot\n"
                "/intents - Daftar intent tersedia"
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

        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config['api_timeout'])
        )

        if not self.skip_api_test:
            await self._safe_test_api()

        try:
            self.application = Application.builder().token(self.token).build()
            self.bot_info = await self.application.bot.get_me()
            logger.info(f"🔑 Token OK → @{self.bot_info.username}")
        except Exception as e:
            logger.error(f"Failed to create Telegram application: {e}")
            raise ValueError(f"Invalid Telegram token or configuration: {e}")

        self._register_handlers()
        logger.info("✅ Initialization complete")

    async def _safe_test_api(self):
        try:
            await self._test_api_connection()
        except Exception as e:
            logger.warning(f"API not ready (this is OK): {e}")

    async def _test_api_connection(self):
        try:
            async with self.session.get(f"{self.api_base_url}/health") as r:
                if r.status != 200:
                    raise Exception(f"API status: {r.status}")
                data = await r.json()
                logger.info(f"✅ API Ready: {data}")
        except Exception as e:
            raise e

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

        self.application.add_handler(CallbackQueryHandler(self._handle_callback))

    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        await update.message.reply_text(
            f"Halo {update.effective_user.first_name}!\n"
            f"{self.config['welcome_message']}",
            parse_mode="HTML"
        )

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        await update.message.reply_text(self.config['help_message'], parse_mode="HTML")

    async def _status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        try:
            data = await self._fetch("/health")
            msg = (
                "🖥️ <b>Status Sistem</b>\n\n"
                f"API: {'✅ Online' if data else '❌ Error'}\n"
                f"Model: {'Ready' if data.get('model_loaded') else 'Offline'}"
            )
            await update.message.reply_text(msg, parse_mode="HTML")
        except Exception:
            await update.message.reply_text("❌ Tidak bisa cek status")

    async def _intents(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        FIX: endpoint /intents return dict {intent_name: {...}},
        bukan list — ambil key-nya saja.
        """
        await self._typing(update)
        try:
            data = await self._fetch("/intents")

            # Endpoint bisa return berbagai format, handle ketiganya
            if isinstance(data, dict):
                # Format 1: {"available_intents": [...]}
                if "available_intents" in data:
                    raw = data["available_intents"]
                    if isinstance(raw, list):
                        intents = raw
                    elif isinstance(raw, dict):
                        intents = list(raw.keys())
                    else:
                        intents = [str(raw)]
                # Format 2: dict langsung {intent_name: {...}, ...}
                else:
                    intents = list(data.keys())
            elif isinstance(data, list):
                intents = data
            else:
                intents = []

            if not intents:
                return await update.message.reply_text("Tidak ada intent tersedia.")

            # Tampilkan dengan paginasi kalau terlalu banyak
            MAX_DISPLAY = 50
            total = len(intents)
            shown = intents[:MAX_DISPLAY]

            lines = [f"<b>Intent Tersedia ({total} total):</b>"]
            for i in sorted(shown):
                lines.append(f"• {html.escape(str(i))}")

            if total > MAX_DISPLAY:
                lines.append(f"\n<i>... dan {total - MAX_DISPLAY} intent lainnya</i>")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")

        except Exception as e:
            logger.error(f"_intents error: {e}")
            await update.message.reply_text("❌ Error mengambil daftar intent")

    # ---------------------------------------------------
    # MESSAGE PROCESSING
    # ---------------------------------------------------
    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        query = update.callback_query
        await query.answer()

        text = query.data
        user = update.effective_user

        logger.info(f"🔄 Callback {user.first_name}: {text}")

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
    def _try_parse_response_payload(self, value: Any) -> Any:
        """
        Parse response payload yang mungkin berupa:
        - dict JSON
        - JSON string normal
        - JSON string escaped dari CSV seperti {""type"":""text"",...}
        - Plain string dari LLM
        """
        if isinstance(value, dict):
            return value

        if not isinstance(value, str):
            return value

        raw = value.strip()
        if not raw:
            return raw

        candidates = [raw]
        if raw.startswith('"') and raw.endswith('"') and len(raw) > 1:
            candidates.append(raw[1:-1])

        if '""' in raw:
            candidates.append(raw.replace('""', '"'))
            if raw.startswith('"') and raw.endswith('"') and len(raw) > 1:
                candidates.append(raw[1:-1].replace('""', '"'))

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            if isinstance(parsed, str):
                inner = parsed.strip()
                if inner.startswith("{") and inner.endswith("}"):
                    try:
                        return json.loads(inner)
                    except Exception:
                        return parsed
            return parsed

        # Tidak bisa di-parse sebagai JSON → plain string (kemungkinan output LLM)
        return value

    def _format_llm_text_as_html(self, text: str) -> str:
        """
        Format output LLM yang kaya markdown menjadi HTML Telegram.
        Handle: **bold**, *italic*, bullet •/-/*, numbered list, link [text](url).
        """
        if not text:
            return ""

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 1. Konversi link markdown [text](url) → <a href="url">text</a>
        text = re.sub(
            r'\[([^\]]+)\]\((https?://[^\)]+)\)',
            lambda m: f'<a href="{m.group(2)}">{html.escape(m.group(1))}</a>',
            text
        )

        # 2. Proses baris per baris
        lines = text.split("\n")
        result = []

        for line in lines:
            stripped = line.strip()

            if not stripped:
                result.append("")
                continue

            # Bold+italic ***text*** atau __text__
            stripped = re.sub(r'\*\*\*(.+?)\*\*\*', lambda m: f'<b><i>{html.escape(m.group(1))}</i></b>', stripped)

            # Bold **text** atau __text__
            stripped = re.sub(r'\*\*(.+?)\*\*', lambda m: f'<b>{html.escape(m.group(1))}</b>', stripped)
            stripped = re.sub(r'__(.+?)__', lambda m: f'<b>{html.escape(m.group(1))}</b>', stripped)

            # Italic *text* atau _text_ (hati-hati tidak bentrok dengan bold)
            stripped = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', lambda m: f'<i>{html.escape(m.group(1))}</i>', stripped)
            stripped = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', lambda m: f'<i>{html.escape(m.group(1))}</i>', stripped)

            # Bullet: •, -, * di awal baris
            bullet_match = re.match(r'^[•\-\*]\s+(.+)$', stripped)
            if bullet_match:
                content = bullet_match.group(1)
                # escape sisa teks yang belum di-escape (bukan tag HTML)
                content = self._escape_non_tag(content)
                result.append(f"• {content}")
                continue

            # Numbered list: "1. " atau "1) "
            num_match = re.match(r'^(\d+)[\.]\s+(.+)$', stripped)
            if num_match:
                num = num_match.group(1)
                content = num_match.group(2)
                content = self._escape_non_tag(content)
                result.append(f"{num}. {content}")
                continue

            # Heading sederhana: baris diakhiri ":" dan pendek
            if stripped.endswith(":") and len(stripped) < 80 and not any(tag in stripped for tag in ["<b>", "<i>", "<a"]):
                result.append(f"<b>{html.escape(stripped)}</b>")
                continue

            # Plain text — escape karakter yang belum di-escape
            result.append(self._escape_non_tag(stripped))

        # Bersihkan baris kosong berturut > 2
        rendered = "\n".join(result)
        rendered = re.sub(r'\n{3,}', '\n\n', rendered)
        return rendered.strip()

    def _escape_non_tag(self, text: str) -> str:
        """
        Escape karakter HTML di teks, tapi jangan escape tag HTML yang sudah ada
        seperti <b>, <i>, <a href=...>.
        """
        # Split berdasarkan tag HTML yang sudah ada
        parts = re.split(r'(<[^>]+>)', text)
        escaped_parts = []
        for part in parts:
            if re.match(r'<[^>]+>', part):
                # Ini tag HTML, biarkan
                escaped_parts.append(part)
            else:
                escaped_parts.append(html.escape(part))
        return "".join(escaped_parts)

    def _format_text_as_html(self, text: str) -> str:
        """
        Format plain text dari dataset (bukan LLM) jadi HTML Telegram.
        Lebih sederhana — tidak perlu handle markdown.
        """
        if not text:
            return ""

        normalized_lines = [line.strip() for line in text.replace("\r\n", "\n").split("\n")]
        non_empty = [line for line in normalized_lines if line]

        if not non_empty:
            return ""

        formatted_lines = []
        heading_written = False

        for line in normalized_lines:
            if not line:
                formatted_lines.append("")
                continue

            if not heading_written and len(non_empty) > 1 and line.endswith(":"):
                formatted_lines.append(f"<b>{html.escape(line)}</b>")
                heading_written = True
                continue

            bullet_match = re.match(r"^(?:[a-zA-Z]|\d+)[\.\)]\s+(.*)$", line)
            if bullet_match:
                bullet_text = bullet_match.group(1).strip()
                formatted_lines.append(f"• {html.escape(bullet_text)}")
            else:
                formatted_lines.append(html.escape(line))

        rendered = "\n".join(formatted_lines).strip()
        return rendered or html.escape(text.strip())

    # ── Keyword heading section di response dataset ──────────────────────────
    _SECTION_KEYWORDS = re.compile(
        r"^(persyaratan|syarat\s*utama|syarat\s*dokumen|prosedur|langkah\s*\-?\s*langkah"
        r"|dokumen|keterangan|proses|biaya|tarif|waktu|catatan|informasi"
        r"|pihak yang terlibat|tujuan|ruang lingkup|tahap pelaksanaan"
        r"|dokumen pendamping|syarat)\b",
        re.IGNORECASE,
    )

    @staticmethod
    def _needs_rich_format(text: str) -> bool:
        """
        Deteksi otomatis apakah teks perlu diformat penuh.
        True jika mengandung:
          - pipe separator `|`
          - literal \\n
          - **markdown bold** (lebih dari satu token)
          - bullet * atau angka di awal baris
          - lebih dari 2 baris nyata
        """
        if "|" in text:
            return True
        if "\\n" in text:
            return True
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) > 2:
            return True
        bold_count = len(re.findall(r"\*\*", text))
        if bold_count >= 4:          # minimal 2 pasang **
            return True
        if re.search(r"(?m)^\s*[\*\-•]\s", text):  # bullet di awal baris
            return True
        return False

    def _format_dataset_response(self, raw: str) -> str:
        """
        Formatter universal untuk response dari dataset.

        - Kalimat sederhana (tidak ada pipe/\\n/markdown multi-item)
          → dikembalikan apa adanya (hanya HTML escape + inline markdown).
        - Teks kaya (ada pipe, \\n literal, bullet, bold multi-item)
          → dipecah per segmen, tiap segmen jadi bullet atau heading.
        """
        if not raw:
            return ""

        # Normalisasi \\n literal → newline nyata
        raw = raw.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")

        # Jika tidak perlu format kaya, kembalikan sebagai teks biasa
        if not self._needs_rich_format(raw):
            return self._apply_inline_markdown(raw.strip())

        # ── Mode format kaya ──

        # Lindungi URL agar pipe di dalamnya tidak ikut dipecah
        _url_stash: dict = {}

        def _stash_url(m: re.Match) -> str:
            key = f"\x00U{len(_url_stash)}\x00"
            # Potong trailing punctuation ) ] yang bukan bagian URL
            url_raw = m.group(0)
            url = re.sub(r'[\)\]]+$', '', url_raw)
            _url_stash[key] = url
            return key + url_raw[len(url):]

        raw = re.sub(r"https?://\S+", _stash_url, raw)

        # Pecah berdasarkan |
        segments = [s.strip() for s in raw.split("|")]

        # Restore URL di tiap segmen
        def _restore(s: str) -> str:
            for k, v in _url_stash.items():
                s = s.replace(k, v)
            return s

        segments = [_restore(s) for s in segments]

        # Pecah lagi segmen yang mengandung newline ATAU pola ** heading: ** di tengah
        expanded: list = []
        for seg in segments:
            # Pertama pecah per newline
            sub_segs = []
            if "\n" in seg:
                sub_segs = [s.strip() for s in seg.split("\n") if s.strip()]
            else:
                sub_segs = [seg] if seg else []

            # Kemudian pecah juga berdasarkan ** kata: ** heading yang tertanam
            # Contoh: "intro. **Persyaratan:** item * item * item **Prosedur:** item"
            for sub in sub_segs:
                # Pisahkan pada batas ** yang diikuti kata heading lalu **
                # Pattern: split sebelum ** (jika ada teks di depannya)
                parts_md = re.split(r'(?=\*\*[A-Za-z\s]{1,30}:\*\*)', sub)
                for p in parts_md:
                    p = p.strip()
                    if p:
                        # Dalam tiap bagian, pecah juga * item di tengah
                        # "**Persyaratan:** * item1 * item2" → heading + 2 bullet
                        inner_split = re.split(r'\s\*\s+', p)
                        if len(inner_split) > 1:
                            expanded.append(inner_split[0].strip())
                            for item in inner_split[1:]:
                                item = item.strip()
                                if item:
                                    expanded.append(item)
                        else:
                            expanded.append(p)

        # Render tiap segmen
        result_lines: list = []
        prev_is_heading = False
        for seg in expanded:
            if not seg:
                continue

            # Bersihkan bullet teks yang sudah ada (* atau • di awal/akhir)
            seg = re.sub(r"^[\*\-•]\s+", "", seg)   # strip bullet di awal
            seg = re.sub(r"\s+\*$", "", seg)          # strip asterisk sisa di akhir
            seg = seg.strip()

            # Inline markdown → HTML
            rendered = self._apply_inline_markdown(seg)

            # Cek heading: periksa plain text (tanpa tag HTML)
            plain = re.sub(r'<[^>]+>', '', rendered).strip()
            is_heading = plain.endswith(":") or bool(self._SECTION_KEYWORDS.match(plain))

            # Heading yang panjang (>60 char) tidak dijadikan heading
            if is_heading and len(plain) > 60:
                is_heading = False

            if is_heading:
                # Jika heading punya ekor teks setelah ':' → pisah heading + bullet
                # Contoh: "**Prosedur:** Pemohon mengisi" → "Prosedur:" + "Pemohon mengisi"
                colon_idx = plain.find(":")
                if colon_idx != -1 and colon_idx < len(plain) - 1:
                    heading_text = plain[:colon_idx + 1].strip()
                    tail_text = plain[colon_idx + 1:].strip()
                    if result_lines and not prev_is_heading:
                        result_lines.append("")
                    result_lines.append(f"<b>{html.escape(heading_text)}</b>")
                    if tail_text:
                        result_lines.append(f"\u2022 {html.escape(tail_text)}")
                else:
                    if result_lines and not prev_is_heading:
                        result_lines.append("")
                    result_lines.append(f"<b>{html.escape(plain)}</b>")
            else:
                result_lines.append(f"\u2022 {rendered}")

            prev_is_heading = is_heading

        rendered_final = "\n".join(result_lines).strip()
        return re.sub(r"\n{3,}", "\n\n", rendered_final)

    def _apply_inline_markdown(self, text: str) -> str:
        """Konversi **bold** *italic* URL telanjang → HTML Telegram-safe."""
        if not text:
            return ""

        # Lindungi URL sebelum escape
        _urls: dict = {}

        def _stash(m: re.Match) -> str:
            key = f"\x01L{len(_urls)}\x01"
            # Potong trailing punctuation ) ] , . yang bukan bagian URL
            url_raw = m.group(0)
            url = re.sub(r'[\)\]\.\,]+$', '', url_raw)
            _urls[key] = url
            return key + url_raw[len(url):]

        text = re.sub(r"https?://\S+", _stash, text)

        # Pecah per token markdown (greedy terpendek)
        parts = re.split(r"(\*\*\*.*?\*\*\*|\*\*.*?\*\*|__.*?__|(?<!\*)\*(?!\*).*?(?<!\*)\*(?!\*))", text)
        out = []
        for part in parts:
            if re.fullmatch(r"\*\*\*(.+?)\*\*\*", part, re.DOTALL):
                out.append(f"<b><i>{html.escape(part[3:-3])}</i></b>")
            elif re.fullmatch(r"\*\*(.+?)\*\*", part, re.DOTALL):
                out.append(f"<b>{html.escape(part[2:-2])}</b>")
            elif re.fullmatch(r"__(.+?)__", part, re.DOTALL):
                out.append(f"<b>{html.escape(part[2:-2])}</b>")
            elif re.fullmatch(r"\*(.+?)\*", part, re.DOTALL):
                # Hanya jadikan italic jika isinya bukan URL/simbol murni
                inner = part[1:-1]
                if re.search(r"[a-zA-Z]", inner):
                    out.append(f"<i>{html.escape(inner)}</i>")
                else:
                    out.append(html.escape(part))
            else:
                out.append(html.escape(part))

        result = "".join(out)

        # Restore URL → tautan yang bisa diklik
        for key, url in _urls.items():
            safe = html.escape(url)
            display = html.escape(url if len(url) <= 55 else url[:52] + "\u2026")
            result = result.replace(html.escape(key), f'<a href="{safe}">{display}</a>')

        return result

    def _render_json_response_html(self, data: Dict[str, Any]) -> str:
        r_type = str(data.get("type", "text")).strip().lower()

        if r_type == "text":
            body = str(data.get("body", "")).strip()
            return self._format_dataset_response(body)

        if r_type == "list":
            title = str(data.get("title", "")).strip()
            items = data.get("items", data.get("body", []))

            lines = []
            if title:
                lines.append(self._format_dataset_response(title))

            if isinstance(items, list):
                for item in items:
                    item_text = str(item).strip()
                    if item_text:
                        lines.append(f"\u2022 {self._apply_inline_markdown(item_text)}")
            else:
                item_text = str(items).strip()
                if item_text:
                    lines.append(f"\u2022 {self._apply_inline_markdown(item_text)}")

            return "\n".join(lines).strip() or self._format_dataset_response(str(data))

        body_fallback = str(data.get("body", "")).strip()
        if body_fallback:
            return self._format_dataset_response(body_fallback)
        return self._format_dataset_response(str(data))

    async def _send_response(self, update: Update, result: Dict[str, Any]):
        api_response_raw = result.get("response", "")
        api_response = self._try_parse_response_payload(api_response_raw)

        # ── Tentukan apakah ini output LLM atau dataset biasa ──
        is_llm_augmented = result.get("augmented", False)

        if isinstance(api_response, dict):
            # Response dari dataset (JSON format)
            text = self._render_json_response_html(api_response)
        else:
            # Semua string response — termasuk LLM dan plain dataset
            # _format_dataset_response mendeteksi otomatis apakah perlu format kaya
            text = self._format_dataset_response(str(api_response))

        if not text:
            text = "Maaf, belum ada jawaban yang bisa ditampilkan."

        # Split kalau melebihi limit Telegram
        MAX_LEN = self.config['max_message_length'] - 100  # buffer
        messages = self._split_message(text, MAX_LEN)

        # Build keyboard dari options (hanya untuk pesan pertama)
        reply_markup = None
        options = result.get("options", [])
        if options and isinstance(options, list):
            keyboard = []
            for opt in options:
                label = opt.get("label", "Option")
                keyboard.append([InlineKeyboardButton(label, callback_data=label)])
            reply_markup = InlineKeyboardMarkup(keyboard)

        for i, msg in enumerate(messages):
            try:
                markup = reply_markup if i == 0 else None
                await update.effective_message.reply_text(
                    msg,
                    reply_markup=markup,
                    parse_mode="HTML"
                )
            except Exception as e:
                logger.error(f"Send HTML failed: {e}, trying plain text")
                try:
                    # Fallback: strip semua HTML tag, kirim plain
                    plain = re.sub(r'<[^>]+>', '', msg)
                    await update.effective_message.reply_text(plain)
                except Exception as e2:
                    logger.error(f"Send plain also failed: {e2}")

    def _split_message(self, text: str, max_len: int) -> list:
        """Split pesan panjang di batas baris agar tidak terpotong di tengah kata."""
        if len(text) <= max_len:
            return [text]

        parts = []
        while len(text) > max_len:
            # Cari newline terdekat sebelum batas
            cut = text.rfind("\n", 0, max_len)
            if cut == -1:
                cut = max_len
            parts.append(text[:cut].strip())
            text = text[cut:].strip()

        if text:
            parts.append(text)

        return parts

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
            f"(conf: {result.get('confidence', 0):.3f}, "
            f"augmented: {result.get('augmented', False)})"
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
def create_telegram_bot(token=None, api_url=None, skip_api_test=False):
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Missing Telegram Bot Token.")
    api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    return TelegramBot(token, api_url, skip_api_test)


# MAIN
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Telegram Bot")
    parser.add_argument("--skip-api-test", action="store_true",
                        help="Skip API connectivity test")
    parser.add_argument("--api-url", default=None,
                        help="API base URL (default: http://localhost:8000)")

    args = parser.parse_args()

    bot = None
    try:
        bot = create_telegram_bot(api_url=args.api_url, skip_api_test=args.skip_api_test)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("⛔ CTRL+C detected, stopping...")
    finally:
        if bot:
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())