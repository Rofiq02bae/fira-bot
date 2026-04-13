"""
telegram_bot.py — Hybrid Telegram Bot
======================================
Interaction model:
  1. Reply Keyboard (rule-based navigation)
  2. Free-text input  (NLP / AI classification)

State machine:
  MAIN_MENU  ──► CATEGORY_MENU ──► INTENT_MENU
                   (Lainnya)         (select)

Author: refactored for Fira Bot
"""

import os
import json
import html
import re
import logging
import asyncio
import aiohttp
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    KeyboardButton,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

# States
STATE_MAIN = "MAIN_MENU"
STATE_CATEGORY = "CATEGORY_MENU"
STATE_INTENT = "INTENT_MENU"

# Main menu button labels (always shown)
MAIN_BUTTONS = ["🪪 KTP", "👨‍👩‍👧 KK", "📄 Akta Kelahiran", "🤖 Bot Info", "📂 Lainnya"]

# Shortcut buttons → natural-language query for /api/chat
SHORTCUT_CHAT_MAP: Dict[str, str] = {
    "🪪 KTP":            "cara membuat KTP",
    "👨‍👩‍👧 KK":            "cara membuat kartu keluarga",
    "📄 Akta Kelahiran": "cara membuat akta kelahiran",
    "🤖 Bot Info":       "siapa kamu",
}

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> dict:
    """Load a JSON file relative to this file's directory."""
    full = os.path.join(os.path.dirname(__file__), path)
    try:
        with open(full, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Cannot load {path}: {e}")
        return {}


def format_intent_label(intent_key: str) -> str:
    """
    Convert snake_case intent key → human-readable button label.

    Examples:
        "ktp_info"          → "Cara Membuat KTP"
        "akta_lahir_info"   → "Cara Membuat Akta Lahir"
        "sop_backup_data"   → "SOP Backup Data"
        "pindah_general"    → "Pindah General"
    """
    parts = intent_key.replace("-", "_").split("_")

    UPPER_TOKENS = {"sop", "ktp", "kk", "nik", "nib", "lkpm", "splp",
                    "jipd", "ippd", "api", "ak1", "itr", "uak", "uat"}

    formatted = []
    for p in parts:
        if p.lower() in UPPER_TOKENS:
            formatted.append(p.upper())
        else:
            formatted.append(p.capitalize())

    # Intent yang berkaitan dengan pembuatan / pendaftaran dokumen
    MAKEABLE = {
        "ktp", "kk", "akta", "lahir", "mati", "mati",
        "pindah", "surat", "nib", "ak1", "loakk", "itr",
    }

    # Replace trailing "Info" with prefix
    if formatted and formatted[-1] == "Info":
        # Cek apakah ada token dokumen dalam label
        lower_parts = [p.lower() for p in formatted[:-1]]
        is_makeable = any(tok in lower_parts for tok in MAKEABLE)
        prefix = "Cara Membuat" if is_makeable else "Info"
        formatted = [prefix] + formatted[:-1]

    return " ".join(formatted)


def build_reply_keyboard(
    buttons: List[str],
    columns: int = 2,
    resize: bool = True,
    placeholder: str = "Pilih menu...",
) -> ReplyKeyboardMarkup:
    """Build a ReplyKeyboardMarkup from a flat list of labels."""
    rows = [buttons[i : i + columns] for i in range(0, len(buttons), columns)]
    keyboard = [[KeyboardButton(label) for label in row] for row in rows]
    return ReplyKeyboardMarkup(
        keyboard,
        resize_keyboard=resize,
        input_field_placeholder=placeholder,
    )


# ──────────────────────────────────────────────────────────────────────────────
# RESPONSE FORMATTER  (universal — same logic as before, kept self-contained)
# ──────────────────────────────────────────────────────────────────────────────

_SECTION_KW = re.compile(
    r"^(persyaratan|syarat(\s*(utama|dokumen))?|prosedur|langkah(\s*-?\s*langkah)?"
    r"|dokumen|keterangan|proses|biaya|tarif|waktu|catatan|informasi"
    r"|pihak yang terlibat|tujuan|ruang lingkup|tahap pelaksanaan"
    r"|dokumen pendamping|syarat)\b",
    re.IGNORECASE,
)


def _needs_rich(text: str) -> bool:
    if "|" in text or "\\n" in text:
        return True
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) > 2:
        return True
    if len(re.findall(r"\*\*", text)) >= 4:
        return True
    if re.search(r"(?m)^\s*[\*\-•]\s", text):
        return True
    return False


def _stash_urls(text: str) -> tuple[str, dict]:
    """Replace URLs with placeholders to protect them during processing."""
    stash: dict = {}

    def _replace(m: re.Match) -> str:
        key = f"\x00U{len(stash)}\x00"
        raw = m.group(0)
        url = re.sub(r"[\)\]\.]+$", "", raw)
        stash[key] = url
        return key + raw[len(url):]

    return re.sub(r"https?://\S+", _replace, text), stash


def _restore_urls(text: str, stash: dict) -> str:
    for key, url in stash.items():
        safe = html.escape(url)
        display = html.escape(url if len(url) <= 55 else url[:52] + "…")
        text = text.replace(html.escape(key), f'<a href="{safe}">{display}</a>')
    return text


def apply_inline_markdown(text: str) -> str:
    """Convert **bold** *italic* and bare URLs → Telegram HTML."""
    if not text:
        return ""

    text, url_stash = _stash_urls(text)

    parts = re.split(
        r"(\*\*\*.*?\*\*\*|\*\*.*?\*\*|__.*?__|(?<!\*)\*(?!\*).*?(?<!\*)\*(?!\*))",
        text,
    )
    out = []
    for part in parts:
        if re.fullmatch(r"\*\*\*(.+?)\*\*\*", part, re.DOTALL):
            out.append(f"<b><i>{html.escape(part[3:-3])}</i></b>")
        elif re.fullmatch(r"\*\*(.+?)\*\*", part, re.DOTALL):
            out.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif re.fullmatch(r"__(.+?)__", part, re.DOTALL):
            out.append(f"<b>{html.escape(part[2:-2])}</b>")
        elif re.fullmatch(r"\*(.+?)\*", part, re.DOTALL):
            inner = part[1:-1]
            if re.search(r"[a-zA-Z]", inner):
                out.append(f"<i>{html.escape(inner)}</i>")
            else:
                out.append(html.escape(part))
        else:
            out.append(html.escape(part))

    result = "".join(out)
    result = _restore_urls(result, url_stash)
    return result


def format_response(raw: str) -> str:
    """
    Universal response formatter.

    - Simple sentence  → plain escaped text (+ inline markdown)
    - Rich text (pipe separators, \\n, bullet asterisks, multi-bold)
      → structured bullets with auto-detected section headings
    """
    if not raw:
        return ""

    # Normalise literal \\n
    raw = raw.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")

    if not _needs_rich(raw):
        return apply_inline_markdown(raw.strip())

    # ── Rich mode ──
    raw, url_stash = _stash_urls(raw)

    def restore(s: str) -> str:
        for k, v in url_stash.items():
            s = s.replace(k, v)
        return s

    # Split on |
    segments = [restore(s.strip()) for s in raw.split("|")]

    # Expand per-newline, then split on embedded **Heading:** markers
    expanded: List[str] = []
    for seg in segments:
        sub_segs = [s.strip() for s in seg.split("\n") if s.strip()] if "\n" in seg else ([seg] if seg else [])
        for sub in sub_segs:
            # Split on embedded ** heading patterns mid-sentence
            parts = re.split(r"(?=\*\*[A-Za-zÀ-ÿ\s]{1,35}:\*\*)", sub)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Split "* item * item" style
                inner = re.split(r"\s\*\s+", p)
                if len(inner) > 1:
                    expanded.append(inner[0].strip())
                    expanded.extend(i.strip() for i in inner[1:] if i.strip())
                else:
                    expanded.append(p)

    # Render each segment
    lines: List[str] = []
    prev_heading = False

    for seg in expanded:
        if not seg:
            continue

        # Strip leading/trailing bullet markers
        seg = re.sub(r"^[\*\-•]\s+", "", seg)
        seg = re.sub(r"\s+\*$", "", seg).strip()
        if not seg:
            continue

        rendered = apply_inline_markdown(seg)
        plain = re.sub(r"<[^>]+>", "", rendered).strip()

        is_heading = (plain.endswith(":") or bool(_SECTION_KW.match(plain))) and len(plain) <= 60

        if is_heading:
            colon = plain.find(":")
            if colon != -1 and colon < len(plain) - 1:
                heading = plain[: colon + 1].strip()
                tail = plain[colon + 1 :].strip()
                if lines and not prev_heading:
                    lines.append("")
                lines.append(f"<b>{html.escape(heading)}</b>")
                if tail:
                    lines.append(f"• {html.escape(tail)}")
            else:
                if lines and not prev_heading:
                    lines.append("")
                lines.append(f"<b>{html.escape(plain)}</b>")
        else:
            lines.append(f"• {rendered}")

        prev_heading = is_heading

    result = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def format_json_response(data: dict) -> str:
    """Render a structured JSON response (type: text | list) to HTML."""
    r_type = str(data.get("type", "text")).strip().lower()

    if r_type == "text":
        return format_response(str(data.get("body", "")).strip())

    if r_type == "list":
        title = str(data.get("title", "")).strip()
        items = data.get("items", data.get("body", []))
        lines = []
        if title:
            lines.append(format_response(title))
        if isinstance(items, list):
            lines.extend(f"• {apply_inline_markdown(str(i).strip())}" for i in items if str(i).strip())
        else:
            item_text = str(items).strip()
            if item_text:
                lines.append(f"• {apply_inline_markdown(item_text)}")
        return "\n".join(lines).strip() or format_response(str(data))

    fallback = str(data.get("body", "")).strip()
    return format_response(fallback or str(data))


# ──────────────────────────────────────────────────────────────────────────────
# JSON PAYLOAD PARSER
# ──────────────────────────────────────────────────────────────────────────────

def try_parse_payload(value: Any) -> Any:
    """Try to parse value as JSON (handles escaped CSV-style double-quotes)."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return value

    raw = value.strip()
    candidates = [raw]
    if raw.startswith('"') and raw.endswith('"') and len(raw) > 1:
        candidates.append(raw[1:-1])
    if '""' in raw:
        candidates.append(raw.replace('""', '"'))
        if raw.startswith('"') and raw.endswith('"') and len(raw) > 1:
            candidates.append(raw[1:-1].replace('""', '"'))

    for c in candidates:
        try:
            parsed = json.loads(c)
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

    return value


# ──────────────────────────────────────────────────────────────────────────────
# NAVIGATION MENUS
# ──────────────────────────────────────────────────────────────────────────────

class MenuNavigator:
    """
    Encapsulates all menu rendering and state logic.
    Keeps TelegramBot clean by delegating navigation here.
    """

    def __init__(self, categories: dict):
        # All categories (from categories.json / dictionary.json)
        self.categories: dict = categories

        # Categories shown as buttons in CATEGORY_MENU
        # (exclude internal/fallback ones users shouldn't browse directly)
        self._HIDDEN_CATEGORIES = {"Fallback"}
        self.browsable: List[str] = [
            k for k in categories if k not in self._HIDDEN_CATEGORIES
        ]

        # Build reverse map: intent_key → category name
        self._intent_to_category: Dict[str, str] = {}
        for cat, intents in categories.items():
            for intent in intents:
                self._intent_to_category[intent] = cat

        # Build intent label → key map (for button presses)
        self._label_to_intent: Dict[str, str] = {}
        for intents in categories.values():
            for intent in intents:
                label = format_intent_label(intent)
                self._label_to_intent[label] = intent

    # ── Keyboards ──

    def main_keyboard(self) -> ReplyKeyboardMarkup:
        return build_reply_keyboard(MAIN_BUTTONS, columns=2, placeholder="Ketik atau pilih menu")

    def category_keyboard(self) -> ReplyKeyboardMarkup:
        cats = self.browsable + ["🏠 Menu Utama"]
        return build_reply_keyboard(cats, columns=2, placeholder="Pilih kategori...")

    def intent_keyboard(self, category: str) -> ReplyKeyboardMarkup:
        intents = self.categories.get(category, [])
        labels = [format_intent_label(i) for i in intents]
        labels.append("◀ Kembali")
        return build_reply_keyboard(labels, columns=1, placeholder="Pilih topik...")

    # ── Resolvers ──

    def resolve_button(self, text: str) -> Optional[str]:
        """
        Map a button press to an intent key.
        Returns None if not a known button.
        """
        # Direct shortcut (main menu)
        if text in SHORTCUT_INTENT_MAP:
            return SHORTCUT_INTENT_MAP[text]
        # Intent label button
        return self._label_to_intent.get(text)

    def is_category(self, text: str) -> bool:
        return text in self.categories

    def category_for_intent(self, intent: str) -> Optional[str]:
        return self._intent_to_category.get(intent)


# ──────────────────────────────────────────────────────────────────────────────
# USER STATE MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class UserStateManager:
    """Simple in-memory state per user."""

    def __init__(self):
        self._states: Dict[int, str] = {}
        self._data: Dict[int, dict] = {}   # extra per-user data (e.g. active category)

    def get_state(self, user_id: int) -> str:
        return self._states.get(user_id, STATE_MAIN)

    def set_state(self, user_id: int, state: str, **data):
        self._states[user_id] = state
        if data:
            self._data.setdefault(user_id, {}).update(data)

    def get_data(self, user_id: int) -> dict:
        return self._data.get(user_id, {})

    def reset(self, user_id: int):
        self._states.pop(user_id, None)
        self._data.pop(user_id, None)


# ──────────────────────────────────────────────────────────────────────────────
# TELEGRAM BOT
# ──────────────────────────────────────────────────────────────────────────────

class TelegramBot:
    def __init__(
        self,
        token: str,
        api_base_url: str = "http://localhost:8000",
        skip_api_test: bool = False,
    ):
        self.token = token
        self.api_base_url = api_base_url.rstrip("/")
        self.skip_api_test = skip_api_test
        self.application: Optional[Application] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Load categories
        categories = _load_json("categories.json")
        self.nav = MenuNavigator(categories)
        self.state_mgr = UserStateManager()

        self.config = {
            "max_message_length": 4096,
            "typing_delay":       0.0,   # dinonaktifkan — menambah latensi tanpa manfaat
            "api_timeout":        90,    # max wait untuk NLU inference
            "health_check_timeout": 5,   # health check pakai timeout singkat
            "welcome_message": (
                "👋 Halo <b>{name}</b>! Saya <b>Firaa</b>, asisten virtual "
                "Diskominfo Kabupaten Tegal.\n\n"
                "Pilih menu di bawah atau ketik pertanyaan langsung ya! 😊"
            ),
            "help_message": (
                "🆘 <b>Bantuan</b>\n\n"
                "/start  — Mulai bot\n"
                "/menu   — Tampilkan menu utama\n"
                "/help   — Bantuan\n"
                "/status — Status sistem"
            ),
            "fallback_message": (
                "😅 Maaf, saya belum cukup paham maksud pesan tersebut.\n"
                "Coba pilih menu atau tanya dengan kalimat lain ya!"
            ),
        }

    # ──────────────────────────────────────────────
    # INITIALIZATION
    # ──────────────────────────────────────────────

    async def initialize(self):
        logger.info("🔄 Initializing bot...")

        # Session utama untuk chat API (timeout panjang = NLU inference bisa lambat)
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                connect=5,
                total=self.config["api_timeout"],
            )
        )

        if not self.skip_api_test:
            try:
                await self._test_api_connection()
            except Exception as e:
                logger.warning(f"API not ready at startup (will retry on first message): {e}")

        try:
            self.application = Application.builder().token(self.token).build()
            info = await self.application.bot.get_me()
            logger.info(f"🔑 Token OK → @{info.username}")
        except Exception as e:
            raise ValueError(f"Invalid Telegram token: {e}")

        self._register_handlers()
        logger.info("✅ Initialization complete")

    async def _test_api_connection(self):
        """Quick health-check dengan timeout singkat agar startup tidak lambat."""
        hc_timeout = aiohttp.ClientTimeout(connect=3, total=self.config["health_check_timeout"])
        async with aiohttp.ClientSession(timeout=hc_timeout) as hc:
            async with hc.get(f"{self.api_base_url}/health") as r:
                if r.status != 200:
                    raise Exception(f"API status {r.status}")
                data = await r.json()
                logger.info(f"✅ API Ready: {data}")

    # ──────────────────────────────────────────────
    # HANDLER REGISTRATION
    # ──────────────────────────────────────────────

    def _register_handlers(self):
        app = self.application
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("menu",  self._cmd_menu))
        app.add_handler(CommandHandler("help",  self._cmd_help))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("intents", self._cmd_intents))

        app.add_handler(MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            self._handle_message,
        ))
        app.add_handler(CallbackQueryHandler(self._handle_callback))

    # ──────────────────────────────────────────────
    # COMMAND HANDLERS
    # ──────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        self.state_mgr.reset(user.id)
        await self._typing(update)
        await update.message.reply_text(
            self.config["welcome_message"].format(name=html.escape(user.first_name)),
            parse_mode="HTML",
            reply_markup=self.nav.main_keyboard(),
        )

    async def _cmd_menu(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        self.state_mgr.set_state(user.id, STATE_MAIN)
        await self._typing(update)
        await update.message.reply_text(
            "🏠 <b>Menu Utama</b>",
            parse_mode="HTML",
            reply_markup=self.nav.main_keyboard(),
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        await update.message.reply_text(self.config["help_message"], parse_mode="HTML")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        try:
            data = await self._http_get("/health")
            msg = (
                "🖥️ <b>Status Sistem</b>\n\n"
                f"API: {'✅ Online' if data else '❌ Error'}\n"
                f"Model: {'✅ Ready' if data.get('model_loaded') else '⚠ Offline'}"
            )
        except Exception:
            msg = "❌ Tidak bisa menghubungi server."
        await update.message.reply_text(msg, parse_mode="HTML")

    async def _cmd_intents(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await self._typing(update)
        try:
            data = await self._http_get("/intents")
            if isinstance(data, dict):
                intents = (
                    list(data["available_intents"].keys())
                    if isinstance(data.get("available_intents"), dict)
                    else data.get("available_intents", list(data.keys()))
                )
            elif isinstance(data, list):
                intents = data
            else:
                intents = []

            if not intents:
                return await update.message.reply_text("Tidak ada intent tersedia.")

            shown = sorted(intents)[:50]
            lines = [f"<b>Intent ({len(intents)} total):</b>"] + [
                f"• {html.escape(str(i))}" for i in shown
            ]
            if len(intents) > 50:
                lines.append(f"\n<i>... dan {len(intents)-50} intent lainnya</i>")

            await update.message.reply_text("\n".join(lines), parse_mode="HTML")
        except Exception as e:
            logger.error(f"_intents error: {e}")
            await update.message.reply_text("❌ Gagal mengambil daftar intent.")

    # ──────────────────────────────────────────────
    # MAIN MESSAGE HANDLER
    # ──────────────────────────────────────────────

    async def _handle_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        if not update.message or not update.message.text:
            await update.message.reply_text("⚠ Saya hanya bisa membaca pesan teks 😊")
            return

        text = update.message.text.strip()
        user = update.effective_user
        user_id = user.id
        state = self.state_mgr.get_state(user_id)

        logger.info(f"💬 [{state}] {user.first_name}: {text}")
        await self._typing(update)

        # ── Navigation shortcuts ──────────────────

        # "Menu Utama" → always go home
        if text in ("🏠 Menu Utama", "/menu"):
            self.state_mgr.set_state(user_id, STATE_MAIN)
            return await self._show_main_menu(update)

        # "Kembali" → go back one level
        if text == "◀ Kembali":
            return await self._go_back(update, user_id, state)

        # ── State-aware routing ───────────────────

        if state == STATE_MAIN:
            await self._handle_main_state(update, user_id, text)

        elif state == STATE_CATEGORY:
            await self._handle_category_state(update, user_id, text)

        elif state == STATE_INTENT:
            await self._handle_intent_state(update, user_id, text)

        else:
            # Unknown state → reset
            self.state_mgr.reset(user_id)
            await self._show_main_menu(update)

    # ──────────────────────────────────────────────
    # STATE HANDLERS
    # ──────────────────────────────────────────────

    async def _handle_main_state(self, update: Update, user_id: int, text: str):
        # Shortcut button → kirim query natural ke /api/chat
        if text in SHORTCUT_CHAT_MAP:
            query = SHORTCUT_CHAT_MAP[text]
            return await self._query_and_reply(update, free_text=query)

        # "Lainnya" → show category browser
        if text == "📂 Lainnya":
            self.state_mgr.set_state(user_id, STATE_CATEGORY)
            return await self._show_categories(update)

        # Free text → NLP
        await self._query_and_reply(update, free_text=text)

    async def _handle_category_state(self, update: Update, user_id: int, text: str):
        if self.nav.is_category(text):
            self.state_mgr.set_state(user_id, STATE_INTENT, active_category=text)
            return await self._show_intents(update, text)

        # Free text in category menu → NLP
        await self._query_and_reply(update, free_text=text)

    async def _handle_intent_state(self, update: Update, user_id: int, text: str):
        # Kirim label tombol langsung ke /api/chat sebagai free text
        # Contoh: "Cara Membuat KTP" dikirim sebagai query, NLP yang memproses
        await self._query_and_reply(update, free_text=text)

    # ──────────────────────────────────────────────
    # MENU DISPLAY HELPERS
    # ──────────────────────────────────────────────

    async def _show_main_menu(self, update: Update):
        await update.effective_message.reply_text(
            "🏠 <b>Menu Utama</b>\nPilih layanan atau ketik pertanyaan:",
            parse_mode="HTML",
            reply_markup=self.nav.main_keyboard(),
        )

    async def _show_categories(self, update: Update):
        lines = ["📂 <b>Pilih Kategori Layanan</b>\n"]
        for i, cat in enumerate(self.nav.browsable, 1):
            lines.append(f"{i}. {html.escape(cat)}")
        await update.effective_message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=self.nav.category_keyboard(),
        )

    async def _show_intents(self, update: Update, category: str):
        intents = self.nav.categories.get(category, [])
        labels = [format_intent_label(i) for i in intents]
        lines = [f"📋 <b>{html.escape(category)}</b>\nPilih topik:\n"]
        lines += [f"• {html.escape(l)}" for l in labels]
        await update.effective_message.reply_text(
            "\n".join(lines),
            parse_mode="HTML",
            reply_markup=self.nav.intent_keyboard(category),
        )

    async def _go_back(self, update: Update, user_id: int, current_state: str):
        if current_state == STATE_INTENT:
            self.state_mgr.set_state(user_id, STATE_CATEGORY)
            await self._show_categories(update)
        else:
            self.state_mgr.set_state(user_id, STATE_MAIN)
            await self._show_main_menu(update)

    # ──────────────────────────────────────────────
    # API CALL + RESPONSE DISPATCH
    # ──────────────────────────────────────────────

    async def _query_and_reply(
        self,
        update: Update,
        *,
        free_text: str,
    ):
        """
        Kirim free_text ke /api/chat dan tampilkan response yang sudah diformat.
        Semua jalur (tombol shortcut, tombol intent, teks bebas) melewati sini.
        """
        try:
            result = await self._call_chat_api(free_text)

            # Low-confidence fallback
            if result.get("confidence", 1.0) < 0.25:
                await update.effective_message.reply_text(
                    self.config["fallback_message"],
                    reply_markup=self.nav.main_keyboard(),
                )
                return

            await self._send_formatted_response(update, result)

        except asyncio.TimeoutError:
            await update.effective_message.reply_text("⏰ Request timeout. Coba lagi ya.")
        except Exception as e:
            logger.error(f"_query_and_reply error: {e}", exc_info=True)
            await update.effective_message.reply_text("❌ Terjadi kesalahan sistem.")

    async def _call_chat_api(self, message: str) -> dict:
        import time
        t0 = time.perf_counter()
        logger.info(f"📤 POST /api/chat → {message!r}")
        try:
            result = await self._http_post("/api/chat", {"text": message})
            elapsed = time.perf_counter() - t0
            intent  = result.get("intent", "?")
            conf    = result.get("confidence", 0)
            logger.info(f"📥 /api/chat replied in {elapsed:.2f}s → intent={intent} conf={conf:.3f}")
            return result
        except asyncio.TimeoutError:
            logger.error(f"⏰ /api/chat TIMEOUT after {self.config['api_timeout']}s for: {message!r}")
            raise
        except Exception as e:
            logger.error(f"❌ /api/chat ERROR: {e}", exc_info=True)
            return {"success": False, "response": "❌ Layanan tidak tersedia.", "confidence": 0.0}

    async def _send_formatted_response(self, update: Update, result: dict):
        raw = result.get("response", "")
        payload = try_parse_payload(raw)

        if isinstance(payload, dict):
            text = format_json_response(payload)
        else:
            text = format_response(str(payload))

        if not text:
            text = "Maaf, belum ada jawaban yang bisa ditampilkan."

        # Split if over Telegram limit
        max_len = self.config["max_message_length"] - 100
        chunks = self._split_message(text, max_len)

        # InlineKeyboard for any options
        markup = None
        options = result.get("options", [])
        if options and isinstance(options, list):
            keyboard = [
                [InlineKeyboardButton(o.get("label", "?"), callback_data=o.get("label", "?"))]
                for o in options
            ]
            markup = InlineKeyboardMarkup(keyboard)

        for i, chunk in enumerate(chunks):
            try:
                await update.effective_message.reply_text(
                    chunk,
                    parse_mode="HTML",
                    reply_markup=markup if i == 0 else None,
                )
            except Exception as e:
                logger.error(f"HTML send failed: {e} — falling back to plain text")
                plain = re.sub(r"<[^>]+>", "", chunk)
                try:
                    await update.effective_message.reply_text(plain)
                except Exception as e2:
                    logger.error(f"Plain send also failed: {e2}")

    # ──────────────────────────────────────────────
    # CALLBACK QUERY (InlineKeyboard)
    # ──────────────────────────────────────────────

    async def _handle_callback(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        await query.answer()
        text = query.data
        user = update.effective_user
        logger.info(f"🔄 Callback {user.first_name}: {text}")

        try:
            await self._typing(update)
            await self._query_and_reply(update, free_text=text)
        except Exception as e:
            logger.error(f"Callback error: {e}")
            await query.message.reply_text("❌ Terjadi kesalahan.")

    # ──────────────────────────────────────────────
    # HTTP UTILS
    # ──────────────────────────────────────────────

    async def _http_get(self, path: str) -> dict:
        async with self.session.get(f"{self.api_base_url}{path}") as r:
            return await r.json()

    async def _http_post(self, path: str, payload: dict) -> dict:
        async with self.session.post(
            f"{self.api_base_url}{path}", json=payload
        ) as r:
            if r.status == 200:
                return await r.json()
            if r.status == 503:
                return {"success": False, "response": "❌ Layanan tidak tersedia.", "confidence": 0.0}
            return {"success": False, "response": "❌ Terjadi kesalahan.", "confidence": 0.0}

    # ──────────────────────────────────────────────
    # UTILITIES
    # ──────────────────────────────────────────────

    async def _typing(self, update: Update):
        try:
            await update.effective_chat.send_action("typing")
            # typing_delay = 0 → tidak ada penundaan tambahan
            if self.config.get("typing_delay", 0) > 0:
                await asyncio.sleep(self.config["typing_delay"])
        except Exception:
            pass

    @staticmethod
    def _split_message(text: str, max_len: int) -> List[str]:
        if len(text) <= max_len:
            return [text]
        parts = []
        while len(text) > max_len:
            cut = text.rfind("\n", 0, max_len)
            if cut == -1:
                cut = max_len
            parts.append(text[:cut].strip())
            text = text[cut:].strip()
        if text:
            parts.append(text)
        return parts

    # ──────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────

    async def start(self):
        if not self.application:
            await self.initialize()

        logger.info("🚀 Starting bot polling...")
        await self.application.initialize()
        await self.application.start()
        await self.application.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        logger.info("✅ Bot is running!")
        await asyncio.Event().wait()   # run forever

    async def stop(self):
        logger.info("🛑 Stopping bot...")
        try:
            if self.application:
                if self.application.updater:
                    try:
                        await self.application.updater.stop()
                    except RuntimeError as e:
                        # Idempotent shutdown: updater may not be running yet/already stopped
                        if "not running" in str(e).lower():
                            logger.debug("Updater already stopped or not running.")
                        else:
                            raise

                try:
                    await self.application.stop()
                except RuntimeError as e:
                    if "not running" in str(e).lower():
                        logger.debug("Application already stopped or not running.")
                    else:
                        raise

                await self.application.shutdown()
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
        logger.info("✅ Bot stopped")


# ──────────────────────────────────────────────────────────────────────────────
# FACTORY & ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def create_telegram_bot(
    token: Optional[str] = None,
    api_url: Optional[str] = None,
    skip_api_test: bool = False,
) -> TelegramBot:
    token = token or os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("Missing TELEGRAM_BOT_TOKEN")
    api_url = api_url or os.getenv("API_BASE_URL", "http://localhost:8000")
    return TelegramBot(token, api_url, skip_api_test)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fira Telegram Bot")
    parser.add_argument("--skip-api-test", action="store_true")
    parser.add_argument("--api-url", default=None)
    args = parser.parse_args()

    bot = None
    try:
        bot = create_telegram_bot(api_url=args.api_url, skip_api_test=args.skip_api_test)
        await bot.start()
    except KeyboardInterrupt:
        logger.info("⛔ CTRL+C — stopping")
    finally:
        if bot:
            await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())