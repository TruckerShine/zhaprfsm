"""Telegram-бот для генерации видео через Veo 3.1.

Запуск:
    python bot.py

Секреты берутся только из .env:
    TELEGRAM_BOT_TOKEN
    GEMINI_API_KEY
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google import genai
from google.genai import types
from telegram import (
    InputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

# ---------------------------------------------------------------------------
# SOCKS5-прокси
# ---------------------------------------------------------------------------

# По умолчанию используем локальный Tor SOCKS5 на 9050. Отключить прокси можно
# явно через USE_SOCKS5_PROXY=0. Адрес 127.0.0.1 означает, что Tor/Dante должен
# работать на том же сервере, где запущен бот.
USE_SOCKS5_PROXY = os.getenv("USE_SOCKS5_PROXY", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
SOCKS5_PROXY = os.getenv(
    "SOCKS5_PROXY",
    "socks5://127.0.0.1:9050",
).strip()
if USE_SOCKS5_PROXY:
    SOCKS5_PROXY = SOCKS5_PROXY or "socks5://127.0.0.1:9050"
    os.environ["HTTP_PROXY"] = SOCKS5_PROXY
    os.environ["HTTPS_PROXY"] = SOCKS5_PROXY
    os.environ["ALL_PROXY"] = SOCKS5_PROXY
else:
    SOCKS5_PROXY = ""

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
VEO_MODEL = os.getenv("VEO_MODEL", "veo-3.1-generate-preview").strip()
IMAGE_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview").strip()
IMAGE_SIZE = os.getenv("GEMINI_IMAGE_SIZE", "1K").strip() or "1K"
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"


def _int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


HISTORY_DB_PATH = Path(os.getenv("HISTORY_DB_PATH", "history.db").strip() or "history.db")
HISTORY_MAX_BYTES = _int_env(
    "HISTORY_MAX_BYTES", 50 * 1024 * 1024, 0, 500 * 1024 * 1024
)
HISTORY_LIMIT = _int_env("HISTORY_LIMIT", 10, 1, 50)


POLL_SECONDS = _int_env("VEO_POLL_SECONDS", 10, 5, 60)
TIMEOUT_SECONDS = _int_env("VEO_TIMEOUT_SECONDS", 420, 30, 1800)
MAX_CONCURRENT_GENERATIONS = _int_env("MAX_CONCURRENT_GENERATIONS", 2, 1, 10)
# Прогресс Veo не сообщает точную ETA. Это стартовая оценка, которая уменьшается
# в сообщении каждые 10 секунд и после нуля показывает, что ожидание продолжается.
GENERATION_ESTIMATE_SECONDS = _int_env(
    "VEO_ESTIMATE_SECONDS", 120, 30, TIMEOUT_SECONDS
)

try:
    DISPLAY_TIMEZONE = ZoneInfo(os.getenv("VEO_LIMIT_TIMEZONE", "Europe/Moscow"))
except Exception:  # noqa: BLE001 — на сервере без tzdata используем системную зону
    DISPLAY_TIMEZONE = datetime.now().astimezone().tzinfo

# Пустое значение оставляет personGeneration на стороне API.
PERSON_GENERATION = os.getenv("VEO_PERSON_GENERATION", "allow_adult").strip()

DEFAULT_SETTINGS: dict[str, Any] = {
    "aspect_ratio": "9:16",
    # Для режима 15 секунд это будет первая часть: 8 + продолжение 7 секунд.
    "duration_seconds": 8,
    "resolution": "720p",
    "long_video": False,
}

ALLOWED_ASPECT_RATIOS = {"9:16", "16:9"}
ALLOWED_DURATIONS = {4, 6, 8, 15}
ALLOWED_RESOLUTIONS = {"720p", "1080p", "4k"}
UNSUPPORTED_RESOLUTIONS = {"320p", "480p"}
MAX_PROMPT_CHARS = 8000

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
)
logger = logging.getLogger("veo_bot")


GENERATION_LIMIT_MARKERS = (
    "429",
    "resource exhausted",
    "resource_exhausted",
    "too many requests",
    "rate limit",
    "rate_limit",
    "quota",
    "exceeded quota",
    "quota exceeded",
)


def looks_like_generation_limit(value: object) -> bool:
    text = str(value).lower().replace("_", " ")
    return any(marker.replace("_", " ") in text for marker in GENERATION_LIMIT_MARKERS)


def extract_retry_after_seconds(value: object) -> int | None:
    """Достаёт retry-after из текста ошибки Google, если он там есть."""
    text = str(value).lower()
    patterns = (
        r"retry[\s_-]+after\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)?",
        r"retry\s+in\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)",
        r"retry[\s_-]*delay[^0-9]{0,30}(\d+(?:\.\d+)?)",
        r"after\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return max(1, math.ceil(float(match.group(1))))
    return None


def format_duration(seconds: int) -> str:
    """Форматирует countdown без вывода 60 секунд вместо одной минуты."""
    seconds = max(0, int(seconds))
    minutes, rest = divmod(seconds, 60)
    if minutes:
        return f"{minutes} мин {rest:02d} сек"
    return f"{rest} сек"


class GenerationLimitError(RuntimeError):
    """Google API сообщил о квоте, rate limit или исчерпанном лимите."""

    def __init__(self, message: str = "") -> None:
        super().__init__(message)
        self.retry_after_seconds = extract_retry_after_seconds(message)


def generation_limit_text(error: GenerationLimitError | None = None) -> str:
    retry_after = error.retry_after_seconds if error else None
    if retry_after:
        reset_at = datetime.now(DISPLAY_TIMEZONE) + timedelta(seconds=retry_after)
        reset_line = (
            f"Повторить можно примерно через {format_duration(retry_after)} "
            f"(ориентировочно в {reset_at:%H:%M})."
        )
    else:
        reset_line = (
            "Google не передал точное время сброса лимита. Проверьте квоту проекта: "
            "для разных лимитов Google окно сброса отличается."
        )
    return (
        "⚠️ Лимит на генерацию достигнут.\n\n"
        "Google временно сообщил, что квота или допустимое число запросов исчерпаны.\n"
        f"{reset_line}\n\n"
        "Если сообщение повторяется, проверьте квоту, оплату и доступ к Veo "
        "в Google AI Studio или Google Cloud."
    )


def generation_progress_text(phase: str, elapsed_seconds: float) -> str:
    """Текст приблизительного обратного отсчёта для статусного сообщения."""
    remaining = GENERATION_ESTIMATE_SECONDS - int(elapsed_seconds)
    if remaining > 0:
        return f"⏳ {phase}\nПримерно осталось: {format_duration(remaining)}."
    return (
        f"⏳ {phase}\n"
        "Прогноз вышел, но Veo ещё работает — продолжаю проверять готовность."
    )


class HistoryStore:
    """Небольшое SQLite-хранилище результатов пользователя.

    Медиа сохраняются как BLOB, чтобы пользователь мог получить предыдущий
    результат даже после удаления временной папки генерации.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS generations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    mime_type TEXT NOT NULL,
                    media BLOB NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_generations_user_created "
                "ON generations(user_id, created_at DESC)"
            )

    def add(
        self,
        user_id: int,
        kind: str,
        prompt: str,
        file_name: str,
        mime_type: str,
        media: bytes,
    ) -> int | None:
        if not media or (HISTORY_MAX_BYTES and len(media) > HISTORY_MAX_BYTES):
            return None
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO generations
                    (user_id, kind, prompt, file_name, mime_type, media, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, kind, prompt, file_name, mime_type, media, created_at),
            )
            return int(cursor.lastrowid)

    def list_for_user(self, user_id: int, limit: int = HISTORY_LIMIT) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, kind, prompt, file_name, mime_type, created_at
                FROM generations
                WHERE user_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_for_user(self, user_id: int, generation_id: int) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, prompt, file_name, mime_type, media, created_at
                FROM generations
                WHERE id = ? AND user_id = ?
                """,
                (generation_id, user_id),
            ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Клиент Veo
# ---------------------------------------------------------------------------

class VeoService:
    """Синхронный SDK-клиент, вызываемый из asyncio через отдельные потоки."""

    def __init__(self, api_key: str) -> None:
        if SOCKS5_PROXY:
            # Переменные окружения используются Telegram/httpx и другими
            # HTTP-клиентами, а client_args гарантирует прокси для google-genai.
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    client_args={"proxy": SOCKS5_PROXY},
                ),
            )
            logger.info("SOCKS5 proxy enabled for HTTP clients")
        else:
            self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _build_config(
        settings: dict[str, Any],
        *,
        extension: bool = False,
        reference_images: list[types.VideoGenerationReferenceImage] | None = None,
    ) -> types.GenerateVideosConfig:
        if extension:
            # По документации расширение Veo добавляет 7 секунд и доступно только
            # в 720p. duration_seconds для extension не задаём.
            kwargs: dict[str, Any] = {
                "number_of_videos": 1,
                "resolution": "720p",
            }
        else:
            kwargs = {
                "number_of_videos": 1,
                "duration_seconds": settings["duration_seconds"],
                "aspect_ratio": settings["aspect_ratio"],
                "resolution": settings["resolution"],
            }

        if reference_images:
            kwargs["reference_images"] = reference_images
        if PERSON_GENERATION:
            kwargs["person_generation"] = PERSON_GENERATION
        return types.GenerateVideosConfig(**kwargs)

    def start_generation_sync(
        self,
        prompt: str,
        settings: dict[str, Any],
        image: types.Image | None,
        source_video: Any | None = None,
        extension: bool = False,
        reference_images: list[types.VideoGenerationReferenceImage] | None = None,
    ) -> Any:
        """Запускает text/image-to-video или extension и возвращает operation."""
        source = types.GenerateVideosSource(
            prompt=prompt,
            image=image,
            video=source_video,
        )
        return self.client.models.generate_videos(
            model=VEO_MODEL,
            source=source,
            config=self._build_config(
                settings,
                extension=extension,
                reference_images=reference_images,
            ),
        )

    def refresh_operation_sync(self, operation: Any) -> Any:
        return self.client.operations.get(operation)

    @staticmethod
    def _operation_error(operation: Any) -> str | None:
        error = getattr(operation, "error", None)
        if not error:
            return None
        message = str(getattr(error, "message", None) or error)
        if looks_like_generation_limit(message):
            raise GenerationLimitError(message)
        return message

    def download_result_sync(self, operation: Any, output_path: Path) -> Any:
        error_message = self._operation_error(operation)
        if error_message:
            raise RuntimeError(error_message)

        response = getattr(operation, "response", None)
        generated_videos = getattr(response, "generated_videos", None)
        if not generated_videos:
            raise RuntimeError("API завершил операцию без готового видео")

        generated_video = generated_videos[0]
        video = getattr(generated_video, "video", None)
        if video is None:
            raise RuntimeError("API не вернул объект видео")

        # Сохраняем локальную копию сразу: Google хранит результат ограниченное
        # время, а Telegram получает файл именно отсюда.
        self.client.files.download(file=video)
        video.save(str(output_path))
        return video

    def generate_image_sync(
        self,
        prompt: str,
        settings: dict[str, Any],
        input_images: list[types.Image] | None,
        output_path: Path,
    ) -> None:
        """Создаёт PNG через Gemini image model, optionally редактируя фото."""
        contents: list[Any] = [prompt]
        for input_image in input_images or []:
            image_bytes = getattr(input_image, "image_bytes", None)
            mime_type = getattr(input_image, "mime_type", None) or "image/jpeg"
            if not image_bytes:
                raise RuntimeError("Не удалось прочитать загруженное изображение")
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )

        response = self.client.models.generate_content(
            model=IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=settings["aspect_ratio"],
                    image_size=IMAGE_SIZE,
                ),
            ),
        )

        for part in getattr(response, "parts", None) or []:
            if getattr(part, "inline_data", None) is not None:
                generated_image = part.as_image()
                if generated_image is not None:
                    generated_image.save(str(output_path))
                    return
        raise RuntimeError("Модель не вернула изображение. Попробуйте изменить промпт.")

    def close(self) -> None:
        self.client.close()


async def generate_video_to_file(
    service: VeoService,
    prompt: str,
    settings: dict[str, Any],
    image: types.Image | None,
    output_path: Path,
    status_message: Any,
    *,
    source_video: Any | None = None,
    extension: bool = False,
    phase: str = "Генерирую видео",
    reference_images: list[types.VideoGenerationReferenceImage] | None = None,
) -> Any:
    """Запускает Veo, опрашивает operation, скачивает результат и возвращает Video."""
    try:
        operation = await asyncio.to_thread(
            service.start_generation_sync,
            prompt,
            settings,
            image,
            source_video,
            extension,
            reference_images,
        )
    except GenerationLimitError:
        raise
    except Exception as exc:
        if looks_like_generation_limit(exc):
            raise GenerationLimitError(str(exc)) from exc
        raise

    started_at = time.monotonic()
    last_status_update = 0.0

    while not getattr(operation, "done", False):
        elapsed = time.monotonic() - started_at
        if elapsed >= TIMEOUT_SECONDS:
            raise TimeoutError(
                "Время ожидания вышло. Операция Veo могла продолжить выполняться "
                "на стороне Google."
            )

        await asyncio.sleep(POLL_SECONDS)
        try:
            operation = await asyncio.to_thread(service.refresh_operation_sync, operation)
        except GenerationLimitError:
            raise
        except Exception as exc:
            if looks_like_generation_limit(exc):
                raise GenerationLimitError(str(exc)) from exc
            raise

        # Точный процент Veo не сообщает. Показываем приблизительный countdown
        # и обновляем его на каждом цикле polling (обычно раз в 10 секунд).
        elapsed = time.monotonic() - started_at
        if elapsed - last_status_update >= POLL_SECONDS:
            last_status_update = elapsed
            try:
                await status_message.edit_text(
                    generation_progress_text(phase, elapsed)
                )
            except TelegramError:
                pass

    try:
        return await asyncio.to_thread(
            service.download_result_sync,
            operation,
            output_path,
        )
    except GenerationLimitError:
        raise
    except Exception as exc:
        if looks_like_generation_limit(exc):
            raise GenerationLimitError(str(exc)) from exc
        raise


async def generate_image_to_file(
    service: VeoService,
    prompt: str,
    settings: dict[str, Any],
    images: list[types.Image],
    output_path: Path,
) -> None:
    """Асинхронная обёртка для генерации/редактирования изображения."""
    try:
        await asyncio.to_thread(
            service.generate_image_sync,
            prompt,
            settings,
            images,
            output_path,
        )
    except GenerationLimitError:
        raise
    except Exception as exc:
        if looks_like_generation_limit(exc):
            raise GenerationLimitError(str(exc)) from exc
        raise


# ---------------------------------------------------------------------------
# Меню и настройки
# ---------------------------------------------------------------------------


def get_user_settings(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    saved = context.user_data.setdefault("settings", dict(DEFAULT_SETTINGS))
    for key, value in DEFAULT_SETTINGS.items():
        saved.setdefault(key, value)
    return saved


def get_generation_mode(context: ContextTypes.DEFAULT_TYPE) -> str:
    # После /start пользователь сначала выбирает режим: видео или фото.
    return context.user_data.get("generation_mode", "idle")


def busy_generation_text(active_mode: str) -> str:
    if active_mode == "image":
        return "⚠️ Сейчас идёт генерация фото. Попробуйте запустить видео позже или используйте /cancel."
    return "⚠️ Сейчас идёт генерация видео. Попробуйте запустить фото позже или используйте /cancel."


def face_reference_prompt(prompt: str) -> str:
    """Усиливает запрос, когда пользователь приложил фото лица."""
    return (
        "Use the person or people shown in the uploaded photo as a visual identity "
        "reference. Preserve their facial features, hairstyle, skin tone, and overall "
        "appearance consistently throughout the result. Do not replace or distort the "
        "face unless the user explicitly asks for it.\n\n"
        f"User request: {prompt}"
    )


def image_prompt_text(settings: dict[str, Any]) -> str:
    return (
        "🖼 Генерация фото\n\n"
        f"Формат: {settings['aspect_ratio']}\n\n"
        "Отправьте текстовый промпт, и я создам изображение.\n"
        "Чтобы сохранить лицо, отправьте фотографию с подписью-промптом, например:\n\n"
        "Добавь моё лицо на космический портрет, кинематографичный свет."
    )


def main_menu_text(settings: dict[str, Any]) -> str:
    orientation = (
        "вертикальный 9:16"
        if settings["aspect_ratio"] == "9:16"
        else "горизонтальный 16:9"
    )
    return (
        "🎬 Главное меню Veo 3.1\n\n"
        "Сначала выберите режим: видео или фото. Затем бот покажет настройки "
        "формата и остальные параметры.\n\n"
        f"Последний выбранный формат: {orientation}\n"
        f"Последняя длительность видео: {settings['duration_seconds']} сек."
    )


def main_menu_keyboard(settings: dict[str, Any]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("🎬 Видео", callback_data="menu:video"),
                InlineKeyboardButton("🖼 Фото", callback_data="menu:image"),
            ],
            [
                InlineKeyboardButton("⚙️ Настройки видео", callback_data="menu:settings"),
                InlineKeyboardButton("📚 История", callback_data="menu:history"),
            ],
            [InlineKeyboardButton("❓ Помощь", callback_data="menu:help")],
        ]
    )


def mode_menu_text(mode: str, settings: dict[str, Any]) -> str:
    if mode == "image":
        return (
            "🖼 Режим фото\n\n"
            "Выберите ориентацию изображения. После этого отправьте промпт "
            "или фотографию с подписью. В режиме фото результатом будет фото, "
            "а не видео.\n\n"
            f"Текущий формат: {settings['aspect_ratio']}"
        )
    return (
        "🎬 Режим видео\n\n"
        "Выберите ориентацию видео. Затем появятся длительность и качество.\n\n"
        f"Текущий формат: {settings['aspect_ratio']}"
    )


def mode_menu_keyboard(settings: dict[str, Any]) -> InlineKeyboardMarkup:
    current_aspect = settings["aspect_ratio"]
    horizontal = "✅ Горизонтально" if current_aspect == "16:9" else "Горизонтально"
    vertical = "✅ Вертикально" if current_aspect == "9:16" else "Вертикально"
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    f"{horizontal} 16:9", callback_data="flow:aspect:16:9"
                ),
                InlineKeyboardButton(
                    f"{vertical} 9:16", callback_data="flow:aspect:9:16"
                ),
            ],
            [InlineKeyboardButton("⬅️ Главное меню", callback_data="menu:home")],
        ]
    )


def back_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton("⬅️ Главное меню", callback_data="menu:home")]]
    )


def bottom_menu_keyboard() -> ReplyKeyboardMarkup:
    """Постоянная кнопка внизу Telegram, под полем ввода пользователя.

    Это ReplyKeyboardMarkup, а не InlineKeyboardMarkup: кнопка находится у
    пользователя под строкой ввода и не прикрепляется к сообщению бота.
    """
    return ReplyKeyboardMarkup(
        [[KeyboardButton("☰ Меню"), KeyboardButton("⏹ Отмена")]],
        resize_keyboard=True,
        is_persistent=True,
        input_field_placeholder="Промпт или кнопка меню",
    )


def duration_menu_text(settings: dict[str, Any]) -> str:
    aspect = settings["aspect_ratio"]
    return (
        "⏱ Выберите длительность видео\n\n"
        f"Формат: {aspect}\n\n"
        "15 секунд создаются как 8 секунд первой части и автоматическое "
        "продолжение ещё на 7 секунд. Вторая часть будет построена как "
        "логичное продолжение первого фрагмента."
    )


def duration_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("4 сек", callback_data="duration:4"),
                InlineKeyboardButton("6 сек", callback_data="duration:6"),
                InlineKeyboardButton("8 сек", callback_data="duration:8"),
                InlineKeyboardButton("15 сек", callback_data="duration:15"),
            ],
            [InlineKeyboardButton("⬅️ Главное меню", callback_data="menu:home")],
        ]
    )


def quality_menu_text(settings: dict[str, Any]) -> str:
    if settings.get("long_video"):
        duration = "15 сек. (8 сек. + продолжение 7 сек.)"
        restriction = "Для режима 15 секунд по документации доступно только 720p."
    else:
        duration = f"{settings['duration_seconds']} сек."
        restriction = (
            "Для 4/6 секунд доступны 720p. 1080p и 4K — только при длительности 8 секунд."
        )
    return (
        "🎞 Выберите качество видео\n\n"
        f"Длительность: {duration}\n\n"
        "По API Veo 3.1 доступны нативные варианты 720p, 1080p и 4K.\n"
        "320p и 480p модель Veo 3.1 напрямую не поддерживает.\n\n"
        f"{restriction}"
    )


def quality_keyboard(settings: dict[str, Any]) -> InlineKeyboardMarkup:
    duration = settings["duration_seconds"]
    long_video = bool(settings.get("long_video"))
    current = settings.get("resolution") or "720p"

    def selected(value: str) -> str:
        return f"✅ {value}" if value == current else value

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton("🚫 320p", callback_data="quality:unavailable:320p"),
            InlineKeyboardButton("🚫 480p", callback_data="quality:unavailable:480p"),
        ],
        [
            InlineKeyboardButton(selected("720p"), callback_data="quality:720p"),
        ],
    ]

    if long_video:
        rows.append(
            [
                InlineKeyboardButton(
                    "🚫 1080p (только 720p)", callback_data="quality:unavailable:1080p"
                ),
                InlineKeyboardButton(
                    "🚫 4K (только 720p)", callback_data="quality:unavailable:4k"
                ),
            ]
        )
    elif duration == 8:
        rows.append(
            [
                InlineKeyboardButton(selected("1080p"), callback_data="quality:1080p"),
                InlineKeyboardButton(selected("4K"), callback_data="quality:4k"),
            ]
        )
    else:
        rows.append(
            [
                InlineKeyboardButton(
                    "🚫 1080p (только 8 сек.)",
                    callback_data="quality:unavailable:1080p",
                ),
                InlineKeyboardButton(
                    "🚫 4K (только 8 сек.)",
                    callback_data="quality:unavailable:4k",
                ),
            ]
        )

    rows.append([InlineKeyboardButton("⬅️ Главное меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(rows)


def prompt_menu_text(settings: dict[str, Any]) -> str:
    duration = "15 сек. (8 + 7 продолжение)" if settings.get("long_video") else f"{settings['duration_seconds']} сек."
    return (
        "✅ Настройки выбраны\n\n"
        f"Формат: {settings['aspect_ratio']}\n"
        f"Длительность: {duration}\n"
        f"Качество: {settings['resolution']}\n\n"
        "Теперь отправьте текстовый промпт.\n"
        "Можно также отправить фотографию с подписью — она станет первым кадром."
    )


async def delete_callback_message(update: Update) -> None:
    query = update.callback_query
    if query is None or query.message is None:
        return
    try:
        await query.message.delete()
    except TelegramError:
        # Если удаление запрещено, новое меню всё равно будет отправлено ниже.
        pass


async def send_menu_after_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    reply_markup: InlineKeyboardMarkup,
) -> None:
    await delete_callback_message(update)
    chat = update.effective_chat
    if chat is not None:
        await context.bot.send_message(
            chat_id=chat.id,
            text=text,
            reply_markup=reply_markup,
        )


def format_history_date(value: str) -> str:
    try:
        created_at = datetime.fromisoformat(value).astimezone(DISPLAY_TIMEZONE)
        return created_at.strftime("%d.%m.%Y %H:%M")
    except (TypeError, ValueError):
        return value


def history_menu_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return (
            "📚 История генераций пока пустая.\n\n"
            "Создайте фото или видео, и оно появится здесь."
        )
    return "📚 История генераций\n\nВыберите результат, чтобы отправить его ещё раз."


def history_keyboard(rows: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    buttons: list[list[InlineKeyboardButton]] = []
    for row in rows:
        icon = "🖼" if row["kind"] == "photo" else "🎬"
        prompt = " ".join(str(row["prompt"]).split())
        if len(prompt) > 28:
            prompt = prompt[:25] + "…"
        label = f"{icon} {format_history_date(row['created_at'])} — {prompt}"
        buttons.append(
            [InlineKeyboardButton(label, callback_data=f"history:view:{row['id']}")]
        )
    buttons.append([InlineKeyboardButton("⬅️ Главное меню", callback_data="menu:home")])
    return InlineKeyboardMarkup(buttons)


async def open_history_after_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    user = update.effective_user
    chat = update.effective_chat
    if user is None or chat is None:
        return
    store: HistoryStore = context.application.bot_data["history_store"]
    rows = await asyncio.to_thread(store.list_for_user, user.id, HISTORY_LIMIT)
    await delete_callback_message(update)
    await context.bot.send_message(
        chat_id=chat.id,
        text=history_menu_text(rows),
        reply_markup=history_keyboard(rows),
    )


# ---------------------------------------------------------------------------
# Работа с файлами
# ---------------------------------------------------------------------------


def extract_last_seconds_sync(
    source_path: Path,
    output_path: Path,
    seconds: int = 7,
) -> bool:
    """Пытается вырезать только продолжение из объединённого extension-видео.

    API Veo возвращает единый файл «исходное видео + продолжение». Если ffmpeg
    установлен, сохраняем последние 7 секунд отдельным MP4. Если его нет,
    вызывающий код отправит только объединённый результат примерно на 15 секунд.
    """
    executable = FFMPEG_BIN
    if not os.path.isabs(executable):
        executable = shutil.which(executable) or ""
    if not executable:
        return False

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-sseof",
        f"-{seconds}",
        "-i",
        str(source_path),
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0


async def save_history_result(
    history_store: HistoryStore | None,
    user_id: int | None,
    kind: str,
    prompt: str,
    path: Path,
    media: bytes,
) -> None:
    if history_store is None or user_id is None:
        return
    try:
        generation_id = await asyncio.to_thread(
            history_store.add,
            user_id,
            kind,
            prompt,
            path.name,
            "video/mp4" if kind == "video" else "image/png",
            media,
        )
        if generation_id is None and HISTORY_MAX_BYTES:
            logger.info(
                "History item skipped because it is larger than HISTORY_MAX_BYTES: %s",
                path,
            )
    except Exception:  # noqa: BLE001 — история не должна ломать отправку результата
        logger.exception("Could not save generation to history")


async def send_video_file(
    message: Any,
    path: Path,
    caption: str,
    *,
    history_store: HistoryStore | None = None,
    user_id: int | None = None,
    prompt: str = "",
) -> None:
    """Отправляет MP4 и после успешной отправки сохраняет его в историю."""
    media = path.read_bytes()
    try:
        await message.reply_video(
            video=InputFile(media, filename=path.name),
            caption=caption,
            supports_streaming=True,
        )
    except TelegramError:
        await message.reply_document(
            document=InputFile(media, filename=path.name),
            caption=caption,
        )
    await save_history_result(history_store, user_id, "video", prompt, path, media)


async def send_image_file(
    message: Any,
    path: Path,
    caption: str,
    *,
    history_store: HistoryStore | None = None,
    user_id: int | None = None,
    prompt: str = "",
) -> None:
    """Отправляет PNG и после успешной отправки сохраняет его в историю."""
    media = path.read_bytes()
    try:
        await message.reply_photo(
            photo=InputFile(media, filename=path.name),
            caption=caption,
        )
    except TelegramError:
        await message.reply_document(
            document=InputFile(media, filename=path.name),
            caption=caption,
        )
    await save_history_result(history_store, user_id, "photo", prompt, path, media)


# ---------------------------------------------------------------------------
# Обработчики генерации
# ---------------------------------------------------------------------------


def clean_prompt(text: str) -> str:
    prompt = text.strip()
    if not prompt:
        raise ValueError("Промпт не должен быть пустым")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"Промпт слишком длинный. Сократите его до {MAX_PROMPT_CHARS} символов."
        )
    return prompt


def caption_for_prompt(prompt: str, title: str = "✅ Готово!") -> str:
    shortened = prompt if len(prompt) <= 800 else prompt[:797] + "…"
    return f"{title}\n\nПромпт: {shortened}"


def continuation_prompt(prompt: str) -> str:
    return (
        "Continue the previous Veo video seamlessly. Preserve the same characters, "
        "setting, visual style, camera movement, lighting, and audio continuity. "
        "Do not restart the scene. Continue the action from the last frame and make "
        "the next seven seconds end naturally and logically. Use this original video "
        f"direction as the story context: {prompt}"
    )


async def safe_edit(message: Any, text: str) -> None:
    try:
        await message.edit_text(text)
    except TelegramError:
        pass


async def launch_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    image: types.Image | None = None,
    images: list[types.Image] | None = None,
) -> None:
    """Ставит одну пользовательскую задачу в локальную очередь."""
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    try:
        prompt = clean_prompt(prompt)
    except ValueError as exc:
        await message.reply_text(f"❌ {exc}")
        return

    application = context.application
    tasks: dict[int, asyncio.Task[Any]] = application.bot_data.setdefault("tasks", {})
    active_modes: dict[int, str] = application.bot_data.setdefault("active_modes", {})
    existing_task = tasks.get(user.id)
    active_mode = active_modes.get(user.id)
    if (existing_task and not existing_task.done()) or active_mode:
        await message.reply_text(busy_generation_text(active_mode or "video"))
        return

    # Резервируем пользователя до первого await, чтобы два быстрых запроса
    # одновременно не прошли обе проверки.
    active_modes[user.id] = "video"

    input_images = list(images or [])
    if image is not None:
        input_images.insert(0, image)
    if len(input_images) > 3:
        await message.reply_text("Можно использовать максимум 3 референсных фото.")
        active_modes.pop(user.id, None)
        return

    settings = dict(get_user_settings(context))
    # Для одного фото используем image как первый кадр. Для 2–3 фото используем
    # официальный reference_images; Veo требует 8 секунд для этого режима.
    primary_image = input_images[0] if len(input_images) == 1 else None
    reference_images = None
    if len(input_images) > 1:
        reference_images = [
            types.VideoGenerationReferenceImage(
                image=reference,
                reference_type="asset",
            )
            for reference in input_images
        ]
        settings["duration_seconds"] = 8

    # Если приложены фото, добавляем инструкцию сохранить внешность людей.
    generation_prompt = face_reference_prompt(prompt) if input_images else prompt
    # Нормализация на случай, если пользователь написал промпт сразу после /start.
    if settings.get("long_video"):
        settings["duration_seconds"] = 8
        settings["resolution"] = "720p"
    elif settings.get("resolution") not in ALLOWED_RESOLUTIONS:
        settings["resolution"] = "720p"
    if settings["duration_seconds"] != 8 and settings["resolution"] in {"1080p", "4k"}:
        settings["resolution"] = "720p"

    status_message = await message.reply_text(
        "🚀 Запрос принят. Veo генерирует видео и звук — это может занять несколько минут."
    )

    async def worker() -> None:
        service: VeoService = application.bot_data["veo_service"]
        semaphore: asyncio.Semaphore = application.bot_data["generation_semaphore"]
        history_store: HistoryStore = application.bot_data["history_store"]
        try:
            await safe_edit(status_message, "⏳ Жду свободный слот Veo…")
            async with semaphore:
                with tempfile.TemporaryDirectory(prefix="veo-telegram-") as temp_dir:
                    temp_path = Path(temp_dir)
                    first_path = temp_path / "veo_part_1.mp4"
                    await safe_edit(
                        status_message,
                        generation_progress_text("Генерирую первую часть видео", 0),
                    )
                    first_video = await generate_video_to_file(
                        service=service,
                        prompt=generation_prompt,
                        settings=settings,
                        image=primary_image,
                        output_path=first_path,
                        status_message=status_message,
                        phase="Генерирую первую часть видео",
                        reference_images=reference_images,
                    )

                    if not settings.get("long_video"):
                        await safe_edit(status_message, "✅ Видео готово, отправляю файл…")
                        await send_video_file(
                            message,
                            first_path,
                            caption_for_prompt(prompt),
                            history_store=history_store,
                            user_id=user.id,
                            prompt=prompt,
                        )
                        await safe_edit(status_message, "✅ Видео отправлено")
                        return

                    # Режим 15 секунд: первая генерация 8 секунд, затем официальный
                    # Veo extension ещё на 7 секунд с логичным продолжением.
                    await safe_edit(
                        status_message,
                        "✅ Первая часть готова. Автоматически создаю логичное продолжение ещё на 7 секунд…\n"
                        + generation_progress_text("Создаю вторую часть видео", 0),
                    )
                    combined_path = temp_path / "veo_15_seconds_combined.mp4"
                    extended_video = await generate_video_to_file(
                        service=service,
                        prompt=continuation_prompt(generation_prompt),
                        settings=settings,
                        image=None,
                        output_path=combined_path,
                        status_message=status_message,
                        source_video=first_video,
                        extension=True,
                        phase="Создаю вторую часть видео",
                    )
                    del extended_video  # Объект нужен SDK только во время операции.

                    # Важно: до этого момента пользователю не отправляем ни одного
                    # файла. Сначала полностью ждём extension, затем проверяем,
                    # получится ли выделить отдельные последние 7 секунд.
                    await safe_edit(
                        status_message,
                        "✅ Обе части готовы. Подготавливаю файлы для отправки…",
                    )
                    second_path = temp_path / "veo_part_2.mp4"
                    has_separate_second = await asyncio.to_thread(
                        extract_last_seconds_sync,
                        combined_path,
                        second_path,
                        7,
                    )

                    if has_separate_second:
                        # FFmpeg есть: отправляем два отдельных файла только после
                        # того, как готовы и первая часть, и вырезанная вторая часть.
                        await safe_edit(status_message, "✅ Отправляю две отдельные части…")
                        await send_video_file(
                            message,
                            first_path,
                            caption_for_prompt(prompt, "🎬 Часть 1 из 2 — 8 секунд"),
                            history_store=history_store,
                            user_id=user.id,
                            prompt=prompt,
                        )
                        await send_video_file(
                            message,
                            second_path,
                            caption_for_prompt(
                                prompt,
                                "🎬 Часть 2 из 2 — продолжение 7 секунд",
                            ),
                            history_store=history_store,
                            user_id=user.id,
                            prompt=prompt,
                        )
                    else:
                        # Без FFmpeg отдельный второй фрагмент получить нельзя.
                        # В этом случае отправляем только полный результат extension,
                        # а part1 не отправляем и удаляем до отправки.
                        try:
                            first_path.unlink()
                        except FileNotFoundError:
                            pass
                        await safe_edit(
                            status_message,
                            "✅ Полное видео готово. FFmpeg не найден — отправляю один файл на 15 секунд…",
                        )
                        await send_video_file(
                            message,
                            combined_path,
                            caption_for_prompt(
                                prompt,
                                "🎬 Полный результат — примерно 15 секунд",
                            ),
                            history_store=history_store,
                            user_id=user.id,
                            prompt=prompt,
                        )
                    await safe_edit(status_message, "✅ Видео отправлено")
        except asyncio.CancelledError:
            logger.info("Generation cancelled locally for user %s", user.id)
            await safe_edit(status_message, "⏹ Ожидание остановлено командой /cancel")
        except GenerationLimitError as exc:
            logger.warning("Veo generation limit for user %s: %s", user.id, exc)
            await safe_edit(status_message, generation_limit_text(exc))
        except TimeoutError as exc:
            logger.warning("Veo timeout for user %s: %s", user.id, exc)
            await safe_edit(status_message, f"⏱ {exc}")
        except Exception as exc:  # noqa: BLE001 — показываем пользователю безопасное сообщение
            if looks_like_generation_limit(exc):
                logger.warning("Veo generation limit for user %s: %s", user.id, exc)
                await safe_edit(
                    status_message,
                    generation_limit_text(GenerationLimitError(str(exc))),
                )
                return
            logger.exception("Veo generation failed for user %s", user.id)
            error_text = str(exc).strip() or "неизвестная ошибка"
            if GEMINI_API_KEY:
                error_text = error_text.replace(GEMINI_API_KEY, "[ключ скрыт]")
            if TELEGRAM_BOT_TOKEN:
                error_text = error_text.replace(TELEGRAM_BOT_TOKEN, "[токен скрыт]")
            if len(error_text) > 700:
                error_text = error_text[:697] + "…"
            await safe_edit(status_message, f"❌ Не удалось создать видео:\n{error_text}")
        finally:
            if tasks.get(user.id) is asyncio.current_task():
                tasks.pop(user.id, None)
            if active_modes.get(user.id) == "video":
                active_modes.pop(user.id, None)

    task = asyncio.create_task(worker(), name=f"veo-generation-{user.id}")
    tasks[user.id] = task


async def launch_image_generation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    prompt: str,
    image: types.Image | None = None,
    images: list[types.Image] | None = None,
) -> None:
    """Ставит задачу text-to-image или image-to-image в ту же очередь."""
    message = update.effective_message
    user = update.effective_user
    if message is None or user is None:
        return

    try:
        prompt = clean_prompt(prompt)
    except ValueError as exc:
        await message.reply_text(f"❌ {exc}")
        return

    application = context.application
    tasks: dict[int, asyncio.Task[Any]] = application.bot_data.setdefault("tasks", {})
    active_modes: dict[int, str] = application.bot_data.setdefault("active_modes", {})
    existing_task = tasks.get(user.id)
    active_mode = active_modes.get(user.id)
    if (existing_task and not existing_task.done()) or active_mode:
        await message.reply_text(busy_generation_text(active_mode or "image"))
        return

    # Резервируем пользователя до первого await для защиты от двух быстрых запросов.
    active_modes[user.id] = "image"

    input_images = list(images or [])
    if image is not None:
        input_images.insert(0, image)
    if len(input_images) > 3:
        await message.reply_text("Можно использовать максимум 3 референсных фото.")
        active_modes.pop(user.id, None)
        return

    settings = dict(get_user_settings(context))
    generation_prompt = face_reference_prompt(prompt) if input_images else prompt
    status_message = await message.reply_text(
        "🚀 Запрос на фото принят. Создаю изображение…"
    )

    async def worker() -> None:
        service: VeoService = application.bot_data["veo_service"]
        semaphore: asyncio.Semaphore = application.bot_data["generation_semaphore"]
        history_store: HistoryStore = application.bot_data["history_store"]
        try:
            async with semaphore:
                await safe_edit(status_message, "⏳ Создаю изображение…")
                with tempfile.TemporaryDirectory(prefix="gemini-image-") as temp_dir:
                    output_path = Path(temp_dir) / "generated_image.png"
                    await generate_image_to_file(
                        service=service,
                        prompt=generation_prompt,
                        settings=settings,
                        images=input_images,
                        output_path=output_path,
                    )
                    await safe_edit(status_message, "✅ Фото готово, отправляю файл…")
                    await send_image_file(
                        message,
                        output_path,
                        caption_for_prompt(prompt, "🖼 Готово"),
                        history_store=history_store,
                        user_id=user.id,
                        prompt=prompt,
                    )
                    await safe_edit(status_message, "✅ Фото отправлено")
        except asyncio.CancelledError:
            logger.info("Image generation cancelled locally for user %s", user.id)
            await safe_edit(status_message, "⏹ Ожидание остановлено командой /cancel")
        except GenerationLimitError as exc:
            logger.warning("Image generation limit for user %s: %s", user.id, exc)
            await safe_edit(status_message, generation_limit_text(exc))
        except Exception as exc:  # noqa: BLE001 — показываем безопасное сообщение
            if looks_like_generation_limit(exc):
                logger.warning("Image generation limit for user %s: %s", user.id, exc)
                await safe_edit(
                    status_message,
                    generation_limit_text(GenerationLimitError(str(exc))),
                )
                return
            logger.exception("Image generation failed for user %s", user.id)
            error_text = str(exc).strip() or "неизвестная ошибка"
            if GEMINI_API_KEY:
                error_text = error_text.replace(GEMINI_API_KEY, "[ключ скрыт]")
            if TELEGRAM_BOT_TOKEN:
                error_text = error_text.replace(TELEGRAM_BOT_TOKEN, "[токен скрыт]")
            if len(error_text) > 700:
                error_text = error_text[:697] + "…"
            await safe_edit(status_message, f"❌ Не удалось создать фото:\n{error_text}")
        finally:
            if tasks.get(user.id) is asyncio.current_task():
                tasks.pop(user.id, None)
            if active_modes.get(user.id) == "image":
                active_modes.pop(user.id, None)

    task = asyncio.create_task(worker(), name=f"image-generation-{user.id}")
    tasks[user.id] = task


# ---------------------------------------------------------------------------
# Обработчики меню и команд
# ---------------------------------------------------------------------------


def help_text() -> str:
    return (
        "📝 Хороший промпт обычно содержит:\n"
        "• тему и действие;\n"
        "• стиль и атмосферу;\n"
        "• движение камеры и композицию;\n"
        "• звуки, SFX и диалоги в кавычках.\n\n"
        "Для фото нажмите «🖼 Создать фото». Чтобы использовать своё лицо, "
        "отправьте фотографию вместе с подписью-промптом, например: "
        "«Добавь моё лицо на кинематографичный портрет в неоновом городе».\n\n"
        "Пример:\n\n"
        "Кинематографичный вертикальный ролик: маленький робот поливает сад "
        "светящихся грибов на миниатюрной планете. Медленный наезд камеры, "
        "мягкий туман, слышны капли воды и тихое электронное эмбиент-звучание."
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["generation_mode"] = "idle"
    settings = get_user_settings(context)
    # ReplyKeyboardMarkup показывается в нижней панели пользователя. Inline-меню
    # отправляем отдельным сообщением, потому что Telegram принимает только один
    # тип reply_markup у одного сообщения.
    await update.effective_message.reply_text(
        "✅ Кнопка «☰ Меню» добавлена под полем ввода.",
        reply_markup=bottom_menu_keyboard(),
    )
    await update.effective_message.reply_text(
        main_menu_text(settings), reply_markup=main_menu_keyboard(settings)
    )


async def bottom_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Открывает inline-меню по постоянной кнопке под вводом пользователя."""
    context.user_data["generation_mode"] = "idle"
    settings = get_user_settings(context)
    await update.effective_message.reply_text(
        main_menu_text(settings), reply_markup=main_menu_keyboard(settings)
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(help_text(), reply_markup=back_keyboard())


async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data["generation_mode"] = "video"
    settings = get_user_settings(context)
    await update.effective_message.reply_text(
        mode_menu_text("video", settings), reply_markup=mode_menu_keyboard(settings)
    )


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return
    store: HistoryStore = context.application.bot_data["history_store"]
    rows = await asyncio.to_thread(store.list_for_user, user.id, HISTORY_LIMIT)
    await update.effective_message.reply_text(
        history_menu_text(rows), reply_markup=history_keyboard(rows)
    )


async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Главное меню: сначала пользователь выбирает режим фото или видео."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    data = query.data or ""
    settings = get_user_settings(context)

    if data == "menu:home":
        context.user_data["generation_mode"] = "idle"
        await send_menu_after_callback(
            update,
            context,
            main_menu_text(settings),
            main_menu_keyboard(settings),
        )
        return

    if data in {"menu:video", "menu:create", "menu:settings"}:
        context.user_data["generation_mode"] = "video"
        await send_menu_after_callback(
            update,
            context,
            mode_menu_text("video", settings),
            mode_menu_keyboard(settings),
        )
        return

    if data == "menu:image":
        context.user_data["generation_mode"] = "image"
        await send_menu_after_callback(
            update,
            context,
            mode_menu_text("image", settings),
            mode_menu_keyboard(settings),
        )
        return

    if data == "menu:history":
        await open_history_after_callback(update, context)
        return

    if data == "menu:help":
        await send_menu_after_callback(update, context, help_text(), back_keyboard())


async def history_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    user = update.effective_user
    chat = update.effective_chat
    if query is None or user is None or chat is None:
        return

    data = query.data or ""
    if data == "history:menu":
        await query.answer()
        await open_history_after_callback(update, context)
        return

    parts = data.split(":", 2)
    if len(parts) != 3 or parts[0] != "history" or parts[1] != "view":
        await query.answer("Неизвестный пункт истории", show_alert=True)
        return

    try:
        generation_id = int(parts[2])
    except ValueError:
        await query.answer("Некорректный результат", show_alert=True)
        return

    await query.answer("Загружаю результат…")
    store: HistoryStore = context.application.bot_data["history_store"]
    row = await asyncio.to_thread(store.get_for_user, user.id, generation_id)
    if row is None:
        await context.bot.send_message(
            chat_id=chat.id,
            text="❌ Результат не найден в вашей истории.",
        )
        return

    caption = caption_for_prompt(
        row["prompt"],
        "🖼 Фото из истории" if row["kind"] == "photo" else "🎬 Видео из истории",
    )
    try:
        if row["kind"] == "photo":
            await context.bot.send_photo(
                chat_id=chat.id,
                photo=InputFile(row["media"], filename=row["file_name"]),
                caption=caption,
            )
        else:
            await context.bot.send_video(
                chat_id=chat.id,
                video=InputFile(row["media"], filename=row["file_name"]),
                caption=caption,
                supports_streaming=True,
            )
    except TelegramError:
        await context.bot.send_document(
            chat_id=chat.id,
            document=InputFile(row["media"], filename=row["file_name"]),
            caption=caption,
        )


async def flow_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сохраняет ориентацию и переходит к следующему экрану выбранного режима."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    parts = (query.data or "").split(":", 2)
    if len(parts) != 3 or parts[0] != "flow" or parts[1] != "aspect":
        return
    if parts[2] not in ALLOWED_ASPECT_RATIOS:
        return

    settings = get_user_settings(context)
    settings["aspect_ratio"] = parts[2]
    mode = get_generation_mode(context)

    if mode == "image":
        await send_menu_after_callback(
            update,
            context,
            image_prompt_text(settings),
            back_keyboard(),
        )
        return

    # Видео: после ориентации открываем длительность, затем качество.
    context.user_data["generation_mode"] = "video"
    settings["duration_seconds"] = 8
    settings["resolution"] = "720p"
    settings["long_video"] = False
    await send_menu_after_callback(
        update,
        context,
        duration_menu_text(settings),
        duration_keyboard(),
    )


async def duration_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    await query.answer()

    value = (query.data or "").split(":", 1)[-1]
    try:
        duration = int(value)
    except ValueError:
        return
    if duration not in ALLOWED_DURATIONS:
        return

    context.user_data["generation_mode"] = "video"
    settings = get_user_settings(context)
    if duration == 15:
        settings["long_video"] = True
        settings["duration_seconds"] = 8
        settings["resolution"] = "720p"
    else:
        settings["long_video"] = False
        settings["duration_seconds"] = duration
        if duration != 8 and settings.get("resolution") in {"1080p", "4k"}:
            settings["resolution"] = "720p"

    await send_menu_after_callback(
        update,
        context,
        quality_menu_text(settings),
        quality_keyboard(settings),
    )


async def quality_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return

    data = (query.data or "").split(":")
    if len(data) == 3 and data[1] == "unavailable":
        quality = data[2]
        if quality in UNSUPPORTED_RESOLUTIONS:
            await query.answer(
                f"{quality} не поддерживается Veo 3.1 по API. Доступно: 720p, 1080p и 4K.",
                show_alert=True,
            )
        else:
            await query.answer(
                "Это качество недоступно для выбранной длительности. По документации "
                "1080p/4K работают только с 8 секундами, а extension — только в 720p.",
                show_alert=True,
            )
        return

    if len(data) != 2 or data[0] != "quality" or data[1] not in ALLOWED_RESOLUTIONS:
        await query.answer("Неизвестное качество", show_alert=True)
        return

    quality = data[1]
    context.user_data["generation_mode"] = "video"
    settings = get_user_settings(context)
    if settings.get("long_video") and quality != "720p":
        await query.answer("Для 15 секунд доступно только 720p", show_alert=True)
        return
    if settings["duration_seconds"] != 8 and quality in {"1080p", "4k"}:
        await query.answer("1080p и 4K доступны только при длительности 8 секунд", show_alert=True)
        return

    settings["resolution"] = quality
    await query.answer("Качество сохранено")
    await send_menu_after_callback(
        update,
        context,
        prompt_menu_text(settings),
        back_keyboard(),
    )


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if user is None:
        return
    tasks: dict[int, asyncio.Task[Any]] = context.application.bot_data.setdefault("tasks", {})
    task = tasks.get(user.id)
    if task and not task.done():
        task.cancel()
        await update.effective_message.reply_text(
            "⏹ Ожидание остановлено. Уже отправленный запрос может ещё завершиться "
            "на стороне Google, но бот не будет ждать его результат."
        )
    else:
        await update.effective_message.reply_text("Сейчас активных генераций нет.")


async def text_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.text:
        return

    mode = get_generation_mode(context)
    if mode not in {"image", "video"}:
        await message.reply_text(
            "Сначала нажмите ☰ Меню и выберите режим: 🎬 Видео или 🖼 Фото."
        )
        return

    pending_payloads = context.user_data.pop("pending_reference_images", [])
    # Совместимость с состоянием, которое могло остаться от старой версии бота.
    legacy_image = context.user_data.pop("pending_image", None)
    if legacy_image:
        pending_payloads.append(legacy_image)

    pending_images = [
        types.Image(image_bytes=item["bytes"], mime_type=item["mime_type"])
        for item in pending_payloads
    ]

    if mode == "image":
        await launch_image_generation(
            update,
            context,
            message.text,
            images=pending_images,
        )
    else:
        await launch_generation(
            update,
            context,
            message.text,
            images=pending_images,
        )


async def photo_prompt(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message is None or not message.photo:
        return

    mode = get_generation_mode(context)
    if mode not in {"image", "video"}:
        await message.reply_text(
            "Сначала нажмите ☰ Меню и выберите режим: 🎬 Видео или 🖼 Фото."
        )
        return

    largest_photo = message.photo[-1]
    telegram_file = await context.bot.get_file(largest_photo.file_id)
    image_bytes = bytes(await telegram_file.download_as_bytearray())
    payload = {"bytes": image_bytes, "mime_type": "image/jpeg"}
    caption = (message.caption or "").strip()
    pending_payloads = list(context.user_data.get("pending_reference_images", []))

    if caption:
        all_payloads = pending_payloads + [payload]
        if len(all_payloads) > 3:
            await message.reply_text("Можно использовать максимум 3 референсных фото.")
            return
        context.user_data.pop("pending_reference_images", None)
        images = [
            types.Image(image_bytes=item["bytes"], mime_type=item["mime_type"])
            for item in all_payloads
        ]
        if mode == "image":
            await launch_image_generation(update, context, caption, images=images)
        else:
            await launch_generation(update, context, caption, images=images)
        return

    if len(pending_payloads) >= 3:
        await message.reply_text(
            "Уже сохранено 3 референсных фото. Теперь отправьте текстовый промпт."
        )
        return

    pending_payloads.append(payload)
    context.user_data["pending_reference_images"] = pending_payloads
    count = len(pending_payloads)
    if mode == "image":
        text = (
            f"🖼 Референсное фото {count}/3 сохранено. Отправьте ещё фото без подписи "
            "или напишите промпт — будет создано фото."
        )
    else:
        text = (
            f"🖼 Референсное фото {count}/3 сохранено. Отправьте ещё фото без подписи "
            "или напишите промпт — будет создано видео."
        )
    await message.reply_text(text)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled Telegram error", exc_info=context.error)


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------


def build_application() -> Application:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Не задан TELEGRAM_BOT_TOKEN в .env")
    if not GEMINI_API_KEY:
        raise RuntimeError("Не задан GEMINI_API_KEY в .env")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    application.bot_data["veo_service"] = VeoService(GEMINI_API_KEY)
    application.bot_data["generation_semaphore"] = asyncio.Semaphore(
        MAX_CONCURRENT_GENERATIONS
    )
    application.bot_data["tasks"] = {}
    application.bot_data["active_modes"] = {}
    application.bot_data["history_store"] = HistoryStore(HISTORY_DB_PATH)

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("settings", settings_command))
    application.add_handler(CommandHandler("history", history_command))
    application.add_handler(CommandHandler("cancel", cancel_command))
    application.add_handler(CallbackQueryHandler(menu_callback, pattern=r"^menu:"))
    application.add_handler(CallbackQueryHandler(history_callback, pattern=r"^history:"))
    application.add_handler(CallbackQueryHandler(flow_callback, pattern=r"^flow:"))
    application.add_handler(CallbackQueryHandler(duration_callback, pattern=r"^duration:"))
    application.add_handler(CallbackQueryHandler(quality_callback, pattern=r"^quality:"))
    application.add_handler(MessageHandler(filters.Regex(r"^☰ Меню$"), bottom_menu_handler))
    application.add_handler(MessageHandler(filters.Regex(r"^⏹ Отмена$"), cancel_command))
    application.add_handler(MessageHandler(filters.PHOTO, photo_prompt))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_prompt))
    application.add_error_handler(error_handler)
    return application


def main() -> None:
    application = build_application()
    logger.info("Telegram-бот запущен; модель: %s", VEO_MODEL)
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    finally:
        service: VeoService | None = application.bot_data.get("veo_service")
        if service is not None:
            service.close()


if __name__ == "__main__":
    main()
