#!/usr/bin/env python3
"""
AskNavidrome production-stabilized Flask/Alexa endpoint.

Purpose of this version:
- Avoid macOS/Python 3.11 multiprocessing startup crashes under launchd.
- Support PR #74-style internal Navidrome URL vs public Alexa stream URL.
- Register both "/" and configured ASKNAVI_PATH routes to work with Caddy
  "handle_path /alexa*" as well as non-stripping reverse proxy configs.
- Add defensive error handling for missing slots, empty queues, failed playback,
  unavailable Navidrome, malformed Alexa requests, and old/new SubsonicConnection
  constructor signatures.
"""

from __future__ import annotations

from datetime import datetime
from threading import Thread, RLock
from types import MethodType
from typing import Any, Iterable, Optional
from urllib.parse import quote
import hashlib
import logging
import os
import random
import secrets
import sys
import traceback

from flask import Flask, jsonify, render_template, request

from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.dispatch_components import (
    AbstractRequestHandler,
    AbstractRequestInterceptor,
    AbstractResponseInterceptor,
    AbstractExceptionHandler,
)
from ask_sdk_core.utils import (
    is_request_type,
    is_intent_name,
    get_slot_value_v2,
    get_request_type,
    get_intent_name,
)
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
from flask_ask_sdk.skill_adapter import SkillAdapter

import asknavidrome.subsonic_api as api
import asknavidrome.media_queue as queue
import asknavidrome.controller as controller


# =============================================================================
# Flask + logging
# =============================================================================

app = Flask(__name__)
sb = SkillBuilder()

logger = logging.getLogger()
logger.handlers.clear()

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# =============================================================================
# Environment/config helpers
# =============================================================================

def env(name: str, required: bool = True, default: Optional[str] = None) -> Optional[str]:
    """Read an environment variable with clear error messages."""
    val = os.getenv(name, default)
    if required and (val is None or str(val).strip() == ""):
        raise RuntimeError(f"Missing env var: {name}")
    return val


def env_int(name: str, required: bool = True, default: Optional[int] = None) -> Optional[int]:
    raw_default = None if default is None else str(default)
    raw = env(name, required=required, default=raw_default)
    if raw is None:
        return None
    try:
        return int(str(raw).strip())
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer env var {name}={raw!r}") from exc


def normalize_path(path: Optional[str], default: str = "/") -> str:
    """Return Flask route path like '/' or '/alexa'."""
    if path is None or str(path).strip() == "":
        return default
    value = str(path).strip()
    if value == "/":
        return "/"
    return "/" + value.strip("/")


def configure_log_level() -> int:
    """
    NAVI_DEBUG:
      0 = WARNING
      1 = INFO
      2 = DEBUG + request/response interceptors
      3 = DEBUG + request/response interceptors + diagnostic web routes
    """
    raw = os.getenv("NAVI_DEBUG", "1")
    try:
        level_num = int(raw)
    except ValueError:
        level_num = 1

    if level_num <= 0:
        logger.setLevel(logging.WARNING)
        logger.warning("Log level set to WARNING")
        return 0

    if level_num == 1:
        logger.setLevel(logging.INFO)
        logger.info("Log level set to INFO")
        return 1

    logger.setLevel(logging.DEBUG)
    logger.debug("Log level set to DEBUG")
    return level_num


navidrome_log_level = configure_log_level()

logger.info("Loading AskNavidrome configuration")

skill_id = env("NAVI_SKILL_ID")
sb.skill_id = skill_id

min_song_count = env_int("NAVI_SONG_COUNT", default=50)

# Internal URL used by app.py to call Navidrome locally.
navidrome_url = env("NAVI_URL")
navidrome_port = str(env_int("NAVI_PORT", default=4533))

# Public URL used in Alexa AudioPlayer stream URLs.
navidrome_url_public = os.getenv("NAVI_URL_PUBLIC", navidrome_url)
navidrome_port_public = str(env_int("NAVI_PORT_PUBLIC", required=False, default=int(navidrome_port)))

navidrome_user = env("NAVI_USER")
navidrome_passwd = env("NAVI_PASS")

navidrome_api_location = env("NAVI_API_PATH", default="/rest")
if not navidrome_api_location.startswith("/"):
    navidrome_api_location = "/" + navidrome_api_location

navidrome_api_version = env("NAVI_API_VER", default="1.16.1")

# If your Caddyfile uses handle_path /alexa*, Caddy strips /alexa before Flask.
# So we register both "/" and this route.
configured_route = normalize_path(os.getenv("ASKNAVI_PATH", "/"), default="/")

logger.info("Skill ID configured")
logger.info(f"AskNavidrome internal Navidrome URL: {navidrome_url}:{navidrome_port}{navidrome_api_location}")
logger.info(f"Alexa public stream URL: {navidrome_url_public}:{navidrome_port_public}{navidrome_api_location}")
logger.info(f"Configured AskNavidrome route: {configured_route}")


# =============================================================================
# General safety helpers
# =============================================================================

def sanitise_speech_output(value: Any) -> str:
    """Sanitize text for Alexa SSML-safe speech."""
    text = "" if value is None else str(value)
    replacements = {
        "&": "and",
        "/": "and",
        "\\": "and",
        '"': "",
        "'": "",
        "<": "",
        ">": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = " ".join(text.split())
    return text or "Sorry, I could not say that."


def safe_speak(handler_input: HandlerInput, speech: str, ask: bool = True) -> Response:
    speech = sanitise_speech_output(speech)
    handler_input.response_builder.speak(speech)
    if ask:
        handler_input.response_builder.ask(speech)
    return handler_input.response_builder.response


def safe_slot(handler_input: HandlerInput, name: str) -> Optional[str]:
    """Safely read a slot value from Alexa request."""
    try:
        slot = get_slot_value_v2(handler_input, name)
    except Exception:
        logger.exception(f"Failed reading slot {name}")
        return None

    if slot is None:
        return None

    value = getattr(slot, "value", slot)
    if value is None:
        return None

    value = str(value).strip()
    return value or None


def first_item(value: Any) -> Optional[Any]:
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, tuple) and value:
        return value[0]
    return None


def is_empty(value: Any) -> bool:
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


def safe_card(text: str) -> dict:
    return {"title": "AskNavidrome", "text": sanitise_speech_output(text)}


def log_exception(context: str, exc: BaseException) -> None:
    logger.error(f"{context}: {exc}")
    logger.debug(traceback.format_exc())


def ensure_two_or_less(song_ids: list[str]) -> list[str]:
    """Return up to first two items without raising on short lists."""
    return song_ids[: min(2, len(song_ids))]


def validate_song_ids(song_ids: Any) -> list[str]:
    if song_ids is None:
        return []
    if not isinstance(song_ids, list):
        try:
            song_ids = list(song_ids)
        except Exception:
            return []
    return [str(x) for x in song_ids if x is not None]


def stop_background_thread_note() -> None:
    # Threads cannot be safely killed in Python. Instead we prevent overlapping loads
    # by refusing to start a second worker while one is still alive.
    pass


# =============================================================================
# Queue system: no multiprocessing under launchd
# =============================================================================

queue_lock = RLock()
play_queue = queue.MediaQueue()
background_thread: Optional[Thread] = None


def queue_worker_thread(connection_obj: object, play_queue_obj: object, song_id_list: list[str]) -> None:
    """Populate remaining queue entries in the background."""
    try:
        logger.debug(f"Background queue worker starting with {len(song_id_list)} songs")
        controller.enqueue_songs(connection_obj, play_queue_obj, song_id_list)
        try:
            play_queue_obj.sync()
        except Exception:
            logger.debug("play_queue.sync() failed or is unnecessary", exc_info=True)
        logger.debug("Background queue worker finished")
    except Exception as exc:
        log_exception("Queue worker failed", exc)


def start_background_queue(song_ids: list[str]) -> None:
    """Start a background thread if one is not already running."""
    global background_thread

    if not song_ids:
        return

    if background_thread is not None and background_thread.is_alive():
        logger.warning("Background queue worker already running; skipping new worker")
        return

    background_thread = Thread(
        target=queue_worker_thread,
        args=(connection, play_queue, song_ids),
        daemon=True,
    )
    background_thread.start()


def clear_and_enqueue(song_ids: list[str], shuffle_first: bool = False) -> Optional[Any]:
    """
    Clear queue, enqueue first tracks synchronously, enqueue rest in background,
    and return next playable track.
    """
    song_ids = validate_song_ids(song_ids)
    if not song_ids:
        return None

    if shuffle_first:
        random.shuffle(song_ids)

    with queue_lock:
        play_queue.clear()
        initial = ensure_two_or_less(song_ids)
        controller.enqueue_songs(connection, play_queue, initial)
        start_background_queue(song_ids[len(initial):])

        try:
            if shuffle_first:
                play_queue.shuffle()
        except Exception:
            logger.debug("Queue shuffle failed", exc_info=True)

        return play_queue.get_next_track()


def get_current_track_safe() -> Optional[Any]:
    try:
        return play_queue.get_current_track()
    except Exception:
        logger.debug("Could not get current track", exc_info=True)
        return None


def get_next_track_safe() -> Optional[Any]:
    try:
        return play_queue.get_next_track()
    except Exception:
        logger.debug("Could not get next track", exc_info=True)
        return None


def get_previous_track_safe() -> Optional[Any]:
    try:
        return play_queue.get_previous_track()
    except Exception:
        logger.debug("Could not get previous track", exc_info=True)
        return None


# =============================================================================
# Navidrome connection with PR #74 compatibility/fallback
# =============================================================================

def public_base_url() -> str:
    """Build public URL base, omitting standard ports."""
    base = str(navidrome_url_public).rstrip("/")
    port = str(navidrome_port_public)

    if (base.startswith("https://") and port == "443") or (base.startswith("http://") and port == "80"):
        return base
    return f"{base}:{port}"


def internal_base_url() -> str:
    base = str(navidrome_url).rstrip("/")
    port = str(navidrome_port)

    if (base.startswith("https://") and port == "443") or (base.startswith("http://") and port == "80"):
        return base
    return f"{base}:{port}"


def patched_public_get_song_uri(self: Any, song_id: str) -> str:
    """
    Force Alexa stream URL to use the public HTTPS URL, even if local
    subsonic_api.py has not yet been updated with PR #74.
    """
    salt = secrets.token_hex(6)
    token = hashlib.md5(f"{self.passwd}{salt}".encode("utf-8")).hexdigest()

    user = quote(str(self.user), safe="")
    song_id_quoted = quote(str(song_id), safe="")
    app_name = quote("AskNavidrome", safe="")

    api_path = getattr(self, "api_location", navidrome_api_location)
    api_ver = getattr(self, "api_version", navidrome_api_version)

    return (
        f"{public_base_url()}{api_path}/stream.view"
        f"?f=json&v={quote(str(api_ver), safe='')}"
        f"&c={app_name}"
        f"&u={user}"
        f"&s={salt}"
        f"&t={token}"
        f"&id={song_id_quoted}"
    )


def create_connection() -> Any:
    """
    Supports both:
    - PR #74 constructor:
      SubsonicConnection(server_url, port, public_url, public_port, user, passwd, api_location, api_version)
    - upstream constructor:
      SubsonicConnection(server_url, user, passwd, port, api_location, api_version)
    """
    try:
        logger.info("Trying PR #74-style SubsonicConnection constructor")
        conn = api.SubsonicConnection(
            navidrome_url,
            navidrome_port,
            navidrome_url_public,
            navidrome_port_public,
            navidrome_user,
            navidrome_passwd,
            navidrome_api_location,
            navidrome_api_version,
        )
    except TypeError:
        logger.warning("PR #74 constructor failed; falling back to upstream SubsonicConnection constructor")
        conn = api.SubsonicConnection(
            navidrome_url,
            navidrome_user,
            navidrome_passwd,
            navidrome_port,
            navidrome_api_location,
            navidrome_api_version,
        )

    # Ensure attributes used by patched get_song_uri exist.
    try:
        conn.server_url = navidrome_url
        conn.port = navidrome_port
        conn.public_url = navidrome_url_public
        conn.public_port = navidrome_port_public
        conn.user = navidrome_user
        conn.passwd = navidrome_passwd
        conn.api_location = navidrome_api_location
        conn.api_version = navidrome_api_version
        conn.get_song_uri = MethodType(patched_public_get_song_uri, conn)
        logger.info("Forced public HTTPS stream URL generation is enabled")
    except Exception:
        logger.exception("Failed to attach public stream URL patch")

    return conn


connection = create_connection()

try:
    connection.ping()
    logger.info("Connected to Navidrome successfully")
except Exception as exc:
    log_exception("Could not connect to Navidrome/Subsonic API", exc)
    raise RuntimeError(
        f"Could not connect to Navidrome at {internal_base_url()}{navidrome_api_location}. "
        f"Check NAVI_URL, NAVI_PORT, NAVI_USER, NAVI_PASS, NAVI_API_PATH, NAVI_API_VER."
    ) from exc


# =============================================================================
# Playback helper
# =============================================================================

def start_track_or_speak(
    handler_input: HandlerInput,
    speech: str,
    track: Optional[Any],
    card: Optional[dict] = None,
) -> Response:
    if track is None:
        return safe_speak(handler_input, "I found the request, but there were no playable tracks.")

    try:
        return controller.start_playback("play", sanitise_speech_output(speech), card, track, handler_input)
    except Exception as exc:
        log_exception("controller.start_playback failed", exc)
        return safe_speak(handler_input, "I found the music, but I could not start playback.")


# =============================================================================
# Request handlers
# =============================================================================

class LaunchRequestHandler(AbstractRequestHandler):
    """Handle LaunchRequest and NavigateHomeIntent."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_request_type("LaunchRequest")(handler_input)
            or is_intent_name("AMAZON.NavigateHomeIntent")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.debug("In LaunchRequestHandler")
        try:
            connection.ping()
            return safe_speak(handler_input, "Ready!")
        except Exception as exc:
            log_exception("Launch ping failed", exc)
            return safe_speak(handler_input, "Navidrome is unavailable.")


class CheckAudioInterfaceHandler(AbstractRequestHandler):
    """Reject unsupported devices without AudioPlayer support."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        try:
            device = handler_input.request_envelope.context.system.device
            if device is None:
                return False
            return device.supported_interfaces.audio_player is None
        except Exception:
            return False

    def handle(self, handler_input: HandlerInput) -> Response:
        return safe_speak(handler_input, "This device does not support audio playback.", ask=False)


class SkillEventHandler(AbstractRequestHandler):
    """Close session for skill events and session ended requests."""

    def can_handle(self, handler_input: HandlerInput) -> bool:
        try:
            obj_type = handler_input.request_envelope.request.object_type or ""
            return obj_type.startswith("AlexaSkillEvent") or is_request_type("SessionEndedRequest")(handler_input)
        except Exception:
            return False

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.debug("In SkillEventHandler")
        return handler_input.response_builder.response


class HelpHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        return safe_speak(
            handler_input,
            "AskNavidrome lets you play music from your Navidrome library. "
            "Try saying, play random music, or play music by an artist.",
        )


class NaviSonicPlayMusicByArtist(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayMusicByArtist")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            artist = safe_slot(handler_input, "artist")
            if not artist:
                return safe_speak(handler_input, "I didn't catch the artist name.")

            artist_lookup = connection.search_artist(artist)
            artist_obj = first_item(artist_lookup)

            if not artist_obj:
                return safe_speak(handler_input, f"I couldn't find the artist {artist} in the collection.")

            albums = connection.albums_by_artist(artist_obj.get("id"))
            song_ids = validate_song_ids(connection.build_song_list_from_albums(albums, min_song_count))

            if not song_ids:
                return safe_speak(handler_input, f"I couldn't find playable songs by {artist}.")

            track = clear_and_enqueue(song_ids, shuffle_first=True)
            speech = f"Playing music by {artist}"
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayMusicByArtist failed", exc)
            return safe_speak(handler_input, "Something went wrong playing that artist.")


class NaviSonicPlayAlbumByArtist(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayAlbumByArtist")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            artist = safe_slot(handler_input, "artist")
            album = safe_slot(handler_input, "album")

            if not album:
                return safe_speak(handler_input, "I didn't catch the album name.")

            if artist:
                artist_obj = first_item(connection.search_artist(artist))
                if not artist_obj:
                    return safe_speak(handler_input, f"I couldn't find the artist {artist}.")

                albums = connection.albums_by_artist(artist_obj.get("id")) or []
                matches = [
                    item for item in albums
                    if str(item.get("name", "")).lower() == album.lower()
                ]

                if not matches:
                    return safe_speak(handler_input, f"I couldn't find {album} by {artist}.")

                song_ids = validate_song_ids(connection.build_song_list_from_albums(matches, -1))
                speech = f"Playing {album} by {artist}"

            else:
                albums = connection.search_album(album)
                if not albums:
                    return safe_speak(handler_input, f"I couldn't find the album {album}.")
                song_ids = validate_song_ids(connection.build_song_list_from_albums(albums, -1))
                speech = f"Playing {album}"

            if not song_ids:
                return safe_speak(handler_input, f"I found {album}, but there were no playable tracks.")

            track = clear_and_enqueue(song_ids)
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayAlbumByArtist failed", exc)
            return safe_speak(handler_input, "Something went wrong playing that album.")


class NaviSonicPlaySongByArtist(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlaySongByArtist")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            artist = safe_slot(handler_input, "artist")
            song = safe_slot(handler_input, "song")

            if not song:
                return safe_speak(handler_input, "I didn't catch the song name.")

            songs = connection.search_song(song) or []

            if artist:
                artist_obj = first_item(connection.search_artist(artist))
                if not artist_obj:
                    return safe_speak(handler_input, f"I couldn't find the artist {artist}.")
                artist_id = artist_obj.get("id")
                song_ids = [
                    item.get("id") for item in songs
                    if item and item.get("artistId") == artist_id
                ]
                speech = f"Playing {song} by {artist}"
            else:
                song_ids = [item.get("id") for item in songs if item and item.get("id")]
                speech = f"Playing {song}"

            song_ids = validate_song_ids(song_ids)

            if not song_ids:
                if artist:
                    return safe_speak(handler_input, f"I couldn't find {song} by {artist}.")
                return safe_speak(handler_input, f"I couldn't find {song}.")

            track = clear_and_enqueue(song_ids)
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlaySongByArtist failed", exc)
            return safe_speak(handler_input, "Something went wrong playing that song.")


class NaviSonicPlayPlaylist(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayPlaylist")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            playlist = safe_slot(handler_input, "playlist")
            if not playlist:
                return safe_speak(handler_input, "I didn't catch the playlist name.")

            playlist_id = connection.search_playlist(playlist)
            if not playlist_id:
                return safe_speak(handler_input, f"I couldn't find the playlist {playlist}.")

            song_ids = validate_song_ids(connection.build_song_list_from_playlist(playlist_id))
            if not song_ids:
                return safe_speak(handler_input, f"The playlist {playlist} has no playable tracks.")

            # Shuffle playlist tracks before queueing them for Alexa playback.
            track = clear_and_enqueue(song_ids, shuffle_first=True)

            speech = f"Shuffling playlist {playlist}"
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayPlaylist failed", exc)
            return safe_speak(handler_input, "Something went wrong playing that playlist.")


class NaviSonicPlayMusicByGenre(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayMusicByGenre")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            genre = safe_slot(handler_input, "genre")
            if not genre:
                return safe_speak(handler_input, "I didn't catch the genre.")

            song_ids = validate_song_ids(connection.build_song_list_from_genre(genre, min_song_count))
            if not song_ids:
                return safe_speak(handler_input, f"I couldn't find any {genre} songs.")

            track = clear_and_enqueue(song_ids, shuffle_first=True)
            speech = f"Playing {genre} music"
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayMusicByGenre failed", exc)
            return safe_speak(handler_input, "Something went wrong playing that genre.")


class NaviSonicPlayMusicRandom(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayMusicRandom")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            song_ids = validate_song_ids(connection.build_random_song_list(min_song_count))
            if not song_ids:
                return safe_speak(handler_input, "I couldn't find any songs in the collection.")

            track = clear_and_enqueue(song_ids, shuffle_first=True)
            speech = "Playing random music"
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayMusicRandom failed", exc)
            return safe_speak(handler_input, "Something went wrong playing random music.")


class NaviSonicPlayFavouriteSongs(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicPlayFavouriteSongs")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            song_ids = validate_song_ids(connection.build_song_list_from_favourites())
            if not song_ids:
                return safe_speak(handler_input, "You don't have any favourite songs in the collection.")

            track = clear_and_enqueue(song_ids, shuffle_first=True)
            speech = "Playing your favourite tracks"
            logger.info(speech)
            return start_track_or_speak(handler_input, speech, track, safe_card(speech))

        except Exception as exc:
            log_exception("NaviSonicPlayFavouriteSongs failed", exc)
            return safe_speak(handler_input, "Something went wrong playing your favourite songs.")


class NaviSonicRandomiseQueue(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicRandomiseQueue")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            play_queue.shuffle()
            try:
                play_queue.sync()
            except Exception:
                logger.debug("Queue sync failed after shuffle", exc_info=True)
            return safe_speak(handler_input, "Queue shuffled.", ask=False)
        except Exception as exc:
            log_exception("NaviSonicRandomiseQueue failed", exc)
            return safe_speak(handler_input, "I could not shuffle the queue.")


class NaviSonicSongDetails(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicSongDetails")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            current = get_current_track_safe()
            if not current:
                return safe_speak(handler_input, "Nothing is currently playing.")

            title = sanitise_speech_output(getattr(current, "title", "this track"))
            artist = sanitise_speech_output(getattr(current, "artist", "unknown artist"))
            album = sanitise_speech_output(getattr(current, "album", "unknown album"))
            return safe_speak(handler_input, f"This is {title} by {artist}, from the album {album}.", ask=False)

        except Exception as exc:
            log_exception("NaviSonicSongDetails failed", exc)
            return safe_speak(handler_input, "I could not get the current song details.")


class NaviSonicStarSong(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicStarSong")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            current = get_current_track_safe()
            if not current:
                return safe_speak(handler_input, "Nothing is currently playing.")
            connection.star_entry(current.id, "song")
            return safe_speak(handler_input, "Song starred.", ask=False)
        except Exception as exc:
            log_exception("NaviSonicStarSong failed", exc)
            return safe_speak(handler_input, "I could not star that song.")


class NaviSonicUnstarSong(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("NaviSonicUnstarSong")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            current = get_current_track_safe()
            if not current:
                return safe_speak(handler_input, "Nothing is currently playing.")
            connection.unstar_entry(current.id, "song")
            return safe_speak(handler_input, "Song unstarred.", ask=False)
        except Exception as exc:
            log_exception("NaviSonicUnstarSong failed", exc)
            return safe_speak(handler_input, "I could not unstar that song.")


# =============================================================================
# AudioPlayer handlers
# =============================================================================

class PlaybackStartedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("AudioPlayer.PlaybackStarted")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        logger.info("Playback started")
        return handler_input.response_builder.response


class PlaybackStoppedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("AudioPlayer.PlaybackStopped")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            offset = getattr(handler_input.request_envelope.request, "offset_in_milliseconds", 0)
            play_queue.set_current_track_offset(offset)
            current = get_current_track_safe()
            if current:
                logger.info(f"Playback stopped at {offset} ms for {getattr(current, 'title', 'unknown track')}")
            try:
                play_queue.sync()
            except Exception:
                logger.debug("Queue sync failed on stop", exc_info=True)
        except Exception as exc:
            log_exception("PlaybackStoppedHandler failed", exc)
        return handler_input.response_builder.response


class PlaybackNearlyFinishedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("AudioPlayer.PlaybackNearlyFinished")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            logger.info("Playback nearly finished; queueing next track")
            track = play_queue.enqueue_next_track()
            if not track:
                logger.warning("No next track available")
                return handler_input.response_builder.response
            return controller.start_playback("continue", None, None, track, handler_input)
        except Exception as exc:
            log_exception("PlaybackNearlyFinishedHandler failed", exc)
            return handler_input.response_builder.response


class PlaybackFinishedHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("AudioPlayer.PlaybackFinished")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            current = get_current_track_safe()
            if current:
                try:
                    connection.scrobble(current.id, datetime.now().timestamp())
                except Exception:
                    logger.debug("Scrobble failed", exc_info=True)
            get_next_track_safe()
        except Exception as exc:
            log_exception("PlaybackFinishedHandler failed", exc)
        return handler_input.response_builder.response


class PausePlaybackHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_intent_name("AMAZON.StopIntent")(handler_input)
            or is_intent_name("AMAZON.CancelIntent")(handler_input)
            or is_intent_name("AMAZON.PauseIntent")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            play_queue.sync()
        except Exception:
            logger.debug("Queue sync failed before stop", exc_info=True)

        try:
            return controller.stop(handler_input)
        except Exception as exc:
            log_exception("controller.stop failed", exc)
            return handler_input.response_builder.response


class ResumePlaybackHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_intent_name("AMAZON.ResumeIntent")(handler_input) or is_intent_name("PlayAudio")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            current = get_current_track_safe()

            if current and getattr(current, "offset", 0) > 0:
                logger.info(f"Resuming {getattr(current, 'title', 'track')} from offset {current.offset}")
                return controller.start_playback("play", None, None, current, handler_input)

            queue_count = 0
            try:
                queue_count = play_queue.get_queue_count()
            except Exception:
                logger.debug("Could not get queue count", exc_info=True)

            if queue_count > 0:
                track = get_next_track_safe()
                if track:
                    return controller.start_playback("play", None, None, track, handler_input)

            return safe_speak(handler_input, "There is nothing to resume.")

        except Exception as exc:
            log_exception("ResumePlaybackHandler failed", exc)
            return safe_speak(handler_input, "I could not resume playback.")


class NextPlaybackHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_intent_name("AMAZON.NextIntent")(handler_input)
            or is_request_type("PlaybackController.NextCommandIssued")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            track = get_next_track_safe()
            if not track:
                return safe_speak(handler_input, "There is no next track.")
            track.offset = 0
            return controller.start_playback("play", None, None, track, handler_input)
        except Exception as exc:
            log_exception("NextPlaybackHandler failed", exc)
            return safe_speak(handler_input, "I could not skip to the next track.")


class PreviousPlaybackHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return (
            is_intent_name("AMAZON.PreviousIntent")(handler_input)
            or is_request_type("PlaybackController.PreviousCommandIssued")(handler_input)
        )

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            track = get_previous_track_safe()
            if not track:
                return safe_speak(handler_input, "There is no previous track.")
            track.offset = 0
            return controller.start_playback("play", None, None, track, handler_input)
        except Exception as exc:
            log_exception("PreviousPlaybackHandler failed", exc)
            return safe_speak(handler_input, "I could not go to the previous track.")


class PlaybackFailedEventHandler(AbstractRequestHandler):
    def can_handle(self, handler_input: HandlerInput) -> bool:
        return is_request_type("AudioPlayer.PlaybackFailed")(handler_input)

    def handle(self, handler_input: HandlerInput) -> Response:
        try:
            err = getattr(handler_input.request_envelope.request, "error", None)
            logger.error(f"Alexa PlaybackFailed error: {err}")

            current = get_current_track_safe()
            if current:
                logger.error(f"Failed track id: {getattr(current, 'id', 'unknown')}")

            track = get_next_track_safe()
            if not track:
                logger.warning("No next track available after playback failure")
                return handler_input.response_builder.response

            track.offset = 0
            return controller.start_playback("play", None, None, track, handler_input)

        except Exception as exc:
            log_exception("PlaybackFailedEventHandler failed", exc)
            return handler_input.response_builder.response


# =============================================================================
# Exception handlers and interceptors
# =============================================================================

class SystemExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        try:
            return is_request_type("System.ExceptionEncountered")(handler_input)
        except Exception:
            return False

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        log_exception("System.ExceptionEncountered", exception)
        try:
            req = handler_input.request_envelope.request
            logger.error(f"System exception request: {req}")
        except Exception:
            pass
        return handler_input.response_builder.response


class GeneralExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input: HandlerInput, exception: Exception) -> bool:
        return True

    def handle(self, handler_input: HandlerInput, exception: Exception) -> Response:
        log_exception("General Alexa exception", exception)
        try:
            logger.error(f"Request type was: {get_request_type(handler_input)}")
            if get_request_type(handler_input) == "IntentRequest":
                logger.error(f"Intent name was: {get_intent_name(handler_input)}")
        except Exception:
            pass

        try:
            return safe_speak(handler_input, "Sorry, something went wrong.")
        except Exception:
            return handler_input.response_builder.response


class LoggingRequestInterceptor(AbstractRequestInterceptor):
    def process(self, handler_input: HandlerInput) -> None:
        try:
            logger.debug(f"Request received: {handler_input.request_envelope.request}")
        except Exception:
            logger.debug("Request received but could not be logged")


class LoggingResponseInterceptor(AbstractResponseInterceptor):
    def process(self, handler_input: HandlerInput, response: Response) -> None:
        try:
            logger.debug(f"Response sent: {response}")
        except Exception:
            logger.debug("Response sent but could not be logged")


# =============================================================================
# Register handlers
# =============================================================================

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(CheckAudioInterfaceHandler())
sb.add_request_handler(SkillEventHandler())
sb.add_request_handler(HelpHandler())

sb.add_request_handler(NaviSonicPlayMusicByArtist())
sb.add_request_handler(NaviSonicPlayAlbumByArtist())
sb.add_request_handler(NaviSonicPlaySongByArtist())
sb.add_request_handler(NaviSonicPlayPlaylist())
sb.add_request_handler(NaviSonicPlayFavouriteSongs())
sb.add_request_handler(NaviSonicPlayMusicByGenre())
sb.add_request_handler(NaviSonicPlayMusicRandom())
sb.add_request_handler(NaviSonicRandomiseQueue())
sb.add_request_handler(NaviSonicSongDetails())
sb.add_request_handler(NaviSonicStarSong())
sb.add_request_handler(NaviSonicUnstarSong())

sb.add_request_handler(PlaybackStartedHandler())
sb.add_request_handler(PlaybackStoppedHandler())
sb.add_request_handler(PlaybackNearlyFinishedHandler())
sb.add_request_handler(PlaybackFinishedHandler())
sb.add_request_handler(PausePlaybackHandler())
sb.add_request_handler(NextPlaybackHandler())
sb.add_request_handler(PreviousPlaybackHandler())
sb.add_request_handler(ResumePlaybackHandler())
sb.add_request_handler(PlaybackFailedEventHandler())

sb.add_exception_handler(SystemExceptionHandler())
sb.add_exception_handler(GeneralExceptionHandler())

if navidrome_log_level >= 2:
    sb.add_global_request_interceptor(LoggingRequestInterceptor())
    sb.add_global_response_interceptor(LoggingResponseInterceptor())


# =============================================================================
# Flask routes
# =============================================================================

@app.route("/health", methods=["GET"])
def health() -> Any:
    status = {
        "service": "AskNavidrome",
        "ok": True,
        "route": configured_route,
        "internal_navidrome": f"{internal_base_url()}{navidrome_api_location}",
        "public_stream_base": f"{public_base_url()}{navidrome_api_location}",
    }

    try:
        connection.ping()
        status["navidrome_ping"] = "ok"
        http_status = 200
    except Exception as exc:
        status["ok"] = False
        status["navidrome_ping"] = f"failed: {exc}"
        http_status = 503

    return jsonify(status), http_status


if navidrome_log_level == 3:
    logger.warning("AskNavidrome debugging is enabled. Diagnostic web endpoints are available.")

    @app.route("/queue", methods=["GET"])
    def view_queue() -> Any:
        current = get_current_track_safe()
        try:
            return render_template(
                "table.html",
                title="AskNavidrome - Queued Tracks",
                tracks=play_queue.get_current_queue(),
                current=current,
            )
        except Exception:
            return jsonify({"error": "Could not render queue", "current": str(current)}), 500

    @app.route("/history", methods=["GET"])
    def view_history() -> Any:
        current = get_current_track_safe()
        try:
            return render_template(
                "table.html",
                title="AskNavidrome - Track History",
                tracks=play_queue.get_history(),
                current=current,
            )
        except Exception:
            return jsonify({"error": "Could not render history", "current": str(current)}), 500

    @app.route("/buffer", methods=["GET"])
    def view_buffer() -> Any:
        current = get_current_track_safe()
        try:
            return render_template(
                "table.html",
                title="AskNavidrome - Buffered Tracks",
                tracks=play_queue.get_buffer(),
                current=current,
            )
        except Exception:
            return jsonify({"error": "Could not render buffer", "current": str(current)}), 500


# Register Alexa adapter on both "/" and configured path.
# "/" is required when Caddy uses "handle_path /alexa*" because Caddy strips /alexa.
skill = sb.create()
sa = SkillAdapter(skill=skill, skill_id=skill_id, app=app)

registered_routes = set()
for route in {"/", configured_route}:
    if route not in registered_routes:
        logger.info(f"Registering Alexa SkillAdapter route: {route}")
        sa.register(app=app, route=route)
        registered_routes.add(route)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    host = os.getenv("ASKNAVI_HOST", "0.0.0.0")
    port = env_int("ASKNAVI_PORT", required=False, default=5001)
    logger.info(f"Starting AskNavidrome Flask app on {host}:{port}")
    app.run(host=host, port=port)
