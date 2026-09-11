"""Media source exposing the Quark Cloud Drive in the HA media browser.

Implementation follows the official media source platform contract
(``media_source.py`` + ``async_get_media_source``). Browsing maps drive
folders to directory items; resolving hands out the streaming proxy URL
(``view.py``) because Quark download urls need an auth cookie. Files
above the open-API 50MB download limit are marked not playable.
"""

from __future__ import annotations

import base64
import inspect
import json
import logging
import mimetypes
from typing import Any

from homeassistant.components.media_player import (
    BrowseError,
    BrowseMedia,
    MediaClass,
    SearchMedia,
    SearchMediaQuery,
)
from homeassistant.components.media_source import (
    BrowseMediaSource,
    MediaSource,
    MediaSourceItem,
    PlayMedia,
    Unresolvable,
)
from homeassistant.core import HomeAssistant

from . import QuarkCloudConfigEntry
from .api import QuarkApiError, QuarkCloudApi
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# The Quark open API refuses get_download_url for files above 50MB
# (errno 23018), so larger files are marked not playable.
MAX_PLAYABLE_BYTES = 50 * 1024 * 1024

_PAGE_SIZE = 100  # file/list server limit: 1-100 per page
_MAX_CHILDREN = 1000

_MEDIA_CLASSES: tuple[MediaClass, ...] = (
    MediaClass.IMAGE,
    MediaClass.VIDEO,
    MediaClass.MUSIC,
)


def _supports_search_filters() -> bool:
    """Whether this HA core accepts search_media_classes on BrowseMedia.

    The search-filter dropdown is a recent core addition; passing it to
    an older BrowseMedia raises TypeError and breaks browsing entirely,
    so probe the constructor signature once at import time.
    """
    try:
        return (
            "search_media_classes"
            in inspect.signature(BrowseMedia.__init__).parameters
        )
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


SEARCH_FILTERS_SUPPORTED = _supports_search_filters()


async def async_get_media_source(hass: HomeAssistant) -> QuarkCloudMediaSource:
    """Set up the Quark Cloud Drive media source."""
    return QuarkCloudMediaSource(hass)


def _encode_item(fid: str, name: str, size: int, is_dir: bool) -> str:
    """Pack item metadata into an opaque URL-safe identifier.

    Quark FIDs and file names may contain arbitrary separators (FIDs
    carry ``|``), so a small JSON blob is more robust than delimiters.
    """
    raw = json.dumps([fid, name, size, is_dir], separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_item(identifier: str) -> tuple[str, str, int, bool]:
    """Unpack an identifier; raises BrowseError for unknown items."""
    try:
        padded = identifier + "=" * (-len(identifier) % 4)
        fid, name, size, is_dir = json.loads(base64.urlsafe_b64decode(padded))
        return str(fid), str(name), int(size), bool(is_dir)
    except (TypeError, ValueError) as err:
        raise BrowseError("Unknown media item") from err


def _b64(value: str) -> str:
    """Encode a string as unpadded URL-safe base64 for a path segment."""
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def _media_class_and_type(name: str) -> tuple[MediaClass, str] | None:
    """Map a filename to (media class, mime type); None for non-media."""
    mime, _ = mimetypes.guess_type(name)
    if mime is None:
        return None
    if mime.startswith("image/"):
        return MediaClass.IMAGE, mime
    if mime.startswith("video/"):
        return MediaClass.VIDEO, mime
    if mime.startswith("audio/"):
        return MediaClass.MUSIC, mime
    return None


class QuarkCloudMediaSource(MediaSource):
    """Represent the Quark Cloud Drive as a browsable media source."""

    name = "Quark Cloud Drive"

    def __init__(self, hass: HomeAssistant) -> None:
        """Initialize the media source."""
        super().__init__(DOMAIN)
        self.hass = hass

    def _api(self) -> QuarkCloudApi:
        """Return the API client of the first loaded config entry."""
        entries = self.hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            raise BrowseError("Quark Cloud Drive integration is not set up")
        entry: QuarkCloudConfigEntry = entries[0]
        return entry.runtime_data

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        """Return the drive root or the children of a folder."""
        api = self._api()
        try:
            if not item.identifier:
                return await self._async_browse_dir(api, "0", self.name, root=True)
            fid, name, _size, is_dir = _decode_item(item.identifier)
            if not is_dir:
                raise BrowseError("Item is not a folder")
            return await self._async_browse_dir(api, fid, name)
        except QuarkApiError as err:
            raise BrowseError(str(err)) from err

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        """Resolve a file to the authenticated streaming proxy URL.

        The fid and the expected mime type travel as base64url PATH
        segments: HA only signs paths without query params
        (media_player/browse_media.py skips authSig when a query string
        is present), and signed URLs are what let browser/casting
        players fetch the stream without a Bearer header.
        """
        fid, name, size, is_dir = _decode_item(item.identifier)
        if is_dir:
            raise Unresolvable("Folders are not playable")
        if size > MAX_PLAYABLE_BYTES:
            raise Unresolvable(
                "File exceeds the 50MB Quark open-API download limit"
            )
        media = _media_class_and_type(name)
        mime = media[1] if media else "application/octet-stream"
        return PlayMedia(
            f"/api/quarkcloud/media/{_b64(fid)}/{_b64(mime)}", mime
        )

    async def async_search_media(
        self, item: MediaSourceItem, query: SearchMediaQuery
    ) -> SearchMedia:
        """Search the whole drive with the query string."""
        api = self._api()
        try:
            result = await api.search_files(query.search_query, size=100)
        except QuarkApiError as err:
            raise BrowseError(str(err)) from err
        children: list[BrowseMediaSource] = []
        for item_info in result.get("file_list") or []:
            child = self._build_child(item_info)
            if child is None or not child.can_play:
                continue
            if (
                query.media_filter_classes
                and child.media_class not in query.media_filter_classes
            ):
                continue
            children.append(child)
        return SearchMedia(result=children)

    async def _async_browse_dir(
        self,
        api: QuarkCloudApi,
        parent_fid: str,
        title: str,
        root: bool = False,
    ) -> BrowseMediaSource:
        """Build the browse tree for one drive directory."""
        children: list[BrowseMediaSource] = []
        not_shown = 0
        for item_info in await self._async_list_dir(api, parent_fid):
            child = self._build_child(item_info)
            if child is None:
                not_shown += 1
            else:
                children.append(child)
        # search_media_classes only exists on newer HA cores; on older
        # ones the search bar (can_search) still works without filters.
        # The key must be omitted entirely - passing None would still
        # reach BrowseMedia.__init__ and crash older cores.
        node = BrowseMediaSource(
            domain=DOMAIN,
            identifier=None,
            media_class=MediaClass.DIRECTORY,
            media_content_type="",
            title=title,
            can_play=False,
            can_expand=True,
            can_search=root,
            children=children,
            children_media_class=MediaClass.DIRECTORY,
            not_shown=not_shown,
        )
        if root and SEARCH_FILTERS_SUPPORTED:
            node.search_media_classes = list(_MEDIA_CLASSES)
        return node

    async def _async_list_dir(
        self, api: QuarkCloudApi, parent_fid: str
    ) -> list[dict[str, Any]]:
        """List a directory, following the cursor up to a safety cap."""
        items: list[dict[str, Any]] = []
        # The server returns next_query_cursor as an object
        # ({"version", "token"}) and expects the same object back on the
        # next page - round-trip it untouched (no str()).
        cursor: dict[str, Any] | str | None = None
        while len(items) < _MAX_CHILDREN:
            data = await api.list_files(
                parent_fid=parent_fid, size=_PAGE_SIZE, cursor=cursor
            )
            page = data.get("file_list") or []
            items.extend(page)
            if data.get("last_page", True) or not page:
                break
            next_cursor = data.get("next_query_cursor")
            if not next_cursor:
                break
            cursor = next_cursor
        return items[:_MAX_CHILDREN]

    def _build_child(self, item: dict[str, Any]) -> BrowseMediaSource | None:
        """Build a browse item; None hides the entry (non-media files)."""
        fid = str(item.get("fid") or "")
        name = str(item.get("filename") or "")
        if not fid or not name:
            return None
        try:
            size = int(item.get("size") or 0)
        except (TypeError, ValueError):
            size = 0
        is_dir = str(item.get("file_type")) == "0" or str(item.get("category")) == "0"
        identifier = _encode_item(fid, name, size, is_dir)
        if is_dir:
            return BrowseMediaSource(
                domain=DOMAIN,
                identifier=identifier,
                media_class=MediaClass.DIRECTORY,
                media_content_type="",
                title=name,
                can_play=False,
                can_expand=True,
            )
        media = _media_class_and_type(name)
        if media is None:
            return None
        media_class, mime = media
        return BrowseMediaSource(
            domain=DOMAIN,
            identifier=identifier,
            media_class=media_class,
            media_content_type=mime,
            title=name,
            can_play=size <= MAX_PLAYABLE_BYTES,
            can_expand=False,
        )
