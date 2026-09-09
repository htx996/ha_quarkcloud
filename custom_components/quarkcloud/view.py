"""HTTP view that streams drive files through Home Assistant.

Quark download urls require an auth cookie that browsers and casting
devices cannot attach themselves, so media playback goes through this
authenticated proxy view (same pattern as the Plex image proxy view in
HA core). Range headers are forwarded so video seeking works.

The fid and expected mime type arrive as base64url path segments (not
query params): HA only signs paths without a query string
(``async_process_play_media_url``), and the signature is what lets
browser/casting players fetch the stream without a Bearer header.
"""

from __future__ import annotations

import base64
import logging
from http import HTTPStatus
from typing import Any

from aiohttp import ClientError, ClientTimeout, web

from homeassistant.components.http import KEY_HASS, HomeAssistantView
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession

from .api import QuarkApiError
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# No total timeout (long media streams are legitimate), but a stalled
# upstream connection must error out instead of spinning forever.
_UPSTREAM_TIMEOUT = ClientTimeout(total=None, connect=10, sock_read=120)


def _unb64(value: str) -> str:
    """Decode an unpadded URL-safe base64 path segment."""
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)).decode()


class QuarkMediaStreamView(HomeAssistantView):
    """Stream a Quark drive file through an authenticated proxy."""

    url = "/api/quarkcloud/media/{b64fid}/{b64mime}"
    name = "api:quarkcloud:media"
    requires_auth = True

    async def get(
        self, request: web.Request, b64fid: str, b64mime: str
    ) -> web.StreamResponse:
        """Stream the requested file."""
        hass: HomeAssistant = request.app[KEY_HASS]
        fid = _unb64(b64fid)
        mime = _unb64(b64mime)
        entries = hass.config_entries.async_loaded_entries(DOMAIN)
        if not entries:
            return web.Response(status=HTTPStatus.SERVICE_UNAVAILABLE)
        api = entries[0].runtime_data

        try:
            info: dict[str, Any] = await api.get_download_url(fid)
        except QuarkApiError as err:
            # 23018: file exceeds the 50MB open-API download limit.
            _LOGGER.debug("Media stream: no download url for %s: %s", fid, err)
            status = (
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE
                if "23018" in str(err)
                else HTTPStatus.BAD_GATEWAY
            )
            return web.Response(status=status)

        urls = info.get("download_url") or info.get("download_urls") or []
        if isinstance(urls, str):
            urls = [urls]

        session = async_get_clientsession(hass)
        forward_headers: dict[str, str] = {"Cookie": api.download_cookie_header()}
        if rng := request.headers.get("Range"):
            forward_headers["Range"] = rng

        upstream = None
        for url in urls:
            try:
                upstream = await session.get(
                    url,
                    headers=forward_headers,
                    timeout=_UPSTREAM_TIMEOUT,
                )
            except (ClientError, TimeoutError) as err:
                _LOGGER.warning("Media stream: upstream GET failed: %s", err)
                continue
            if upstream.status in (HTTPStatus.OK, HTTPStatus.PARTIAL_CONTENT):
                break
            _LOGGER.debug(
                "Media stream: upstream %s for %s -> HTTP %s",
                url[:60],
                fid,
                upstream.status,
            )
            await upstream.release()
            upstream = None
        if upstream is None:
            return web.Response(status=HTTPStatus.BAD_GATEWAY)

        content_type = upstream.headers.get("Content-Type") or mime
        if content_type in ("", "application/octet-stream", "binary/octet-stream"):
            content_type = mime
        response = web.StreamResponse(status=upstream.status)
        response.headers["Content-Type"] = content_type
        for header in ("Content-Length", "Content-Range", "Accept-Ranges"):
            if (value := upstream.headers.get(header)) is not None:
                response.headers[header] = value
        response.headers.setdefault("Accept-Ranges", "bytes")
        _LOGGER.debug(
            "Media stream: serving %s (upstream HTTP %s, type=%s, len=%s, range=%s)",
            fid,
            upstream.status,
            content_type,
            upstream.headers.get("Content-Length"),
            bool(request.headers.get("Range")),
        )
        try:
            await response.prepare(request)
            async for chunk in upstream.content.iter_chunked(65536):
                await response.write(chunk)
            await response.write_eof()
        finally:
            upstream.release()
        return response
