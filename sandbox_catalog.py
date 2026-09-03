"""Read the official sandbox camera catalogue without hard-coding stream URLs."""

from __future__ import annotations

import argparse
import json
from urllib.request import Request, urlopen


def fetch_catalog(url: str, timeout: float = 10.0) -> list[dict]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "Gujarat-Prahari-AI/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    cameras = payload.get("cameras", payload) if isinstance(payload, dict) else payload
    if not isinstance(cameras, list):
        raise ValueError("Catalogue response must contain a camera list")
    return cameras


def select_camera(cameras: list[dict], camera_id: str) -> dict:
    wanted = camera_id.casefold()
    for camera in cameras:
        identifier = str(camera.get("id", camera.get("camera_id", "")))
        if identifier.casefold() == wanted:
            return camera
    raise KeyError(f"Camera {camera_id!r} not present in current catalogue")


def rtsp_url(camera: dict) -> str:
    for key in ("rtsp_url", "rtsp", "url_rtsp"):
        if camera.get(key):
            return str(camera[key])
    urls = camera.get("urls", {})
    if isinstance(urls, dict) and urls.get("rtsp"):
        return str(urls["rtsp"])
    raise KeyError("Selected camera has no RTSP URL in the catalogue")


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover cameras from the Sentinel sandbox catalogue")
    parser.add_argument("--catalog", required=True, help="Catalogue URL ending in /api/ingest")
    parser.add_argument("--camera", help="Camera id; omit to print safe camera metadata")
    args = parser.parse_args()
    cameras = fetch_catalog(args.catalog)
    if not args.camera:
        safe = [{k: v for k, v in item.items() if "url" not in k.lower()} for item in cameras]
        print(json.dumps(safe, indent=2))
        return
    print(rtsp_url(select_camera(cameras, args.camera)))


if __name__ == "__main__":
    main()
