"""Route lookups (walking, transit, driving) backed by the Amap web API."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

AMAP_BASE_URL = "https://restapi.amap.com/v3"
AMAP_KEY_ENV_VAR = "AMAP_KEY"
AMAP_STATUS_OK = "1"
SECONDS_PER_MINUTE = 60
DEFAULT_CITY_CODE = "110000"


def _amap_key() -> str:
    """Return the Amap API key from the environment.

    Returns:
        The API key read from the ``AMAP_KEY`` environment variable.

    Raises:
        RuntimeError: If the environment variable is unset or empty.
    """
    key = os.environ.get(AMAP_KEY_ENV_VAR)
    if not key:
        raise RuntimeError(
            f"Missing Amap API key: set the {AMAP_KEY_ENV_VAR} environment variable "
            f"to a valid Amap web-service key before calling the route tools."
        )
    return key


def geocode(address: str) -> str:
    """Resolve a place name to coordinates.

    Args:
        address: Place name to look up.

    Returns:
        The location as a ``"lng,lat"`` string.

    Raises:
        ValueError: If Amap returns no match for ``address``.
        RuntimeError: If the Amap API key is not configured.
    """
    url = f"{AMAP_BASE_URL}/geocode/geo?address={address}&key={_amap_key()}"
    response = requests.get(url).json()
    if response["status"] == AMAP_STATUS_OK and response["geocodes"]:
        return response["geocodes"][0]["location"]  # "lng,lat"
    else:
        raise ValueError(f"无法找到地址: {address}")


def get_walking(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    """Return the walking route between two coordinate pairs.

    Args:
        origin: Start coordinates as ``"lng,lat"``.
        destination: End coordinates as ``"lng,lat"``.

    Returns:
        Route summary, or ``None`` when Amap reports no walking route.

    Raises:
        RuntimeError: If the Amap API key is not configured.
    """
    url = f"{AMAP_BASE_URL}/direction/walking?origin={origin}&destination={destination}&key={_amap_key()}"
    response = requests.get(url).json()
    if response["status"] == AMAP_STATUS_OK and response["route"]["paths"]:
        route_path = response["route"]["paths"][0]
        return {
            "总时间(分钟)": int(route_path["duration"]) // SECONDS_PER_MINUTE,
            "总距离(米)": int(route_path["distance"]),
        }
    return None


def get_transit(origin: str, destination: str, city_code: str) -> Optional[Dict[str, Any]]:
    """Return the public-transit route between two coordinate pairs.

    Args:
        origin: Start coordinates as ``"lng,lat"``.
        destination: End coordinates as ``"lng,lat"``.
        city_code: Amap city code the trip takes place in.

    Returns:
        Route summary, or ``None`` when Amap reports no transit route.

    Raises:
        RuntimeError: If the Amap API key is not configured.
    """
    url = f"{AMAP_BASE_URL}/direction/transit/integrated?origin={origin}&destination={destination}&city={city_code}&key={_amap_key()}"
    response = requests.get(url).json()
    if response["status"] == AMAP_STATUS_OK and response["route"]["transits"]:
        transit = response["route"]["transits"][0]
        segments = []
        for segment in transit["segments"]:
            # Bus/metro leg.
            if "bus" in segment and segment["bus"].get("buslines"):
                buslines = segment["bus"]["buslines"]
                if buslines:  # Guard against an empty list.
                    busline = buslines[0]
                    segments.append(
                        f"乘坐 {busline['name']} "
                        f"({busline['departure_stop']['name']} 上车 → {busline['arrival_stop']['name']} 下车)"
                    )
            # Walking leg (tolerate either a dict or a list payload).
            if "walking" in segment and isinstance(segment["walking"], dict):
                if "distance" in segment["walking"]:
                    segments.append(f"步行 {segment['walking']['distance']} 米")

        return {
            "总时间(分钟)": int(transit["duration"]) // SECONDS_PER_MINUTE,
            "票价(元)": transit.get("cost", "未知"),
            "详细路线": segments
        }
    return None


def get_driving(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    """Return the driving route between two coordinate pairs.

    Args:
        origin: Start coordinates as ``"lng,lat"``.
        destination: End coordinates as ``"lng,lat"``.

    Returns:
        Route summary, or ``None`` when Amap reports no driving route.

    Raises:
        RuntimeError: If the Amap API key is not configured.
    """
    url = f"{AMAP_BASE_URL}/direction/driving?origin={origin}&destination={destination}&extensions=all&key={_amap_key()}"
    response = requests.get(url).json()
    if response["status"] == AMAP_STATUS_OK and response["route"]["paths"]:
        route_path = response["route"]["paths"][0]
        return {
            "总时间(分钟)": int(route_path["duration"]) // SECONDS_PER_MINUTE,
            "总距离(米)": int(route_path["distance"]),
            "过路费(元)": route_path.get("tolls", "0"),
        }
    return None


def query_routes(start_coords: str, end_place: str, city_code: str = DEFAULT_CITY_CODE) -> Dict[str, Optional[Dict[str, Any]]]:
    """Return walking, transit and driving routes to a destination.

    Args:
        start_coords: Start coordinates as ``"lng,lat"``.
        end_place: Destination place name, geocoded before routing.
        city_code: Amap city code used for the transit query.

    Returns:
        Mapping of travel mode to route summary (``None`` per mode with no route).

    Raises:
        ValueError: If ``end_place`` cannot be geocoded.
        RuntimeError: If the Amap API key is not configured.
    """
    destination = geocode(end_place)

    routes = {
        "步行": get_walking(start_coords, destination),
        "公交": get_transit(start_coords, destination, city_code),
        "驾车/打车": get_driving(start_coords, destination)
    }
    return routes


# Example usage.
if __name__ == "__main__":
    start_coords = "116.481028,39.989643"   # Start coordinates (Wangjing SOHO).
    end_place = "天坛"
    routes = query_routes(start_coords, end_place, city_code=DEFAULT_CITY_CODE)

    for mode, route_summary in routes.items():
        print(f"\n【{mode}】")
        if route_summary:
            for key, value in route_summary.items():
                print(f"{key}: {value}")
        else:
            print("未查询到结果")
