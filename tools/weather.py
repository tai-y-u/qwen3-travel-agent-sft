"""Daily weather lookups backed by the QWeather API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

API_HOST = "https://nk4d92xupx.re.qweatherapi.com"
API_KEY = ""

DATE_FORMAT = "%Y-%m-%d"
DEFAULT_QUERY_RANGE = "30d"


def get_weather_by_date_range(location: str, start_date: str, num_days: int, query_range: str = DEFAULT_QUERY_RANGE) -> List[Dict[str, Any]]:
    """Look up the weather for a place over a range of days.

    Args:
        location: Place identifier (city ID such as ``101010100``, or ``'116.41,39.92'``).
        start_date: First day of the range, formatted ``'YYYY-MM-DD'``.
        num_days: Number of days to report, counting from ``start_date``.
        query_range: API forecast window; one of ``3d`` / ``7d`` / ``10d`` / ``15d`` / ``30d``.

    Returns:
        One entry per matching day, or an empty list when nothing matches.
    """
    url = f"{API_HOST}/v7/weather/{query_range}"
    params = {"location": location, "key": API_KEY}
    response = requests.get(url, params=params)
    data = response.json()

    if "daily" not in data:
        print("❌ 未获取到天气数据:", data)
        return []

    # Build the list of target dates.
    start_datetime = datetime.strptime(start_date, DATE_FORMAT)
    target_dates = []
    for day_offset in range(num_days):
        target_date = start_datetime + timedelta(days=day_offset)
        target_dates.append(target_date.strftime(DATE_FORMAT))

    # Collect the forecast entries that fall inside the target range.
    weather_list = []
    for daily_forecast in data["daily"]:
        if daily_forecast["fxDate"] in target_dates:
            weather_info = {
                "日期": daily_forecast["fxDate"],
                "日出": daily_forecast["sunrise"],
                "日落": daily_forecast["sunset"],
                "白天天气": daily_forecast["textDay"],
                "夜间天气": daily_forecast["textNight"],
                "最高温": daily_forecast["tempMax"] + "℃",
                "最低温": daily_forecast["tempMin"] + "℃",
            }
            weather_list.append(weather_info)
    
    # Sort by date.
    weather_list.sort(key=lambda entry: entry["日期"])
    return weather_list


def get_weather_by_date(location: str, date: str, query_range: str = DEFAULT_QUERY_RANGE) -> Optional[Dict[str, Any]]:
    """Look up the weather for a place on a single day (legacy interface).

    Args:
        location: Place identifier (city ID such as ``101010100``, or ``'116.41,39.92'``).
        date: Day to report, formatted ``'YYYY-MM-DD'``.
        query_range: API forecast window; one of ``3d`` / ``7d`` / ``10d`` / ``15d`` / ``30d``.

    Returns:
        That day's weather, or ``None`` when the day is not in the forecast.
    """
    weather_list = get_weather_by_date_range(location, date, 1, query_range)
    return weather_list[0] if weather_list else None


# Example usage.
if __name__ == "__main__":
    weather_info = get_weather_by_date("101010100", "2025-09-19")
    if weather_info:
        for field, value in weather_info.items():
            print(f"{field}: {value}")
    else:
        print("未找到对应日期的天气数据")


