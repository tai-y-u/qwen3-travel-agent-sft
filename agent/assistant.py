#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Function-calling travel assistant (with the duplicate tool-call bug fixed).

Lets the qwen-plus model decide on its own which tools to call, then runs each
requested tool exactly once per tool call.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List

import requests
from openai import OpenAI

from tools.route import query_routes, geocode
from tools.weather import get_weather_by_date_range
from tools.hotel import get_hotel_recommendations, get_hotel_reviews

logger = logging.getLogger(__name__)

# OpenAI-compatible endpoint of Alibaba Cloud DashScope.
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# Model used when the caller does not request a specific one.
DEFAULT_MODEL_NAME = "qwen-plus"
# Local travel-guide RAG service.
RAG_API_URL = "http://127.0.0.1:8010"
RAG_SEARCH_LIMIT = 3
RAG_REQUEST_TIMEOUT_SECONDS = 10

# Default user profile applied when the caller passes nothing.
DEFAULT_USER_CITY_ID = "101010100"
DEFAULT_TRAVEL_DATE_RANGE = "2025-09-15~2025-10-05"
DEFAULT_TRAVEL_START_DATE = "2025-09-15"
DEFAULT_TRAVEL_END_DATE = "2025-10-05"
DEFAULT_START_COORDINATES = "116.481028,39.989643"
DEFAULT_CITY_CODE = "110000"

# Number of chat messages kept in the rolling conversation history.
MAX_HISTORY_MESSAGES = 20
# Truncation limits applied when rendering tool results for the model.
MAX_HOTELS_SHOWN = 3
MAX_REVIEWS_SHOWN = 2
# Characters of a tool result written to the debug log.
TOOL_RESULT_PREVIEW_CHARS = 200
# Width of the "-" rule printed between interactive CLI turns.
CLI_SEPARATOR_WIDTH = 50


class TravelAssistantFuncCall:
    """Travel assistant driven by OpenAI-style function calling."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, user_name: str = "用户",
                 user_city_id: str = DEFAULT_USER_CITY_ID,
                 travel_date_range: str = DEFAULT_TRAVEL_DATE_RANGE,
                 start_coordinates: str = DEFAULT_START_COORDINATES) -> None:
        """Initialise the assistant and its tool schemas.

        Args:
            model_name: DashScope model to call.
            user_name: Display name of the user.
            user_city_id: Weather-service city ID of the user's city, e.g.
                ``"101010100"`` for Beijing.
            travel_date_range: Departure window as ``"YYYY-MM-DD~YYYY-MM-DD"``.
            start_coordinates: Default route origin as ``"lng,lat"``.
        """
        self.model_name = model_name
        self.conversation_history: List[Dict[str, str]] = []
        self.rag_api_url = RAG_API_URL

        # User profile passed to the model on every turn.
        self.user_info = {
            "name": user_name,
            "city_id": user_city_id,  # City ID, e.g. 101010100 for Beijing
            "travel_date_range": travel_date_range,  # e.g. 2025-09-15~2025-10-05
            "start_coordinates": start_coordinates,  # e.g. 116.481028,39.989643
            "current_date": self._get_current_date()
        }

        # Split the departure window into start and end dates.
        self._parse_travel_dates()

        self.client = OpenAI(
            api_key="",
            base_url=DASHSCOPE_BASE_URL
        )

        # Tool schemas advertised to the model.
        self.tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "search_travel_guide",
                    "description": "搜索旅行攻略信息，获取目的地的详细旅游信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "目的地城市名称，如'北京'、'上海'"
                            },
                            "search_mode": {
                                "type": "string",
                                "description": "搜索模式：vector(向量搜索)、keyword(关键词搜索)、hybrid(混合搜索)",
                                "enum": ["vector", "keyword", "hybrid"],
                                "default": "hybrid"
                            }
                        },
                        "required": ["location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather_info",
                    "description": "查询指定地点和时间段的天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市名称，如'北京'、'上海'，或城市ID如'101010100'"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "开始日期，格式YYYY-MM-DD，如'2025-09-15'"
                            },
                            "num_days": {
                                "type": "integer",
                                "description": "查询天数，默认1天。根据旅行攻略中的行程天数确定"
                            }
                        },
                        "required": ["location", "start_date"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_route",
                    "description": "查询两地之间的路线，包括步行、公交、驾车路线",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_location": {
                                "type": "string",
                                "description": "起点坐标，如'116.481028,39.989643'"
                            },
                            "end_location": {
                                "type": "string",
                                "description": "终点地址，如'颐和园'"
                            },
                            "city_code": {
                                "type": "string",
                                "description": "城市代码，默认'110000'(北京)",
                                "default": "110000"
                            }
                        },
                        "required": ["start_location", "end_location"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "recommend_hotels",
                    "description": "根据用户需求推荐酒店",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requirements": {
                                "type": "string",
                                "description": "用户对酒店的需求描述，包括地点、预算、设施等"
                            }
                        },
                        "required": ["requirements"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_hotel_reviews",
                    "description": "获取指定酒店的用户评价信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hotel_name": {
                                "type": "string",
                                "description": "酒店名称"
                            }
                        },
                        "required": ["hotel_name"]
                    }
                }
            }
        ]

    def _get_current_date(self) -> str:
        """Return today's date as ``YYYY-MM-DD``."""
        return datetime.now().strftime("%Y-%m-%d")

    def _parse_travel_dates(self) -> None:
        """Split the departure window into start and end dates.

        Stores ``travel_start_date`` and ``travel_end_date`` in
        :attr:`user_info`. A value without ``~`` is treated as a single day; an
        unparsable value falls back to the module defaults.
        """
        try:
            if "~" in self.user_info["travel_date_range"]:
                start_date, end_date = self.user_info["travel_date_range"].split("~")
                self.user_info["travel_start_date"] = start_date.strip()
                self.user_info["travel_end_date"] = end_date.strip()
            else:
                # No range given, so treat the value as a single date.
                self.user_info["travel_start_date"] = self.user_info["travel_date_range"]
                self.user_info["travel_end_date"] = self.user_info["travel_date_range"]
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.warning(
                "Could not parse travel date range %r, falling back to defaults: %s",
                self.user_info.get("travel_date_range"), e
            )
            print(f"解析出发日期出错: {e}")
            self.user_info["travel_start_date"] = DEFAULT_TRAVEL_START_DATE
            self.user_info["travel_end_date"] = DEFAULT_TRAVEL_END_DATE

    def add_to_history(self, role: str, content: str) -> None:
        """Append a message to the conversation history.

        Args:
            role: Chat role, ``"user"`` or ``"assistant"``.
            content: Message text.
        """
        self.conversation_history.append({"role": role, "content": content})
        # Keep the history at a reasonable length.
        if len(self.conversation_history) > MAX_HISTORY_MESSAGES:
            self.conversation_history = self.conversation_history[-MAX_HISTORY_MESSAGES:]

    def search_travel_guide(self, location: str, search_mode: str = "hybrid") -> str:
        """Search the travel-guide RAG service.

        Args:
            location: Destination city to search for.
            search_mode: One of ``"vector"``, ``"keyword"`` or ``"hybrid"``.

        Returns:
            The matching guide passages joined by blank lines, or a message
            explaining why nothing was found.
        """
        try:
            url = f"{self.rag_api_url}/search"
            request_payload = {
                "query": location,
                "search_type": search_mode,
                "limit": RAG_SEARCH_LIMIT
            }
            response = requests.post(url, json=request_payload, timeout=RAG_REQUEST_TIMEOUT_SECONDS)

            if response.status_code == 200:
                response_body = response.json()
                if "results" in response_body and response_body["results"]:
                    guides = []
                    for guide_match in response_body["results"]:
                        if "content" in guide_match:
                            guides.append(guide_match["content"])
                    return "\n\n".join(guides)
                else:
                    return "未找到相关旅行攻略信息"
            else:
                return "旅行攻略查询服务暂时不可用"
        except (requests.RequestException, ValueError) as e:
            logger.exception("Travel guide search failed for location %r", location)
            return f"旅行攻略查询出错: {e}"

    def get_weather_info(self, location: str, start_date: str, num_days: int = 1) -> str:
        """Look up the weather forecast for a location.

        Args:
            location: City name, or a numeric weather-service city ID.
            start_date: First day of the forecast as ``YYYY-MM-DD``.
            num_days: Number of days to fetch, starting at ``start_date``.

        Returns:
            A human-readable forecast summary, or a message explaining why no
            forecast is available.
        """
        try:
            # City name to city ID mapping.
            city_id_map = {
                "北京": "101010100",
                "上海": "101020100",
                "广州": "101280101",
                "深圳": "101280601",
                "杭州": "101210101",
                "南京": "101190101"
            }

            # Resolve the city ID.
            if location in city_id_map:
                city_id = city_id_map[location]
            elif location.isdigit():
                city_id = location
            else:
                # Fall back to the user's own city ID.
                city_id = self.user_info["city_id"]

            weather_list = get_weather_by_date_range(city_id, start_date, num_days)

            if weather_list:
                weather_text = f"天气信息({location} - {start_date}起{num_days}天):\n"
                for i, weather in enumerate(weather_list, 1):
                    weather_text += f"\n第{i}天 ({weather['日期']}):\n"
                    weather_text += f"  白天: {weather['白天天气']}，夜间: {weather['夜间天气']}\n"
                    weather_text += f"  温度: {weather['最低温']} ~ {weather['最高温']}\n"
                return weather_text
            else:
                return f"无法获取{location}在{start_date}起{num_days}天的天气信息"
        except (requests.RequestException, IndexError, KeyError, RuntimeError, TypeError, ValueError) as e:
            logger.exception(
                "Weather lookup failed for location %r starting %s", location, start_date
            )
            return f"天气查询出错: {e}"

    def query_route(self, start_location: str, end_location: str,
                    city_code: str = DEFAULT_CITY_CODE) -> str:
        """Look up walking, transit and driving routes between two places.

        Args:
            start_location: Origin as ``"lng,lat"`` coordinates, a place name to
                geocode, or the literal "current location" phrase understood by
                the model; unresolvable origins fall back to the user's default
                coordinates.
            end_location: Destination address or place name.
            city_code: Amap city code used for the transit lookup.

        Returns:
            A human-readable summary of every available travel mode, or a
            message explaining why no route could be found.
        """
        try:
            # Resolve the origin coordinates.
            if "," in start_location and len(start_location.split(",")) == 2:
                start_coordinates = start_location
            elif start_location == "当前位置":
                start_coordinates = self.user_info["start_coordinates"]
            else:
                try:
                    start_coordinates = geocode(start_location)
                except (requests.RequestException, IndexError, KeyError, RuntimeError, ValueError) as exc:
                    # Geocoding failed, so use the user's default coordinates.
                    logger.warning(
                        "Geocoding %r failed, using the user's default coordinates: %s",
                        start_location, exc
                    )
                    start_coordinates = self.user_info["start_coordinates"]

            try:
                routes = query_routes(start_coordinates, end_location, city_code)
            except ValueError as e:
                return f"路线查询出错: {e}"

            route_text = f"从 {start_location} 到 {end_location} 的路线:\n\n"

            for mode, route_info in routes.items():
                route_text += f"【{mode}】\n"
                if route_info:
                    if mode == "步行":
                        route_text += f"总时间: {route_info['总时间(分钟)']}分钟\n"
                        route_text += f"总距离: {route_info['总距离(米)']}米\n"
                    elif mode == "公交":
                        route_text += f"总时间: {route_info['总时间(分钟)']}分钟\n"
                        if route_info['票价(元)'] != "未知":
                            route_text += f"票价: {route_info['票价(元)']}元\n"
                        if '详细路线' in route_info and route_info['详细路线']:
                            route_text += "路线: " + " → ".join(route_info['详细路线'][:3]) + "\n"
                    elif mode == "驾车/打车":
                        route_text += f"总时间: {route_info['总时间(分钟)']}分钟\n"
                        route_text += f"总距离: {route_info['总距离(米)']}米\n"
                        if route_info['过路费(元)'] != "0":
                            route_text += f"过路费: {route_info['过路费(元)']}元\n"
                else:
                    route_text += "暂无路线信息\n"
                route_text += "\n"

            return route_text
        except (requests.RequestException, AttributeError, IndexError, KeyError, RuntimeError,
                TypeError) as e:
            logger.exception(
                "Route lookup failed from %r to %r", start_location, end_location
            )
            return f"路线查询出错: {e}"

    def recommend_hotels(self, requirements: str) -> str:
        """Recommend hotels that match a free-text requirement description.

        Args:
            requirements: The user's hotel requirements, e.g. city, budget and
                desired amenities.

        Returns:
            A human-readable list of up to three hotels, or a message saying no
            match was found.
        """
        try:
            hotels = get_hotel_recommendations(requirements)
            if hotels:
                hotel_text = "根据您的需求，推荐以下酒店:\n\n"
                for i, hotel in enumerate(hotels[:MAX_HOTELS_SHOWN], 1):
                    hotel_text += f"酒店{i}: {hotel['hotel_name']}\n"
                    hotel_text += f"位置: {hotel['location']}\n"
                    # The new JSON format may only carry hotel_name and
                    # location, so read the remaining fields defensively.
                    if hotel.get('price_range'):
                        hotel_text += f"价格: {hotel['price_range']}\n"
                    if hotel.get('rating'):
                        hotel_text += f"评分: {hotel['rating']}/5\n"
                    if hotel.get('amenities'):
                        hotel_text += f"设施: {', '.join(hotel['amenities'][:4])}\n"
                    if hotel.get('distance_to_transport'):
                        hotel_text += f"交通: {hotel['distance_to_transport']}\n"
                    hotel_text += "\n"
                return hotel_text
            else:
                return "暂时没有找到符合您需求的酒店推荐"
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.exception("Hotel recommendation failed for requirements %r", requirements)
            return f"酒店推荐出错: {e}"

    def get_hotel_reviews_func(self, hotel_name: str) -> str:
        """Fetch user reviews for a hotel.

        Args:
            hotel_name: Name of the hotel to look up.

        Returns:
            A human-readable summary of up to two reviews, or a message saying
            no review was found.
        """
        try:
            reviews = get_hotel_reviews(hotel_name)
            if reviews:
                review_text = f"{hotel_name}的用户评价:\n\n"
                for i, review in enumerate(reviews[:MAX_REVIEWS_SHOWN], 1):
                    review_text += f"评价{i}:\n"
                    review_text += f"评分: {review.get('rating', '未知')}/5\n"
                    review_text += f"评论: {review.get('review_content', '')}\n\n"
                return review_text
            else:
                return f"暂时没有找到{hotel_name}的用户评价信息"
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            logger.exception("Hotel review lookup failed for hotel %r", hotel_name)
            return f"酒店评价查询出错: {e}"

    def _should_continue_tool_chain(self, function_name: str, result: str) -> bool:
        """Decide whether the tool result should trigger another tool round.

        Args:
            function_name: Name of the tool that just ran.
            result: Text the tool returned.

        Returns:
            ``True`` when a follow-up tool call is expected: hotel
            recommendations should be followed by a review lookup, and a
            travel-guide search should be followed by a weather lookup.
        """
        if function_name == "recommend_hotels":
            # Valid hotel recommendations must be followed by the review tool.
            try:
                # Check whether the result carries usable hotel information.
                if "酒店" in result and ("位置" in result or "价格" in result):
                    return True
                # Check whether the result is empty.
                if "没有找到" in result or "暂时没有" in result or len(result.strip()) < 20:
                    return False
            except (AttributeError, TypeError) as exc:
                logger.debug("Unexpected hotel result type %s: %s", type(result), exc)
            return False
        elif function_name == "search_travel_guide":
            # A valid travel guide must be followed by the weather tool.
            try:
                if "未找到" in result or "不可用" in result or len(result.strip()) < 50:
                    return False
                return True
            except (AttributeError, TypeError) as exc:
                logger.debug("Unexpected guide result type %s: %s", type(result), exc)
            return False
        else:
            return False

    def call_function(self, function_name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a model tool call to the matching Python method.

        Args:
            function_name: Tool name the model requested.
            arguments: Decoded JSON arguments for the tool.

        Returns:
            The tool's rendered result, or a message naming the unknown tool.
        """
        if function_name == "search_travel_guide":
            return self.search_travel_guide(
                arguments.get("location"),
                arguments.get("search_mode", "hybrid")
            )
        elif function_name == "get_weather_info":
            return self.get_weather_info(
                arguments.get("location"),
                arguments.get("start_date"),
                arguments.get("num_days", 1)
            )
        elif function_name == "query_route":
            return self.query_route(
                arguments.get("start_location"),
                arguments.get("end_location"),
                arguments.get("city_code", DEFAULT_CITY_CODE)
            )
        elif function_name == "recommend_hotels":
            return self.recommend_hotels(arguments.get("requirements"))
        elif function_name == "get_hotel_reviews":
            return self.get_hotel_reviews_func(arguments.get("hotel_name"))
        else:
            return f"未知函数: {function_name}"

    def process_user_input(self, user_input: str) -> str:
        """Answer one user turn, running any tools the model asks for.

        Sends the system prompt plus the conversation history to the model, runs
        each requested tool exactly once, optionally lets the model request a
        second tool round, and then asks for the final answer. The reply is
        appended to the conversation history.

        Args:
            user_input: The user's message.

        Returns:
            The assistant's reply, or an error message if the call failed.
        """
        # Record the user turn in the conversation history.
        self.add_to_history("user", user_input)

        # Build the message list.
        user_info_text = f"""
## 用户信息
- 用户名: {self.user_info['name']}
- 当前城市ID: {self.user_info['city_id']}
- 出发日期: {self.user_info['travel_date_range']}
- 起点坐标: {self.user_info['start_coordinates']}

请在处理用户请求时考虑这些信息，比如：
- 问路时如果没有明确起点，使用起点坐标{self.user_info['start_coordinates']}
- 旅行规划时根据用户的出发日期范围提供建议
- 天气查询时根据旅行攻略中的天数来确定查询天数，使用时间段查询
- 路线查询时起点优先使用起点坐标
"""

        messages = [
            {
                "role": "system", 
                "content": user_info_text + """
你是一个专业的旅行助手，严格按照以下工作流程处理用户请求：

## 工作流1: 旅行规划
**触发条件**: 用户想制定旅行计划、询问某地旅游攻略、景点推荐等
**处理流程**:
1. 首先检查是否有目的地信息
   - 如果没有目的地或目的地不明确，必须反问："请告诉我您想去哪个城市旅行？"
   - 如果用户问"附近有什么好玩的"，可以基于用户当前城市提供建议
   - 不要进行任何工具调用，直接反问
2. 如果有明确的目的地，检查是否有出行日期
   - 如果没有明确日期，可以基于当前日期推荐合适的出行时间
   - 如果完全没有时间信息，反问："请问您计划什么时候出行？"
3. 信息完整后，必须**按照顺序**调用以下两个工具：
   - `search_travel_guide`: 搜索目的地旅行攻略
   - `get_weather_info`: 查询出行日期的天气信息
4. 综合天气和攻略信息，制定详细的旅行计划，规划里面说明每一天的天气信息。不需要详细，只需要说明每天的温度，晴雨天等信息，无需根据天气改变景点顺序，如果search_travel_guide这个工具返回的结果为空，则不再调用天气工具，直接返回无相应旅行路线

## 工作流2: 问路/地图导航
**触发条件**: 用户询问路线、问路、导航、"怎么走"、"如何到达"等
**处理流程**:
1. 检查起点和终点信息，起点信息在起点坐标获得，无需询问，也无需询问城市，城市就是当前城市ID
   - 如果用户说"从X到Y"、"X到Y怎么走"，则X是起点，Y是终点，直接调用工具，如果说了火车站、医院、学校等公共设施的信息则无需追问，直接调用工具查询即可，无需追问具体在哪里
   - 如果用户说"怎么回家"，可以提醒用户提供具体地址
   - 如果没有明确的终点，必须反问："请问您要去哪里？"
2. 信息完整后，调用`query_route`工具获取路线
3. 为用户提供步行、公交、驾车等多种路线选择，如果query_route这个工具返回的结果为空，则直接返回查询不到对应路线

## 工作流3: 酒店查询
**触发条件**: 用户询问酒店推荐、酒店预订、住宿等
**处理流程**:
### 3A: 酒店推荐
1. 检查必要信息：目的地，只有在缺少城市的时候追问，其余时候都直接调用工具
   - 缺少目的地：反问"请问您要在哪个城市找酒店？"
   - 如果用户说"本地酒店"或"附近酒店"，使用当前城市

2. 信息收集完毕后，**必须按顺序执行**：
   - 第一步：调用`recommend_hotels`工具获取酒店推荐
   - 第二步：**立即**为推荐的酒店调用`get_hotel_reviews`工具获取评价
   - **重要**：不要在第一步后就返回结果，必须完成所有工具调用，如果第一步没有推荐结果，则直接返回"暂时没有找到符合您需求的酒店推荐"，无需进行第二步工具调用
3. 整合酒店信息和评价，生成完整的推荐结果（包含酒店详情+用户评价）

### 3B: 酒店评价查询
**触发条件**: 询问某酒店"怎么样"、"评价"、"好不好"等
1. 提取酒店名称
   - 如果没有明确酒店名称，反问："请问您想了解哪家酒店的评价？"，
2. 调用`get_hotel_reviews`获取评价信息

## 工作流4: 闲聊
**触发条件**: 旅行相关的一般性问题、打招呼等
**处理流程**:
1. 判断是否与旅行相关
   - 如果是旅行相关话题，直接回答，不调用工具
   - 如果完全无关（如数学、编程等），礼貌拒绝："抱歉，我是专门的旅行助手，只能回答旅行相关的问题。"

## 重要原则:
1. **信息不足时必须反问，不要猜测**
2. **一次只问一个关键信息，避免一次问太多**
3. **工具调用顺序很重要，旅行规划时必须先调用攻略，再根据返回情况决定是否调用天气**
4. **酒店推荐工作流特别重要**：
   - 先调用`recommend_hotels`获取酒店列表
   - 然后为推荐酒店调用`get_hotel_reviews`
   - 最后整合所有信息统一回复
   - 绝不在获得酒店推荐后立即回复，必须获取评价后才能给最终答案
5. **非旅行相关问题要礼貌拒绝**

## 反问示例:
- "请告诉我您想去哪个城市旅行？"
- "请问您计划什么时候出行？（请提供具体日期）"
- "请问您要去哪里？"
- "请问您要在哪个城市找酒店？"
- "请问您的预算范围是多少？"

严格按照以上流程处理用户请求，确保信息完整后再调用相应工具。直接输出结果，不要思考/no_think"""
            }
        ]

        # Append the conversation history.
        messages.extend(self.conversation_history)

        try:
            # First round: find out whether any tool call is needed.
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            message = response.choices[0].message

            # Return straight away when no tool call is needed.
            if not message.tool_calls:
                content = message.content
                self.add_to_history("assistant", content)
                return content

            # Run the requested tools.
            first_round_results = []
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)

                logger.debug("Calling tool %s with arguments %s", function_name, arguments)
                print(f"🔧 调用工具: {function_name}")
                print(f"参数: {arguments}")

                # Run the tool once and keep its result.
                result = self.call_function(function_name, arguments)
                logger.debug("Tool %s result: %s", function_name,
                             result[:TOOL_RESULT_PREVIEW_CHARS])
                print(f"🔧 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")

                first_round_results.append({
                    "call": tool_call,
                    "result": result,
                    "should_continue": self._should_continue_tool_chain(function_name, result)
                })

            # Add the first-round tool calls to the message list.
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in message.tool_calls]
            })

            # Add the tool results.
            for first_round_result in first_round_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": first_round_result["call"].id,
                    "content": first_round_result["result"]
                })

            # Decide whether more tool calls are needed.
            needs_more_tools = any(
                first_round_result["should_continue"] for first_round_result in first_round_results
            )

            if needs_more_tools:
                # Second round: the model may request more tool calls.
                second_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto"
                )

                second_message = second_response.choices[0].message

                if second_message.tool_calls:
                    # Handle the second tool round; each call runs exactly once.
                    second_round_results = []
                    for tool_call in second_message.tool_calls:
                        function_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments)

                        logger.debug("Continuing with tool %s, arguments %s",
                                     function_name, arguments)
                        print(f"🔧 继续调用工具: {function_name}")
                        print(f"参数: {arguments}")

                        result = self.call_function(function_name, arguments)
                        logger.debug("Tool %s result: %s", function_name,
                                     result[:TOOL_RESULT_PREVIEW_CHARS])
                        print(f"🔧 工具结果: {result[:200]}{'...' if len(result) > 200 else ''}")

                        second_round_results.append({
                            "tool_call_id": tool_call.id,
                            "content": result
                        })

                    # Add the second-round tool calls to the message list.
                    messages.append({
                        "role": "assistant",
                        "content": second_message.content,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in second_message.tool_calls]
                    })

                    # Add the second-round tool results. The fix: reuse the
                    # results already collected instead of calling again.
                    for second_round_result in second_round_results:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": second_round_result["tool_call_id"],
                            "content": second_round_result["content"]
                        })

            # Final call: produce the answer for the user.
            final_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )

            final_content = final_response.choices[0].message.content
            self.add_to_history("assistant", final_content)

            return final_content

        except Exception as e:
            # Last-resort guard: any model, network or tool failure is reported
            # to the user instead of crashing the conversation loop.
            logger.exception("Failed to process user input")
            error_msg = f"处理请求时出错: {e}"
            self.add_to_history("assistant", error_msg)
            return error_msg


def main() -> None:
    """Run the interactive command-line chat loop."""
    print("🌍 欢迎使用智能旅行助手！")
    print("我可以帮您：")
    print("1. 制定旅行计划（会自动查询天气和攻略）")
    print("2. 查询路线（问路导航）")
    print("3. 推荐酒店和查看评价")
    print("4. 回答旅行相关问题")

    # Collect the user profile.
    user_name = input("\n请输入您的姓名（默认：旅行者）：").strip() or "旅行者"
    user_city_input = input("请输入您所在的城市（默认：北京）：").strip() or "北京"

    # City name to city ID mapping.
    city_id_map = {
        "北京": "101010100", "上海": "101020100", "广州": "101280101",
        "深圳": "101280601", "杭州": "101210101", "南京": "101190101"
    }
    user_city_id = city_id_map.get(user_city_input, DEFAULT_USER_CITY_ID)

    travel_range = input("请输入出发日期范围（默认：2025-09-15~2025-10-05）：").strip() or "2025-09-15~2025-10-05"
    start_coordinates = input("请输入起点坐标（默认：116.481028,39.989643）：").strip() or "116.481028,39.989643"

    print(f"✅ 用户信息设置完成:")
    print(f"   姓名: {user_name}")
    print(f"   城市: {user_city_input} (ID: {user_city_id})")
    print(f"   出发范围: {travel_range}")
    print(f"   起点坐标: {start_coordinates}")
    print("输入 'quit' 或 'exit' 退出\n")

    assistant = TravelAssistantFuncCall(
        user_name=user_name,
        user_city_id=user_city_id,
        travel_date_range=travel_range,
        start_coordinates=start_coordinates
    )

    while True:
        try:
            user_input = input("您: ").strip()

            if user_input.lower() in ['quit', 'exit', '退出', '再见']:
                print("🌍 感谢使用旅行助手，祝您旅途愉快！")
                break

            if not user_input:
                continue

            print("\n正在处理您的请求...")
            response = assistant.process_user_input(user_input)
            print(f"\n助手: {response}\n")
            print("-" * CLI_SEPARATOR_WIDTH)

        except KeyboardInterrupt:
            print("\n\n🌍 感谢使用旅行助手，祝您旅途愉快！")
            break
        except Exception as e:
            logger.exception("Interactive turn failed")
            print(f"\n❌ 出现错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    main()
