# -*- coding: utf-8 -*-
"""LLM-generated hotel recommendations and reviews."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from llm_client.client import call_llm

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "qwen-plus"


def _parse_json_array(response: str) -> List[Dict[str, Any]]:
    """Parse the JSON array a model returned, ignoring any surrounding text.

    Slices between the first ``[`` and the last ``]`` so markdown fences and
    commentary around the payload are tolerated; falls back to parsing the whole
    stripped response when no brackets are present.

    Args:
        response: Raw model output.

    Returns:
        The decoded JSON payload.

    Raises:
        json.JSONDecodeError: If the extracted text is not valid JSON.
    """
    # Extract the JSON section (dropping any markdown formatting).
    json_start = response.find('[')
    json_end = response.rfind(']') + 1
    if json_start != -1 and json_end > json_start:
        return json.loads(response[json_start:json_end])
    # No JSON array found, so try parsing the whole response.
    return json.loads(response.strip())


def get_hotel_recommendations(user_requirements: str, model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, Any]]:
    """Get a list of recommended hotels.

    Args:
        user_requirements: The user's requirements, for example
            "a business hotel in Shanghai under 1000 CNY, near a metro station".
        model_name: Name of the LLM model to use.

    Returns:
        Recommended hotels; each entry carries the name, location and similar
        details. Falls back to canned data if the model output cannot be parsed.
    """


    system_prompt = """你是一个专业的酒店推荐专家。根据用户需求，推荐合适的酒店。
请严格按照以下JSON格式返回酒店推荐列表：

[
  {
    "hotel_name": "酒店名称",
    "location": "具体位置/地址"
  }
]

请返回1个推荐酒店，确保数据真实合理。只返回JSON数组格式，不要其他文字。请用中文填写内容。也可以返回为空，表示没有找到合适的酒店。但是整体JSON格式要正确，只是key对应的value为空"""

    user_prompt = f"""用户需求：{user_requirements}

请根据以上需求推荐合适的酒店，返回JSON格式的酒店列表。"""

    try:
        response = call_llm(model_name, user_prompt, system_prompt, stream=False, show_thinking=False)

        # Parse the JSON response.
        try:
            return _parse_json_array(response)
        except json.JSONDecodeError as exc:
            # Unparseable model output, so fall back to canned data.
            logger.warning("Could not parse hotel recommendations as JSON, using fallback data: %s", exc)
            return _generate_fallback_hotels(user_requirements)

    # Broad on purpose: any provider/transport failure must still yield a result.
    except Exception as e:
        logger.exception("Hotel recommendation call failed, using fallback data")
        print(f"调用LLM时出错: {e}")
        return _generate_fallback_hotels(user_requirements)


def get_hotel_reviews(hotel_name: str, model_name: str = DEFAULT_MODEL_NAME) -> List[Dict[str, Any]]:
    """Get reviews for a hotel.

    Args:
        hotel_name: Name of the hotel.
        model_name: Name of the LLM model to use.

    Returns:
        User reviews, each with a rating, review text and reviewer details.
        Falls back to canned data if the model output cannot be parsed.
    """


    system_prompt = """你是一个酒店评论数据专家。根据酒店名称，生成真实合理，包含了正面负面的用户评论。
请严格按照以下JSON格式返回评论列表：

[
  {
    "reviewer_name": "评论者昵称",
    "rating": 4,
    "review_date": "2024-12-15",
    "review_content": "详细的评论内容，要真实自然",
  }
]

请返回1条评论，确保评论内容真实自然，直接输出评价的文字即可。请用中文填写内容。3-5句话，要真实，可以正面也可以负面"""

    user_prompt = f"""用户问题：{hotel_name}怎么样

请回答这个问题，不要输出其他文字"""

    try:
        response = call_llm(model_name, user_prompt, system_prompt, stream=False, show_thinking=False)

        # Parse the JSON response.
        try:
            return _parse_json_array(response)
        except json.JSONDecodeError as exc:
            # Unparseable model output, so fall back to canned data.
            logger.warning("Could not parse hotel reviews as JSON, using fallback data: %s", exc)
            return _generate_fallback_reviews(hotel_name)

    # Broad on purpose: any provider/transport failure must still yield a result.
    except Exception as e:
        logger.exception("Hotel review call failed, using fallback data")
        print(f"调用LLM时出错: {e}")
        return _generate_fallback_reviews(hotel_name)


def _generate_fallback_hotels(user_requirements: str) -> List[Dict[str, Any]]:
    """Return canned hotel recommendations used when the model call fails."""
    fallback_hotels = [
        {
            "hotel_name": "假日酒店",
            "location": "市中心商业区",
            "price_range": "400-600元/晚",
            "room_types": ["标准间", "豪华间", "套房"],
            "rating": 4.2,
            "amenities": ["免费WiFi", "健身房", "早餐", "停车场"],
            "distance_to_transport": "地铁站500米"
        },
        {
            "hotel_name": "商务精选酒店",
            "location": "金融中心",
            "price_range": "600-900元/晚",
            "room_types": ["商务间", "行政套房"],
            "rating": 4.5,
            "amenities": ["商务中心", "会议室", "免费WiFi", "接送服务"],
            "distance_to_transport": "地铁站300米"
        },
        {
            "hotel_name": "经济型连锁酒店",
            "location": "交通枢纽附近",
            "price_range": "200-350元/晚",
            "room_types": ["标准间", "大床房"],
            "rating": 3.8,
            "amenities": ["免费WiFi", "24小时前台"],
            "distance_to_transport": "火车站200米"
        }
    ]
    return fallback_hotels


def _generate_fallback_reviews(hotel_name: str) -> List[Dict[str, Any]]:
    """Return canned hotel reviews used when the model call fails."""
    fallback_reviews = [
        {
            "reviewer_name": "旅行达人小王",
            "rating": 4,
            "review_date": "2024-12-15",
            "review_content": "酒店位置很好，房间干净整洁，服务人员态度友好。早餐种类丰富，性价比不错。",
        },
        {
            "reviewer_name": "商务出差人",
            "rating": 5,
            "review_date": "2024-12-10",
            "review_content": "商务设施齐全，会议室很专业，网络稳定。房间安静，适合工作和休息。",
        },
    ]
    return fallback_reviews


DEMO_RULE_WIDTH = 50
MISSING_FIELD_PLACEHOLDER = "N/A"

# Example usage.
if __name__ == "__main__":
    # Demo the hotel recommendation helper.
    # The prompt only asks for hotel_name and location, so every other field is
    # optional and read with .get(); only the fallback data carries them.
    print("=== 酒店推荐测试 ===")
    user_requirements = "北京商务酒店，预算800元以内，近地铁站，需要会议室"
    hotels = get_hotel_recommendations(user_requirements)
    for i, hotel in enumerate(hotels, 1):
        print(f"\n酒店 {i}:")
        print(f"名称: {hotel.get('hotel_name', MISSING_FIELD_PLACEHOLDER)}")
        print(f"位置: {hotel.get('location', MISSING_FIELD_PLACEHOLDER)}")
        print(f"价格: {hotel.get('price_range', MISSING_FIELD_PLACEHOLDER)}")
        print(f"评分: {hotel.get('rating', MISSING_FIELD_PLACEHOLDER)}")

    print("\n" + "=" * DEMO_RULE_WIDTH)

    # Demo the hotel review helper.
    print("=== 酒店评论测试 ===")
    hotel_name = "北京假日酒店"
    reviews = get_hotel_reviews(hotel_name)
    for i, review in enumerate(reviews, 1):
        print(f"\n评论 {i}:")
        print(f"评论者: {review.get('reviewer_name', MISSING_FIELD_PLACEHOLDER)}")
        print(f"评分: {review.get('rating', MISSING_FIELD_PLACEHOLDER)}/5")
        print(f"评论: {review.get('review_content', MISSING_FIELD_PLACEHOLDER)}")
        print(f"入住目的: {review.get('stay_purpose', MISSING_FIELD_PLACEHOLDER)}")