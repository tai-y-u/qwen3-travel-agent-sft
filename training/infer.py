#!/usr/bin/env python3
"""Interactive inference harness for the merged Qwen travel-assistant model.

Loads a merged checkpoint (see :mod:`training.merge_lora`), renders the chat
messages defined at the bottom of this file through the model's chat template,
generates a reply, and prints it. ``--show_template`` stops after printing the
templated prompt and the head of its token ids, which is the quickest way to
check that tool schemas and the system prompt land where they should.

Despite the original filename this is a manual harness, not an automated test.

``MERGED_MODEL_DIR`` comes from :mod:`paths`, which owns it so that this module
reads exactly where ``training.merge_lora`` writes. Checkpoints live on the
remote GPU box rather than in this repository, so override with
``TRAVEL_AGENT_MERGED_MODEL`` or pass ``--model_path``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paths import MERGED_MODEL_DIR

logger = logging.getLogger(__name__)


# Qwen chat-template markers used to carve the assistant reply out of the
# decoded sequence.
ASSISTANT_START_TAG = "<|im_start|>assistant"
IM_END_TAG = "<|im_end|>"

# How many leading token ids --show_template prints.
TOKEN_ID_PREVIEW_COUNT = 128

# Separator width for the console sections.
RULE_WIDTH = 100


def parse_args() -> argparse.Namespace:
    """Parse inference arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Test inference for merged Qwen model (chat)")
    parser.add_argument("--model_path", type=str, default=MERGED_MODEL_DIR, help="Path to merged model dir")
    # parser.add_argument("--message", type=str, required=True, help="User message to send")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=6)
    parser.add_argument("--bad_words", type=str, default="", help="Optional pipe-separated phrases to ban, e.g. 'phrase A|phrase B'")
    parser.add_argument("--show_tokens", action="store_true", help="Print generated tokens (not IDs) before decode")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device_map_auto", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    # Tools injection for inference prompt
    parser.add_argument("--tools_file", type=str, default="", help="Path to JSON file with a list of tools (OpenAI/Qwen schema)")
    parser.add_argument("--tools_json", type=str, default="", help="Inline JSON string (list) of tools to pass into chat template")
    parser.add_argument("--show_template", action="store_true", help="Only print apply_chat_template text and token ids, then exit")
    return parser.parse_args()


def extract_assistant_reply(decoded: str) -> str:
    """Pull the final assistant turn out of a decoded generation.

    Args:
        decoded: Full decoded sequence, special tokens included.

    Returns:
        The text of the last assistant turn, or the whole stripped input when no
        assistant marker is present.
    """
    # Try to extract between assistant markers if present
    if ASSISTANT_START_TAG in decoded:
        last_start = decoded.rfind(ASSISTANT_START_TAG)
        content = decoded[last_start + len(ASSISTANT_START_TAG):]
        # Strip a leading newline if present
        if content.startswith("\n"):
            content = content[1:]
        # Cut off at end tag if present
        if IM_END_TAG in content:
            content = content.split(IM_END_TAG, 1)[0]
        return content.strip()
    # Fallback: return full decoded
    return decoded.strip()


def _supports_tools_kw(tokenizer: Any) -> bool:
    """Report whether ``tokenizer.apply_chat_template`` accepts a ``tools`` kwarg.

    Args:
        tokenizer: Tokenizer whose chat template is inspected.

    Returns:
        ``True`` if tool schemas can be passed to the template, ``False``
        otherwise (including when the signature cannot be inspected).
    """
    try:
        sig = inspect.signature(tokenizer.apply_chat_template)
    except (TypeError, ValueError) as exc:
        logger.debug("Cannot inspect apply_chat_template signature: %s", exc)
        return False
    return any(p.name == "tools" for p in sig.parameters.values())


def load_tools(args: argparse.Namespace) -> list[dict[str, Any]] | None:
    """Load the tool schemas to render into the prompt.

    Args:
        args: Parsed arguments; ``--tools_file`` wins over ``--tools_json``.

    Returns:
        The tool list, or ``None`` when neither option was given.

    Raises:
        ValueError: If the loaded JSON is not a list.
    """
    if args.tools_file:
        with Path(args.tools_file).open("r", encoding="utf-8") as f:
            loaded_tools = json.load(f)
        if not isinstance(loaded_tools, list):
            raise ValueError("tools_file must contain a JSON array of tool objects")
        return loaded_tools
    if args.tools_json:
        loaded_tools = json.loads(args.tools_json)
        if not isinstance(loaded_tools, list):
            raise ValueError("tools_json must be a JSON array of tool objects")
        return loaded_tools
    return None


def main(args: argparse.Namespace, messages: list[dict[str, Any]], is_print: bool = True) -> str | None:
    """Generate one assistant reply for ``messages``.

    Args:
        args: Parsed arguments (see :func:`parse_args`).
        messages: Chat messages to condition on, in OpenAI/Qwen format.
        is_print: Print the reply to stdout instead of returning it.

    Returns:
        The reply text when ``is_print`` is false, otherwise ``None``. Also
        returns ``None`` in ``--show_template`` mode, which never generates.
    """
    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Optionally load tools
    tools = load_tools(args)
    pass_tools = tools is not None and _supports_tools_kw(tokenizer)

    tensor_kwargs: dict[str, Any] = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt")
    if pass_tools:
        tensor_kwargs["tools"] = tools

    if args.show_template:
        text_kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        if pass_tools:
            text_kwargs["tools"] = tools
        templated_text = tokenizer.apply_chat_template(
            messages,
            **text_kwargs,
        )
        if is_print:
            print("===== apply_chat_template (text) =====")
            print(templated_text)

        # Print the leading token ids and the total token length.
        ids_kwargs: dict[str, Any] = dict(
            tokenize=True, add_generation_prompt=True, return_tensors=None
        )
        if pass_tools:
            ids_kwargs["tools"] = tools
        token_ids = tokenizer.apply_chat_template(
            messages,
            **ids_kwargs,
        )
        try:
            if isinstance(token_ids, list):
                flat_token_ids = token_ids
            else:
                flat_token_ids = token_ids.squeeze().tolist()
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Failed to present token ids (type %s): %s", type(token_ids), exc
            )
        else:
            if is_print:
                print("===== token ids (head) =====")
                print(flat_token_ids[:TOKEN_ID_PREVIEW_COUNT])
                print(f"total token length: {len(flat_token_ids)}")
        return None

    # The model is only needed once we actually generate, so it is loaded here
    # rather than up front (--show_template returns above without touching it).
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=torch_dtype,
        device_map="auto" if args.device_map_auto else None,
    )

    # Ensure use_cache is enabled for inference
    if hasattr(model, "config"):
        model.config.use_cache = True

    # Build inputs using chat template
    input_ids = tokenizer.apply_chat_template(
        messages,
        **tensor_kwargs,
    )
    input_ids = input_ids.to(model.device)

    # Compose EOS ids to include <|im_end|> if distinct from eos_token
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    try:
        im_end_id = tokenizer.convert_tokens_to_ids(IM_END_TAG)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Could not resolve the id of %s: %s", IM_END_TAG, exc)
    else:
        if isinstance(im_end_id, int) and im_end_id != -1 and im_end_id != tokenizer.eos_token_id:
            eos_ids.append(im_end_id)

    # Optional bad words ids
    bad_words_ids = None
    if args.bad_words.strip():
        phrases = [phrase.strip() for phrase in args.bad_words.split("|") if phrase.strip()]
        if phrases:
            encoded_phrases = tokenizer(phrases, add_special_tokens=False).input_ids
            bad_words_ids = [ids for ids in encoded_phrases if len(ids) > 0]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": eos_ids if len(eos_ids) > 0 else tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }
    # Sampling parameters only mean something when sampling. Passing them under
    # greedy decoding makes transformers warn, so send them only in sampling
    # mode. The defaults (temperature 0.0, top_k 1) describe greedy decoding and
    # are not valid sampling settings, hence the check.
    if args.do_sample:
        if args.temperature <= 0:
            raise ValueError(
                f"--do_sample needs a positive --temperature, got {args.temperature}. "
                "The default of 0.0 means greedy decoding; drop --do_sample for that."
            )
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
        generation_kwargs["top_k"] = args.top_k
    if bad_words_ids:
        generation_kwargs["bad_words_ids"] = bad_words_ids

    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids, **generation_kwargs)

    # Slice out only the generated continuation (exclude the prompt length)
    generated_ids = output_ids[0, input_ids.shape[1]:]

    if args.show_tokens and is_print:
        print("-" * RULE_WIDTH)
        print("看解码前token信息")
        # Convert token IDs to token strings without joining so you can see BPE units
        token_strings = tokenizer.convert_ids_to_tokens(generated_ids.tolist())
        print("TOKENS:")
        print(token_strings)

    # Decode the whole sequence, then keep only the last assistant turn.
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    reply = extract_assistant_reply(decoded)

    if is_print:
        print("-" * RULE_WIDTH)
        print("正式回复：")
        print(reply)
        return None
    return reply


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    args = parse_args()

    # Fixed conversation used for manual probing. The system prompt and the user
    # turn are product content and are kept verbatim; the commented-out turns
    # below are a reference example of a full tool-calling round trip.
    messages = [
      {
        "role": "system",
        "content": "## 用户信息\n- 用户名: 许飞\n- 当前城市ID: 101220601\n- 出发日期: 2025-09-16\n- 起点坐标: 116.794664,33.971707\n\n请在处理用户请求时考虑这些信息，比如：\n- 问路时如果没有明确起点，使用起点坐标116.794664,33.971707\n- 旅行规划时根据用户的出发日期范围提供建议\n- 天气查询时根据旅行攻略中的天数来确定查询天数，使用时间段查询\n- 路线查询时起点优先使用起点坐标\n\n你是一个专业的旅行助手，严格按照以下工作流程处理用户请求：\n\n## 工作流1: 旅行规划\n**触发条件**: 用户想制定旅行计划、询问某地旅游攻略、景点推荐等\n**处理流程**:\n1. 首先检查是否有目的地信息\n   - 如果没有目的地或目的地不明确，必须反问：\"请告诉我您想去哪个城市旅行？\"\n   - 如果用户问\"附近有什么好玩的\"，可以基于用户当前城市提供建议\n   - 不要进行任何工具调用，直接反问\n2. 如果有明确的目的地，检查是否有出行日期\n   - 如果没有明确日期，可以基于当前日期推荐合适的出行时间\n   - 如果完全没有时间信息，反问：\"请问您计划什么时候出行？\"\n3. 信息完整后，必须**按照顺序**调用以下两个工具：\n   - `search_travel_guide`: 搜索目的地旅行攻略\n   - `get_weather_info`: 查询出行日期的天气信息\n4. 综合天气和攻略信息，制定详细的旅行计划，规划里面说明每一天的天气信息。不需要详细，只需要说明每天的温度，晴雨天等信息，无需根据天气改变景点顺序，如果search_travel_guide这个工具返回的结果为空，则不再调用天气工具，直接返回无相应旅行路线\n\n## 工作流2: 问路/地图导航\n**触发条件**: 用户询问路线、问路、导航、\"怎么走\"、\"如何到达\"等\n**处理流程**:\n1. 检查起点和终点信息，起点信息在起点坐标获得，无需询问，也无需询问城市，城市就是当前城市ID\n   - 如果用户说\"从X到Y\"、\"X到Y怎么走\"，则X是起点，Y是终点，直接调用工具，只要说了大致的地点，比如说了火车站、医院、学校等信息，则无需追问，直接调用工具查询即可，无需追问具体在哪里\n   - 如果用户说\"怎么回家\"，可以提醒用户提供具体地址\n   - 如果完全没有重点的地点信息，才反问：\"请问您要去哪里？\"能不追问尽量不追问，只要有信息则直接调用工具开始查询\n2. 信息完整后，调用`query_route`工具获取路线\n3. 为用户提供步行、公交、驾车等多种路线选择，如果query_route这个工具返回的结果为空，则直接返回查询不到对应路线\n\n## 工作流3: 酒店查询\n**触发条件**: 用户询问酒店推荐、酒店预订、住宿等\n**处理流程**:\n### 3A: 酒店推荐\n1. 检查必要信息：目的地，只有在缺少城市的时候追问，其余时候都直接调用工具\n   - 缺少目的地：反问\"请问您要在哪个城市找酒店？\"\n   - 如果用户说\"本地酒店\"或\"附近酒店\"，使用当前城市\n\n2. 信息收集完毕后，**必须按顺序执行**：\n   - 第一步：调用`recommend_hotels`工具获取酒店推荐\n   - 第二步：**立即**为推荐的酒店调用`get_hotel_reviews`工具获取评价\n   - **重要**：不要在第一步后就返回结果，必须完成所有工具调用，如果第一步没有推荐结果，则直接返回\"暂时没有找到符合您需求的酒店推荐\"，无需进行第二步工具调用\n3. 整合酒店信息和评价，生成完整的推荐结果（包含酒店详情+用户评价）\n\n### 3B: 酒店评价查询\n**触发条件**: 询问某酒店\"怎么样\"、\"评价\"、\"好不好\"等\n1. 提取酒店名称\n   - 如果没有明确酒店名称，反问：\"请问您想了解哪家酒店的评价？\"，\n2. 调用`get_hotel_reviews`获取评价信息\n\n## 工作流4: 闲聊\n**触发条件**: 旅行相关的一般性问题、打招呼等\n**处理流程**:\n1. 判断是否与旅行相关\n   - 如果是旅行相关话题，直接回答，不调用工具\n   - 如果完全无关（如数学、编程等），礼貌拒绝：\"抱歉，我是专门的旅行助手，只能回答旅行相关的问题。\"\n\n## 重要原则:\n1. **信息不足时必须反问，不要猜测**\n2. **一次只问一个关键信息，避免一次问太多**\n3. **工具调用顺序很重要，旅行规划时必须先调用攻略，再根据返回情况决定是否调用天气**\n4. **酒店推荐工作流特别重要**：\n   - 先调用`recommend_hotels`获取酒店列表\n   - 然后为推荐酒店调用`get_hotel_reviews`\n   - 最后整合所有信息统一回复\n   - 绝不在获得酒店推荐后立即回复，必须获取评价后才能给最终答案\n5. **非旅行相关问题要礼貌拒绝**\n\n## 反问示例:\n- \"请告诉我您想去哪个城市旅行？\"\n- \"请问您计划什么时候出行？（请提供具体日期）\"\n- \"请问您要去哪里？\"\n- \"请问您要在哪个城市找酒店？\"\n- \"请问您的预算范围是多少？\"\n\n严格按照以上流程处理用户请求，确保信息完整后再调用相应工具。"
      },
      {
        "role": "user",
        "content": "想去嘉兴玩"
      },
    #   {
    #     "role": "assistant",
    #     "content": "",
    #     "tool_calls": [
    #       {
    #         "id": "call_710eefea",
    #         "type": "function",
    #         "function": {
    #           "name": "search_travel_guide",
    #           "arguments": "{\"location\": \"嘉兴\", \"search_mode\": \"hybrid\"}"
    #         }
    #       },
    #       {
    #         "id": "call_28a4d191",
    #         "type": "function",
    #         "function": {
    #           "name": "get_weather_info",
    #           "arguments": "{\"location\": \"嘉兴\", \"num_days\": 5, \"start_date\": \"2025-09-16\"}"
    #         }
    #       }
    #     ]
    #   },
    #   {
    #     "role": "tool",
    #     "content": "# 嘉兴市旅游攻略\n城市编码: 330400\n生成时间: 2025-09-12 10:11:18\n生成用时: 55.51秒\n生成模型: deepseek-v3\n==================================================\n\n# 嘉兴市5日游攻略\n\n## 🏙️ 城市简介\n嘉兴市位于浙江省东北部，地处长江三角洲中心地带，是一座有着2500多年历史的江南水乡名城。作为中国革命红船的起航地，嘉兴素有\"鱼米之乡、丝绸之府\"的美誉。这里水网密布，古桥众多，典型的江南水乡风貌令人流连忘返。最佳旅行时间为春季（3-5月）和秋季（9-11月），此时气候宜人，景色最美。\n\n## 🚗 交通指南\n**到达方式：**\n- 高铁：嘉兴南站连接上海、杭州、南京等主要城市，上海虹桥至嘉兴南仅需27分钟\n- 自驾：沪昆高速、常台高速等多条高速公路贯穿全境\n- 长途汽车：嘉兴汽车北站有发往周边城市的长途班车\n\n**市内交通：**\n- 公交车：覆盖主要景点，票价1-2元\n- 出租车：起步价10元\n- 公共自行车：市区设有多个租赁点\n- 水上巴士：体验水乡特色的交通方式\n\n## 🏨 住宿推荐\n💰 经济型（150-300元/晚）：\n- 嘉兴月河客栈：位于月河历史街区，具有江南特色\n- 如家快捷酒店（中山路店）：交通便利，性价比高\n\n💰 舒适型（300-600元/晚）：\n- 嘉兴富悦大酒店：四星级，设施完善\n- 嘉兴沙龙国际宾馆：位于市中心，购物方便\n\n💰 豪华型（600元以上/晚）：\n- 嘉兴隆聚汇金大酒店：五星级，服务一流\n- 乌镇枕水度假酒店：体验水乡奢华住宿\n\n## 📅 详细行程安排\n\n### 第1天：南湖红船与月河古街\n- **上午**：参观南湖革命纪念馆，瞻仰红船，了解中共一大历史\n- **下午**：游览南湖景区，乘船欣赏湖光山色，参观烟雨楼\n- **晚上**：漫步月河历史街区，感受古镇夜景\n- **推荐美食**：南湖菱角、月河臭豆腐\n\n### 第2天：乌镇西栅深度游\n- **上午**：前往乌镇西栅景区，游览昭明书院、乌镇大剧院\n- **下午**：体验摇橹船，参观草木本色染坊，欣赏水乡建筑\n- **晚上**：欣赏西栅夜景，灯光下的古镇别有一番韵味\n- **推荐美食**：乌镇白水鱼、定胜糕\n\n### 第3天：西塘古镇体验\n- **上午**：前往西塘古镇，游览烟雨长廊、送子来凤桥\n- **下午**：参观西园、醉园等古典园林，体验古镇慢生活\n- **晚上**：夜游西塘，欣赏河灯美景\n- **推荐美食**：芡实糕、西塘黄酒\n\n### 第4天：盐官观潮与海宁皮革\n- **上午**：前往盐官观潮胜地公园，观看钱塘江大潮（需查询潮汐时间）\n- **下午**：参观海宁中国皮革城，选购皮具制品\n- **晚上**：返回嘉兴市区，体验当地夜市\n- **推荐美食**：海宁缸肉、钱江鲈鱼\n\n### 第5天：梅花洲与返程\n- **上午**：游览梅花洲景区，参观石佛古寺，体验禅意文化\n- **下午**：购买特产纪念品，整理行装准备返程\n- **晚上**：根据返程时间安排晚餐，结束愉快旅程\n- **推荐美食**：嘉兴粽子、文虎酱鸭\n\n## 🍽️ 必吃美食\n1. **嘉兴粽子**：五芳斋粽子最为著名，鲜肉粽、豆沙粽都值得尝试\n   - 推荐：五芳斋总店（南湖大道）\n   \n2. **南湖菱角**：当地特产，可生吃或煮熟，清甜爽口\n   - 推荐：月河历史街区小吃摊\n\n3. **文虎酱鸭**：嘉兴传统名菜，酱香浓郁\n   - 推荐：文虎酱鸭总店（中山路）\n\n4. **乌镇白水鱼**：清蒸做法，肉质鲜嫩\n   - 推荐：乌镇景区内老字号餐馆\n\n5. **海宁缸肉**：用特制陶缸焖制，肥而不腻\n   - 推荐：盐官古镇农家乐\n\n## 🎁 购物推荐\n1. **嘉兴粽子**：五芳斋真空包装粽子，便于携带\n2. **杭白菊**：桐乡特产，清热去火\n3. **皮革制品**：海宁皮革城的皮衣、皮包\n4. **蓝印花布**：乌镇传统手工艺品\n5. **西塘黄酒**：古镇特产，醇香可口\n\n## 💡 实用小贴士\n1. 春秋季节温差较大，建议带件外套\n2. 雨季（6-7月）记得带雨具\n3. 观潮务必提前查询潮汐时间，注意安全\n4. 古镇石板路不平，建议穿舒适的运动鞋\n5. 提前预订酒店，特别是在节假日期间\n6. 保存好景区门票，部分景点可二次入园\n\n## 💰 预算参考（每人）\n**交通：**\n- 大交通：根据出发地不同，往返高铁票约200-500元\n- 市内交通：200元（含景区间交通）\n\n**住宿：**\n- 经济型：750元（150元/晚×5晚）\n- 舒适型：1500元（300元/晚×5晚）\n\n**餐饮：**\n- 每日150元，5天共计750元\n\n**门票：**\n- 南湖景区：60元\n- 乌镇西栅：150元\n- 西塘古镇：95元\n- 盐官观潮公园：30元\n- 梅花洲：60元\n- 总计：约400元\n\n**总计：**\n- 经济型：约2100-2400元\n- 舒适型：约2850-3150元\n\n（以上为大致预算，实际花费因个人消费习惯而异）",
    #     "tool_call_id": "call_710eefea"
    #   },
    #   {
    #     "role": "tool",
    #     "content": "天气信息(嘉兴 - 2025-09-16起5天):\n\n第1天 (2025-09-16):\n  白天: 晴，夜间: 晴\n  温度: 26℃ ~ 35℃\n\n第2天 (2025-09-17):\n  白天: 小雨，夜间: 小雨\n  温度: 23℃ ~ 29℃\n\n第3天 (2025-09-18):\n  白天: 阴，夜间: 多云\n  温度: 20℃ ~ 24℃\n\n第4天 (2025-09-19):\n  白天: 多云，夜间: 阴\n  温度: 22℃ ~ 31℃\n\n第5天 (2025-09-20):\n  白天: 多云，夜间: 多云\n  温度: 23℃ ~ 31℃\n",
    #     "tool_call_id": "call_28a4d191"
    #   },
    #   {
    #     "role": "assistant",
    #     "content": "# 嘉兴5日度假旅行计划\n\n## 📅 出行日期\n2025年9月16日 - 2025年9月20日\n\n## ☀️ 天气概况\n- 第1天：晴天，26℃~35℃，天气炎热\n- 第2天：小雨，23℃~29℃，较凉爽，注意防雨\n- 第3天：阴/多云，20℃~24℃，天气舒适\n- 第4天：多云/阴，22℃~31℃，天气较热\n- 第5天：多云，23℃~31℃，天气较热\n\n## 🏞️ 行程安排建议\n\n### 第1天（晴天，炎热）\n- 推荐室内景点：南湖革命纪念馆、月河历史街区\n- 建议避开午间高温时段外出\n- 注意防晒和补水\n\n### 第2天（小雨）\n- 推荐游览乌镇西栅：雨中的水乡别有一番韵味\n- 体验摇橹船和草木本色染坊\n- 建议携带雨具\n\n### 第3天（阴/多云，舒适）\n- 推荐前往西塘古镇：适合户外游览\n- 体验烟雨长廊和古镇慢生活\n- 气温适宜，可安排较多户外活动\n\n### 第4天（多云/阴）\n- 可安排盐官观潮：注意查看具体潮汐时间\n- 参观海宁中国皮革城\n- 建议准备防晒用品\n\n### 第5天（多云）\n- 推荐游览梅花洲景区：感受禅意文化\n- 购买特产纪念品\n- 根据返程时间安排最后的购物\n\n## 🍽️ 美食推荐\n必尝美食：嘉兴粽子、南湖菱角、文虎酱鸭、乌镇白水鱼、海宁缸肉\n\n## 💡 实用建议\n1. 由于第2天有雨，建议携带雨具或准备一次性雨衣\n2. 第1天和最后两天天气较热，注意做好防晒措施\n3. 古镇游览建议穿舒适的平底鞋，因为需要较多步行\n4. 提前预订酒店和景点门票，特别是在旅游旺季\n5. 带一件薄外套，应对第2天和第3天的凉爽天气\n\n祝您在嘉兴的5日度假愉快！这座江南水乡一定会给您留下难忘的回忆。"
    #   }
    ]

    main(args, messages)


# Example invocations (run from the repository root; --model_path points at a
# checkpoint on the GPU box):
# python -m training.infer \
#   --model_path /path/to/qwen3-0_6b_lora_v5_merged --local_files_only --show_template
#
# python -u -m training.infer \
#   --model_path /path/to/qwen3-0_6b_lora_v2_last_assistant \
#   --bf16 --device_map_auto --local_files_only