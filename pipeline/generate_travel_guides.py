#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate Chinese city travel guides in parallel.

For every city in the city-code mapping, ask deepseek-v3 for a detailed 1-5 day
travel guide and write it to ``data/raw/travel_guides`` as a plain-text file.
Work is spread over a thread pool and progress is journalled to
``data/seed/generation_progress.json``, so an interrupted run can be resumed:
cities already recorded as completed, or whose guide file already exists, are
skipped.
"""

from __future__ import annotations

import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from llm_client.client import call_llm
from paths import (
    CITY_CODE_MAPPING_FILE,
    GENERATION_PROGRESS_FILE,
    TRAVEL_GUIDES_DIR,
    ensure_dir,
)

# Global locks guarding thread-safe file writes and console output.
file_lock = threading.Lock()
console_lock = threading.Lock()

MODEL_NAME = "deepseek-v3"
DEFAULT_MAX_WORKERS = 5
# Worker count used by the interactive entry point; raise it to go faster.
INTERACTIVE_MAX_WORKERS = 3

# A guide shorter than this is treated as a failed generation and retried later.
MIN_GUIDE_LENGTH = 500
# Itinerary length asked of the model, in days.
MIN_GUIDE_DAYS = 1
MAX_GUIDE_DAYS = 5
# Random pause after each successful call, to avoid hammering the API.
MIN_REQUEST_DELAY_SECONDS = 1
MAX_REQUEST_DELAY_SECONDS = 3

# Keys inside city_code_mapping.json and generation_progress.json. These are
# on-disk data contracts; do not rename them.
CITY_MAPPING_KEY = "城市编码映射"
COMPLETED_KEY = "completed"
FAILED_KEY = "failed"
LAST_UPDATE_KEY = "last_update"

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
JSON_INDENT = 2

# Guide filenames look like ``110000_北京_travel_guide.txt``; the trailing
# administrative suffix is stripped from the city name. 361 existing files
# follow this pattern, so it must not change.
CITY_NAME_SUFFIX = "市"
GUIDE_FILENAME_SUFFIX = "_travel_guide.txt"
GUIDE_HEADER_RULE = "=" * 50
SEPARATOR_WIDTH = 50


class TravelGuideGenerator:
    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        """Set up the generator and load city data plus previous progress.

        Args:
            max_workers: Maximum number of guides to generate concurrently.
        """
        self.max_workers = max_workers
        self.city_mapping_file = CITY_CODE_MAPPING_FILE
        self.guides_dir = TRAVEL_GUIDES_DIR
        self.progress_file = GENERATION_PROGRESS_FILE

        ensure_dir(self.guides_dir)
        ensure_dir(self.progress_file.parent)

        self.cities = self._load_cities()
        self.progress = self._load_progress()

        with console_lock:
            print(f"✓ 初始化完成，共{len(self.cities)}个城市")
            print(f"✓ 已完成{len(self.progress.get(COMPLETED_KEY, []))}个城市的攻略生成")

    def _load_cities(self) -> dict[str, dict[str, Any]]:
        """Read the city-code mapping.

        Returns:
            Mapping of city code to city metadata, or an empty mapping if the
            file is missing or malformed.
        """
        try:
            with self.city_mapping_file.open(encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            with console_lock:
                print(f"❌ 加载城市数据失败: {exc}")
            return {}
        return data.get(CITY_MAPPING_KEY, {})

    def _load_progress(self) -> dict[str, Any]:
        """Read the generation journal.

        Returns:
            The journal read from disk, or a fresh empty journal if it does not
            exist yet or cannot be parsed.
        """
        if self.progress_file.exists():
            try:
                with self.progress_file.open(encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError) as exc:
                with console_lock:
                    print(f"⚠️ 加载进度文件失败: {exc}")
        return {COMPLETED_KEY: [], FAILED_KEY: [], LAST_UPDATE_KEY: ""}

    def _write_progress(self) -> None:
        """Stamp and flush the progress journal. Caller must hold ``file_lock``."""
        self.progress[LAST_UPDATE_KEY] = time.strftime(TIMESTAMP_FORMAT)
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(self.progress, f, ensure_ascii=False, indent=JSON_INDENT)

    def flush_progress(self) -> None:
        """Persist the journal without recording an outcome for any city.

        Used after editing the lists directly, where ``_save_progress`` would
        have to be passed a fake city code.
        """
        with file_lock:
            try:
                self._write_progress()
            except (OSError, TypeError) as exc:
                logger.warning("Failed to write progress journal: %s", exc)

    def _save_progress(self, city_code: str, city_name: str, success: bool) -> None:
        """Record the outcome for one city and flush the journal to disk.

        A successful city is added to the completed list and removed from the
        failed list; a failed one is added to the failed list. The whole update
        happens under ``file_lock`` so concurrent workers cannot interleave.

        Args:
            city_code: Administrative code of the city.
            city_name: Display name of the city, used only in error messages.
            success: Whether the guide was generated successfully.
        """
        with file_lock:
            try:
                if success:
                    if city_code not in self.progress.get(COMPLETED_KEY, []):
                        self.progress.setdefault(COMPLETED_KEY, []).append(city_code)
                    # Drop it from the failed list if a previous run recorded it.
                    if city_code in self.progress.get(FAILED_KEY, []):
                        self.progress[FAILED_KEY].remove(city_code)
                else:
                    if city_code not in self.progress.get(FAILED_KEY, []):
                        self.progress.setdefault(FAILED_KEY, []).append(city_code)

                self._write_progress()
            except (OSError, TypeError) as exc:
                with console_lock:
                    print(f"❌ 保存进度失败 ({city_name}): {exc}")

    def _generate_prompt(self, city_name: str, city_info: dict[str, Any]) -> tuple[str, str]:
        """Build the prompts for one city.

        The itinerary length is picked at random between ``MIN_GUIDE_DAYS`` and
        ``MAX_GUIDE_DAYS``.

        Args:
            city_name: Display name of the city.
            city_info: City metadata (``name``, ``level``, ``province``).

        Returns:
            The system prompt and the user prompt, in that order.
        """
        days = random.randint(MIN_GUIDE_DAYS, MAX_GUIDE_DAYS)

        system_prompt = """你是一位专业的旅游攻略规划师，擅长为游客制定详细实用的旅行计划。
请根据用户提供的城市信息，生成一份专业、详细、实用的旅游攻略。

攻略要求：
1. 内容要准确真实，基于该城市的实际情况
2. 包含景点推荐、美食推荐、住宿建议、交通指南等
3. 按天数合理安排行程，避免过于紧凑
4. 提供实用的旅行小贴士
5. 语言生动有趣，富有吸引力"""

        user_prompt = f"""请为{city_info.get('province', '')}的{city_name}制定一份{days}天的详细旅游攻略。

城市信息：
- 城市名称：{city_name}
- 行政级别：{city_info.get('level', '未知')}
- 所属省份：{city_info.get('province', '未知')}
- 攻略天数：{days}天

请按以下格式输出攻略：

# {city_name}{days}日游攻略

## 🏙️ 城市简介
[城市的基本介绍、历史文化背景、最佳旅行时间等]

## 🚗 交通指南
[如何到达该城市，以及市内交通方式]

## 🏨 住宿推荐
[推荐3-5个不同价位的住宿区域或酒店]

## 📅 详细行程安排

### 第1天：[主题]
- **上午**：[具体安排]
- **下午**：[具体安排] 
- **晚上**：[具体安排]
- **推荐美食**：[当地特色菜品]

[如果是多天行程，继续第2天、第3天等...]

## 🍽️ 必吃美食
[详细介绍当地特色美食，包括推荐餐厅]

## 🎁 购物推荐
[特产、纪念品购买建议]

## 💡 实用小贴士
[气候、着装、注意事项等实用信息]

## 💰 预算参考
[大致的花费预算，包含交通、住宿、餐饮、门票等]

请确保内容丰富详实，具有很强的实用性。"""

        return system_prompt, user_prompt

    def _generate_single_guide(self, city_code: str, city_info: dict[str, Any]) -> bool:
        """Generate and store the guide for one city.

        The city is skipped (and reported as a success) when it is already in the
        completed list or its guide file is already on disk, which is what makes
        the run resumable.

        Args:
            city_code: Administrative code of the city.
            city_info: City metadata (``name``, ``level``, ``province``).

        Returns:
            True if the guide exists after this call, False if generation failed.
        """
        city_name = city_info.get("name", "未知城市")

        if city_code in self.progress.get(COMPLETED_KEY, []):
            with console_lock:
                print(f"⏭️ {city_name} 攻略已存在，跳过")
            return True

        # The city code prefixes the filename so that duplicate city names
        # cannot collide.
        filename = f"{city_code}_{city_name.replace(CITY_NAME_SUFFIX, '')}{GUIDE_FILENAME_SUFFIX}"
        filepath = self.guides_dir / filename

        if filepath.exists():
            with console_lock:
                print(f"⏭️ {city_name} 攻略文件已存在，跳过")
            self._save_progress(city_code, city_name, True)
            return True

        try:
            with console_lock:
                print(f"🚀 开始生成 {city_name} 的旅游攻略...")

            system_prompt, user_prompt = self._generate_prompt(city_name, city_info)

            start_time = time.time()
            guide_content = call_llm(
                model_name=MODEL_NAME,
                prompt=user_prompt,
                system_prompt=system_prompt,
                stream=False,
                show_thinking=False,
            )
            end_time = time.time()

            if not guide_content or len(guide_content.strip()) < MIN_GUIDE_LENGTH:
                raise ValueError("生成的攻略内容过短或为空")

            with file_lock, filepath.open("w", encoding="utf-8") as f:
                f.write(f"# {city_name}旅游攻略\n")
                f.write(f"城市编码: {city_code}\n")
                f.write(f"生成时间: {time.strftime(TIMESTAMP_FORMAT)}\n")
                f.write(f"生成用时: {end_time - start_time:.2f}秒\n")
                f.write(f"生成模型: {MODEL_NAME}\n")
                f.write(GUIDE_HEADER_RULE + "\n\n")
                f.write(guide_content)

            self._save_progress(city_code, city_name, True)

            with console_lock:
                print(f"✅ {city_name} 攻略生成完成 ({end_time - start_time:.2f}秒)")

            time.sleep(random.uniform(MIN_REQUEST_DELAY_SECONDS, MAX_REQUEST_DELAY_SECONDS))
            return True

        # Broad on purpose: any LLM, network or filesystem failure must be
        # journalled as a failed city instead of killing the worker pool.
        except Exception as exc:
            with console_lock:
                print(f"❌ {city_name} 攻略生成失败: {exc}")
            self._save_progress(city_code, city_name, False)
            return False

    def generate_all_guides(self) -> None:
        """Generate guides for every city that is not already completed."""
        if not self.cities:
            print("❌ 没有找到城市数据")
            return

        pending_cities = {
            city_code: city_info
            for city_code, city_info in self.cities.items()
            if city_code not in self.progress.get(COMPLETED_KEY, [])
        }

        if not pending_cities:
            print("🎉 所有城市的攻略都已生成完成！")
            return

        print(f"📋 准备为{len(pending_cities)}个城市生成旅游攻略...")
        print(f"🔧 使用{self.max_workers}个并行线程")
        print(f"🤖 使用模型: {MODEL_NAME}")
        print("-" * SEPARATOR_WIDTH)

        success_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_city = {
                executor.submit(self._generate_single_guide, city_code, city_info): (
                    city_code,
                    city_info,
                )
                for city_code, city_info in pending_cities.items()
            }

            for future in as_completed(future_to_city):
                _city_code, city_info = future_to_city[future]
                city_name = city_info.get("name", "未知城市")

                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed_count += 1
                # The worker already handles its own errors; this only catches a
                # failure raised while retrieving the result itself.
                except Exception as exc:
                    with console_lock:
                        print(f"❌ {city_name} 处理异常: {exc}")
                    failed_count += 1

                total_processed = success_count + failed_count
                total_cities = len(pending_cities)
                progress_percent = (total_processed / total_cities) * 100

                with console_lock:
                    print(
                        f"📊 总进度: {total_processed}/{total_cities} ({progress_percent:.1f}%) | "
                        f"成功: {success_count} | 失败: {failed_count}"
                    )

        print("\n" + "=" * SEPARATOR_WIDTH)
        print("🎯 攻略生成完成!")
        print(f"✅ 成功生成: {success_count}个城市")
        print(f"❌ 生成失败: {failed_count}个城市")
        print(f"📁 攻略文件保存在: {self.guides_dir}")

        if failed_count > 0:
            print("\n⚠️ 失败的城市可以重新运行脚本进行重试")

    def retry_failed_cities(self) -> None:
        """Clear the failed list and generate those cities again, one at a time."""
        failed_city_codes = self.progress.get(FAILED_KEY, [])
        if not failed_city_codes:
            print("😊 没有需要重试的城市")
            return

        print(f"🔄 开始重试{len(failed_city_codes)}个失败的城市...")

        # Reset the failed list before retrying so this run repopulates it.
        self.progress[FAILED_KEY] = []
        self.flush_progress()

        for city_code in failed_city_codes:
            if city_code in self.cities:
                self._generate_single_guide(city_code, self.cities[city_code])


def main() -> None:
    """Run the interactive menu for generating, retrying and inspecting guides."""
    print("🎯 中国城市旅游攻略生成器")
    print("=" * SEPARATOR_WIDTH)

    generator = TravelGuideGenerator(max_workers=INTERACTIVE_MAX_WORKERS)

    while True:
        print("\n请选择操作:")
        print("1. 生成所有城市攻略")
        print("2. 重试失败的城市")
        print("3. 查看生成进度")
        print("4. 退出")

        choice = input("\n请输入选择 (1-4): ").strip()

        if choice == "1":
            generator.generate_all_guides()
        elif choice == "2":
            generator.retry_failed_cities()
        elif choice == "3":
            completed_count = len(generator.progress.get(COMPLETED_KEY, []))
            failed_count = len(generator.progress.get(FAILED_KEY, []))
            total_cities = len(generator.cities)
            print("\n📊 生成进度统计:")
            # total_cities is 0 when the city mapping failed to load.
            completed_percent = completed_count / total_cities * 100 if total_cities else 0.0
            print(
                f"✅ 已完成: {completed_count}/{total_cities} "
                f"({completed_percent:.1f}%)"
            )
            print(f"❌ 失败: {failed_count}")
            print(f"📅 最后更新: {generator.progress.get(LAST_UPDATE_KEY, '未知')}")
        elif choice == "4":
            print("👋 退出程序")
            break
        else:
            print("❌ 无效选择，请重新输入")


if __name__ == "__main__":
    main()
