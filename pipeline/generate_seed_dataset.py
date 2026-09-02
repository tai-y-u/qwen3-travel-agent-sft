#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate the seed dataset for the travel assistant.

The seed dataset is a list of user requests, one per row, tagged with the
workflow that should handle it. Rows are emitted with the Chinese field names the
rest of the pipeline reads, and the result is written to ``SEED_DIR``.
"""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from paths import SEED_DIR, ensure_dir

# Departure dates are drawn uniformly from this closed window.
DEPARTURE_WINDOW_START = datetime(2025, 9, 15)
DEPARTURE_WINDOW_END = datetime(2025, 10, 5)
DATE_FORMAT = "%Y-%m-%d"
FILE_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Share of route questions that use a real landmark of the sampled city; the rest
# use generic landmarks to simulate places the geocoder cannot resolve.
REAL_LANDMARK_RATIO = 0.9
# Share of hotel questions that ask for a recommendation rather than a review.
HOTEL_RECOMMEND_RATIO = 0.6

# One worker per generator method.
MAX_WORKERS = 8

# Generator method name -> number of rows it should produce. The dataset total is
# derived from this table, so the two can never drift apart.
TASK_COUNTS: tuple[tuple[str, int], ...] = (
    ("generate_travel_planning_no_ask", 400),
    ("generate_travel_planning_ask", 50),
    ("generate_route_no_ask", 100),
    ("generate_route_ask", 20),
    ("generate_hotel_no_ask", 200),
    ("generate_hotel_ask", 40),
    ("generate_travel_chat", 100),
    ("generate_reject", 100),
)
TOTAL_SAMPLES = sum(count for _, count in TASK_COUNTS)

# Workflow id -> label used in the summary printed after generation.
WORKFLOW_NAMES: dict[int, str] = {
    1: "旅行规划", 2: "问路", 3: "查询酒店", 4: "旅行相关", 5: "拒答"
}

# City id -> display name and "longitude,latitude" origin coordinate.
CITIES: dict[str, dict[str, str]] = {
    "101010100": {"name": "北京", "coord": "116.481028,39.989643"},
    "101020100": {"name": "上海", "coord": "121.473701,31.230416"},
    "101280101": {"name": "广州", "coord": "113.280637,23.125178"},
    "101280601": {"name": "深圳", "coord": "114.085947,22.547"},
    "101210101": {"name": "杭州", "coord": "120.153576,30.287459"},
    "101190101": {"name": "南京", "coord": "118.767413,32.041544"},
    "101200101": {"name": "武汉", "coord": "114.298572,30.584355"},
    "101270101": {"name": "成都", "coord": "104.065735,30.659462"},
    "101120101": {"name": "天津", "coord": "117.190182,39.125596"},
    "101230101": {"name": "福州", "coord": "119.306239,26.075302"},
    "101240101": {"name": "南昌", "coord": "115.892151,28.676493"},
    "101250101": {"name": "长沙", "coord": "112.982279,28.19409"},
    "101260101": {"name": "贵阳", "coord": "106.713478,26.578343"},
    "101290101": {"name": "昆明", "coord": "102.712251,25.040609"},
    "101300101": {"name": "西安", "coord": "108.948024,34.263161"},
    "101110101": {"name": "石家庄", "coord": "114.502461,38.045474"},
    "101140101": {"name": "济南", "coord": "117.000923,36.675807"},
    "101180101": {"name": "郑州", "coord": "113.665412,34.757975"},
    "101160101": {"name": "太原", "coord": "112.549248,37.857014"},
    "101170101": {"name": "呼和浩特", "coord": "111.670801,40.818311"},
    "101040100": {"name": "重庆", "coord": "106.504962,29.533155"},
    "101050101": {"name": "哈尔滨", "coord": "126.642464,45.756967"},
    "101060101": {"name": "长春", "coord": "125.3245,43.886841"},
    "101070101": {"name": "沈阳", "coord": "123.429096,41.796767"},
    "101080101": {"name": "呼和浩特", "coord": "111.670801,40.818311"},
    "101090101": {"name": "银川", "coord": "106.278179,38.46637"},
    "101100101": {"name": "西宁", "coord": "101.778916,36.623178"},
    "101150101": {"name": "兰州", "coord": "103.73438,36.03122"},
    "101130101": {"name": "乌鲁木齐", "coord": "87.617733,43.792818"},
    "101310101": {"name": "拉萨", "coord": "91.132212,29.660361"},
    "101320101": {"name": "海口", "coord": "110.35,20.02"},
    "101320201": {"name": "三亚", "coord": "109.508268,18.247872"},
    "101220101": {"name": "合肥", "coord": "117.283042,31.86119"},
    "101220201": {"name": "芜湖", "coord": "118.376451,31.326319"},
    "101220301": {"name": "蚌埠", "coord": "117.363228,32.929499"},
    "101220401": {"name": "淮南", "coord": "117.018329,32.647574"},
    "101220501": {"name": "马鞍山", "coord": "118.507906,31.689362"},
    "101220601": {"name": "淮北", "coord": "116.794664,33.971707"},
    "101220701": {"name": "铜陵", "coord": "117.816576,30.929935"},
    "101220801": {"name": "安庆", "coord": "117.043551,30.50883"},
    "101220901": {"name": "黄山", "coord": "118.317325,29.709239"},
    "101221001": {"name": "滁州", "coord": "118.316264,32.317351"},
    "101221101": {"name": "阜阳", "coord": "115.819729,32.896969"},
    "101221201": {"name": "宿州", "coord": "116.984084,33.646307"},
    "101221301": {"name": "六安", "coord": "116.507676,31.752889"},
    "101221401": {"name": "亳州", "coord": "115.782939,33.844582"},
    "101221501": {"name": "池州", "coord": "117.489157,30.656037"},
    "101221601": {"name": "宣城", "coord": "118.757995,30.945667"},
    "101250201": {"name": "株洲", "coord": "113.151737,27.835806"},
    "101250301": {"name": "湘潭", "coord": "112.944052,27.82973"},
    "101250401": {"name": "衡阳", "coord": "112.607693,26.900358"},
    "101250501": {"name": "邵阳", "coord": "111.461525,27.237842"},
    "101250601": {"name": "岳阳", "coord": "113.132855,29.37029"},
    "101250701": {"name": "常德", "coord": "111.691347,29.040225"},
    "101250801": {"name": "张家界", "coord": "110.479921,29.127401"},
    "101250901": {"name": "益阳", "coord": "112.355042,28.570066"},
    "101251001": {"name": "郴州", "coord": "113.032067,25.770509"},
    "101251101": {"name": "永州", "coord": "111.608019,26.434516"},
    "101251201": {"name": "怀化", "coord": "109.97824,27.550082"},
    "101251301": {"name": "娄底", "coord": "112.008497,27.728136"},
    "101251401": {"name": "湘西", "coord": "109.739735,28.314296"},
    "101210201": {"name": "宁波", "coord": "121.549792,29.868388"},
    "101210301": {"name": "温州", "coord": "120.672111,28.000575"},
    "101210401": {"name": "嘉兴", "coord": "120.750865,30.762653"},
    "101210501": {"name": "湖州", "coord": "120.102398,30.867198"},
    "101210601": {"name": "绍兴", "coord": "120.582112,29.997117"},
    "101210701": {"name": "金华", "coord": "119.649506,29.089524"},
    "101210801": {"name": "衢州", "coord": "118.87263,28.941708"},
    "101210901": {"name": "舟山", "coord": "122.207216,29.985295"},
    "101211001": {"name": "台州", "coord": "121.428599,28.661378"},
    "101211101": {"name": "丽水", "coord": "119.921786,28.451993"},
    "101230201": {"name": "厦门", "coord": "118.11022,24.490474"},
    "101230301": {"name": "莆田", "coord": "119.007558,25.431011"},
    "101230401": {"name": "三明", "coord": "117.635001,26.265444"},
    "101230501": {"name": "泉州", "coord": "118.514861,24.901652"},
    "101230601": {"name": "漳州", "coord": "117.661801,24.510897"},
    "101230701": {"name": "南平", "coord": "118.178459,26.635627"},
    "101230801": {"name": "龙岩", "coord": "117.02978,25.091603"},
    "101230901": {"name": "宁德", "coord": "119.527082,26.65924"},
    "101190201": {"name": "无锡", "coord": "120.301663,31.574729"},
    "101190301": {"name": "徐州", "coord": "117.184811,34.261792"},
    "101190401": {"name": "常州", "coord": "119.946973,31.772752"},
    "101190501": {"name": "苏州", "coord": "120.619585,31.299379"},
    "101190601": {"name": "南通", "coord": "120.864608,32.016212"},
    "101190701": {"name": "连云港", "coord": "119.178821,34.600018"},
    "101190801": {"name": "淮安", "coord": "119.015285,33.597506"},
    "101190901": {"name": "盐城", "coord": "120.139998,33.377631"},
    "101191001": {"name": "扬州", "coord": "119.421003,32.393159"},
    "101191101": {"name": "镇江", "coord": "119.452753,32.204402"},
    "101191201": {"name": "泰州", "coord": "119.915176,32.484882"},
    "101191301": {"name": "宿迁", "coord": "118.275162,33.963008"},
    "101280201": {"name": "韶关", "coord": "113.591544,24.801322"},
    "101280301": {"name": "珠海", "coord": "113.553986,22.224979"},
    "101280401": {"name": "汕头", "coord": "116.708463,23.37102"},
    "101280501": {"name": "佛山", "coord": "113.122717,23.028762"},
    "101280701": {"name": "江门", "coord": "113.094942,22.590431"},
    "101280801": {"name": "湛江", "coord": "110.364977,21.274898"},
    "101280901": {"name": "茂名", "coord": "110.88018,21.659751"},
    "101281001": {"name": "肇庆", "coord": "112.472529,23.051546"},
    "101281101": {"name": "惠州", "coord": "114.412599,23.079404"},
    "101281201": {"name": "梅州", "coord": "116.117582,24.299112"},
    "101281301": {"name": "汕尾", "coord": "115.364238,22.774485"},
    "101281401": {"name": "河源", "coord": "114.697802,23.746266"},
    "101281501": {"name": "阳江", "coord": "111.975107,21.859222"},
    "101281601": {"name": "清远", "coord": "113.051227,23.685022"},
    "101281701": {"name": "东莞", "coord": "113.746262,23.046237"},
    "101281801": {"name": "中山", "coord": "113.382391,22.521113"},
    "101281901": {"name": "潮州", "coord": "116.632301,23.661701"},
    "101282001": {"name": "揭阳", "coord": "116.355733,23.543778"},
    "101282101": {"name": "云浮", "coord": "112.044439,22.929801"}
}

# Person names assigned to the generated users.
NAMES: list[str] = [
    "张伟", "李明", "王芳", "刘强", "陈浩", "杨静", "赵丽", "孙磊", "周敏", "吴静",
    "徐娜", "朱勇", "马超", "冯亮", "邓华", "林雪", "郑涛", "黄伟", "许飞", "何敏",
    "苏杰", "潘蕾", "谢军", "董琳", "薛斌", "石磊", "罗娟", "韩超", "彭丽", "贾伟",
    "卢明", "曹静", "蒋华", "田勇", "余敏", "叶杰", "程蕾", "魏军", "方琳", "任斌"
]

# Attractions and landmarks per city, used to build route questions.
CITY_LANDMARKS: dict[str, list[str]] = {
    "北京": ["天安门", "故宫", "颐和园", "长城", "鸟巢", "水立方", "天坛", "雍和宫", "北海公园", "景山公园",
           "圆明园", "恭王府", "什刹海", "南锣鼓巷", "798艺术区", "首都机场", "北京南站", "北京西站", "清华大学", "北京大学",
           "中关村", "王府井", "西单", "三里屯", "朝阳公园", "奥林匹克公园", "香山公园", "明十三陵", "八达岭长城", "慕田峪长城"],
    "上海": ["外滩", "东方明珠", "上海迪士尼", "豫园", "南京路", "陆家嘴", "城隍庙", "田子坊", "新天地", "淮海路",
           "上海博物馆", "上海科技馆", "世博园", "朱家角", "七宝古镇", "浦东机场", "虹桥机场", "上海站", "虹桥站", "复旦大学",
           "交通大学", "同济大学", "人民广场", "静安寺", "徐家汇", "中山公园", "世纪公园", "上海野生动物园", "东方绿舟", "佘山"],
    "广州": ["广州塔", "陈家祠", "白云山", "长隆欢乐世界", "沙面", "珠江夜游", "越秀公园", "中山纪念堂", "荔枝湾", "北京路",
           "上下九步行街", "花城广场", "海心沙", "黄埔军校", "南沙湿地", "白云机场", "广州南站", "广州站", "中山大学", "华南理工大学",
           "天河城", "正佳广场", "太古汇", "岭南印象园", "长隆野生动物园", "长隆水上乐园", "宝墨园", "余荫山房", "从化温泉", "增城白水寨"],
    "深圳": ["世界之窗", "欢乐谷", "大梅沙", "小梅沙", "莲花山公园", "华强北", "深圳湾公园", "仙湖植物园", "东门老街", "海上世界",
           "中英街", "大鹏所城", "较场尾", "西冲海滩", "东冲海滩", "宝安机场", "深圳北站", "深圳站", "深圳大学", "南方科技大学",
           "平安金融中心", "京基100", "地王大厦", "红树林", "笔架山", "梧桐山", "大小南山", "蛇口", "福田口岸", "罗湖口岸"],
    "杭州": ["西湖", "雷峰塔", "灵隐寺", "千岛湖", "宋城", "西溪湿地", "断桥", "苏堤", "白堤", "三潭印月",
           "岳王庙", "六和塔", "飞来峰", "虎跑泉", "龙井茶园", "萧山机场", "杭州东站", "杭州站", "浙江大学", "杭州师范大学",
           "河坊街", "南宋御街", "武林广场", "湖滨", "钱塘江", "九溪十八涧", "云栖竹径", "满陇桂雨", "柳浪闻莺", "曲院风荷"],
    "南京": ["中山陵", "夫子庙", "总统府", "玄武湖", "紫金山", "秦淮河", "明孝陵", "雨花台", "莫愁湖", "栖霞山",
           "鸡鸣寺", "阅江楼", "中华门", "老门东", "1912街区", "禄口机场", "南京南站", "南京站", "南京大学", "东南大学",
           "新街口", "湖南路", "山西路", "燕子矶", "汤山温泉", "珍珠泉", "红山森林动物园", "南京博物院", "江宁织造博物馆", "六朝博物馆"]
}

# Fallback landmarks that exist in every city.
GENERIC_LANDMARKS: list[str] = [
    "机场", "火车站", "汽车站", "地铁站", "医院", "银行", "商场", "超市", "公园", "图书馆",
    "博物馆", "体育馆", "大学", "中学", "小学", "酒店", "餐厅", "咖啡厅", "电影院", "KTV"
]

# Workflow 1 - travel planning that names a destination, so no follow-up is needed.
TRAVEL_PLANNING_NO_ASK_QUESTIONS: list[str] = [
    "我想去{}旅游，帮我制定旅行计划",
    "{}有什么好玩的景点推荐？",
    "帮我规划一下去{}的行程",
    "想去{}玩，有什么推荐的路线吗？",
    "{}旅游攻略有哪些？",
    "去{}应该怎么安排行程？",
    "{}有哪些必去的景点？",
    "我计划去{}，帮我推荐一些好玩的地方",
    "{}的旅游景点有哪些值得去的？",
    "想了解一下{}的旅游信息",
    "{}有什么特色美食和景点？",
    "我要去{}度假，求推荐",
    "{}自由行攻略怎么做？",
    "第一次去{}，有什么建议吗？",
    "{}周边游有什么好玩的？",
    "我想在{}旅游，怎么安排比较好？",
    "{}的经典旅游路线是什么？",
    "去{}玩什么季节最合适？",
    "{}当地有什么特色活动？",
    "我对{}很感兴趣，能介绍一下吗？",
    "{}的历史文化景点有哪些？",
    "想去{}拍照，有什么网红景点？",
    "{}适合亲子游的地方有哪些？",
    "我想去{}看风景，推荐几个地方",
    "{}的夜生活怎么样？",
    "想体验{}的当地文化",
    "{}有什么购物的地方？",
    "我计划{}月份去{}，合适吗？",
    "{}的交通方便吗？怎么玩比较好？",
    "听说{}很美，能帮我规划一下吗？"
]

# Workflow 1 - vague travel planning, plus the destinations the follow-up elicits.
TRAVEL_PLANNING_ASK_QUESTIONS: list[str] = [
    "我想出去旅游，有什么推荐的地方吗？",
    "帮我规划一下行程",
    "想出去玩几天，去哪里比较好？",
    "有什么好的旅游目的地推荐吗？",
    "我想旅行，但不知道去哪里",
    "帮我推荐一个旅游地点",
    "想找个地方度假，有推荐吗？",
    "计划出游，有什么建议吗？",
    "我要出去散心，去哪里好？",
    "想要一个完美的假期，推荐个地方",
    "我想放松一下，去哪里旅游好？",
    "有什么适合周末游的地方？",
    "想找个安静的地方旅行",
    "我想看海，有推荐吗？",
    "想去爬山，哪里比较好？",
    "有什么古镇推荐吗？",
    "我想体验不同的文化",
    "想去温泉度假",
    "有什么适合拍照的地方？",
    "我想吃美食，去哪里好？",
    "想去看雪，推荐个地方",
    "有什么浪漫的旅行地点？",
    "我想去历史悠久的城市",
    "想体验现代都市生活",
    "有什么适合家庭游的地方？",
    "我想去自然风光好的地方",
    "想找个不太商业化的地方",
    "有什么小众旅行地推荐？",
    "我想去感受慢生活",
    "想找个有特色的地方旅行"
]
TRAVEL_PLANNING_ASK_ANSWERS: list[str] = [
    "我想去西安", "想去厦门看看", "我想去三亚", "想去桂林", "我想去青岛",
    "想去大连", "我想去苏州", "想去无锡", "我想去宁波", "想去温州",
    "我想去重庆", "想去哈尔滨", "我想去长春", "想去沈阳", "我想去银川",
    "想去西宁", "我想去兰州", "想去乌鲁木齐", "我想去拉萨", "想去海口",
    "我想去合肥", "想去芜湖", "我想去蚌埠", "想去淮南", "我想去马鞍山",
    "想去铜陵", "我想去安庆", "想去黄山", "我想去滁州", "想去阜阳"
]

# Workflow 2 - route questions that already name their endpoints.
ROUTE_NO_ASK_QUESTIONS: list[str] = [
    "从{}到{}怎么走？",
    "去{}怎么走？",
    "{}到{}的路线",
    "如何到达{}？",
    "从{}怎么去{}？",
    "{}的具体路线",
    "怎么去{}？",
    "到{}的交通路线",
    "从这里到{}怎么走？",
    "{}在哪里，怎么去？",
    "我想去{}，请问路线",
    "{}怎么坐车去？",
    "从{}出发到{}",
    "请问{}怎么走？",
    "{}的地址在哪里？",
    "我要到{}，坐什么车？",
    "{}离这里远吗？",
    "去{}坐地铁怎么走？",
    "从{}打车到{}要多久？",
    "{}附近有地铁站吗？",
    "我在{}，想去{}",
    "{}到{}开车怎么走？",
    "去{}坐公交怎么走？",
    "{}的交通方便吗？",
    "从{}步行到{}要多久？",
    "{}有直达的公交吗？",
    "我要去{}，最快的路线",
    "{}到{}有多远？",
    "去{}的最佳路线是什么？",
    "从{}到{}需要换乘吗？"
]

# Workflow 2 - route questions with an implicit destination, grouped by the kind of
# place the follow-up answer resolves to.
ROUTE_ASK_QUESTION_ANSWER_PAIRS: list[tuple[list[str], list[str]]] = [
    # Going home
    (["怎么回家？", "我要回家", "回家的路线", "我想回家", "怎么回去？"],
     ["回天河区珠江新城", "回海淀区中关村", "回浦东新区陆家嘴", "回福田中心区", "回西湖区文三路"]),

    # Going to school
    (["怎么回学校？", "我要去学校", "去学校怎么走？", "我要回学校", "学校在哪里？"],
     ["去北京大学", "去清华大学", "去复旦大学", "去中山大学", "去浙江大学"]),

    # Going to the airport
    (["我要去机场", "怎么去机场？", "机场在哪里？", "去机场怎么走？", "机场路线"],
     ["首都国际机场", "浦东国际机场", "白云国际机场", "宝安国际机场", "萧山国际机场"]),

    # Going to the railway station
    (["我要去火车站", "火车站怎么走？", "去车站的路", "火车站在哪？", "我要坐火车"],
     ["北京南站", "上海虹桥站", "广州南站", "深圳北站", "杭州东站"]),

    # Going to a hospital
    (["我要去医院", "最近的医院在哪？", "去医院怎么走？", "医院在哪里？", "我要看病"],
     ["协和医院", "华西医院", "中山医院", "人民医院", "第一人民医院"]),

    # Going to a bank
    (["我要去银行", "附近的银行", "银行怎么走？", "最近的银行在哪？", "我要取钱"],
     ["中国银行", "建设银行", "工商银行", "农业银行", "招商银行"])
]

# Workflow 3 - asking for a hotel recommendation.
HOTEL_RECOMMEND_QUESTIONS: list[str] = [
    "推荐一些{}{}元的酒店",
    "{}有什么好的酒店推荐？",
    "我要在{}住酒店，预算{}元",
    "{}附近有什么酒店？",
    "{}的酒店哪家比较好？",
    "想在{}找个酒店，价位{}元左右",
    "{}有性价比高的酒店吗？",
    "{}地区的酒店推荐",
    "我要在{}订酒店，{}元预算",
    "{}有什么五星级酒店？",
    "{}的经济型酒店有哪些？",
    "想在{}住一晚，有推荐吗？",
    "{}商务酒店哪家好？",
    "我要在{}找个干净的酒店",
    "{}有什么特色酒店？",
    "{}的连锁酒店有哪些？",
    "想在{}找个安静的酒店",
    "{}有什么豪华酒店？",
    "我要在{}住几天，推荐酒店",
    "{}的民宿怎么样？",
    "想在{}找个交通便利的酒店",
    "{}有什么新开的酒店？",
    "我要在{}找个有早餐的酒店"
]

# Workflow 3 - asking what a specific hotel is like.
HOTEL_REVIEW_QUESTIONS: list[str] = [
    "{}怎么样？", "{}的评价如何？", "{}这个酒店好不好？",
    "{}值得住吗？", "{}的服务怎么样？", "{}干净吗？",
    "{}的设施如何？", "{}性价比高吗？", "{}的房间大吗？",
    "{}的位置好吗？", "{}的早餐怎么样？", "{}的前台服务好吗？",
    "{}的网络好吗？", "{}有停车位吗？", "{}的装修新吗？",
    "{}的隔音效果好吗？", "{}的卫生间干净吗？", "{}的床舒服吗？",
    "{}的空调好用吗？", "{}的电梯快吗？", "{}的周边环境怎么样？",
    "{}适合商务出差吗？", "{}适合家庭入住吗？", "{}的性价比如何？"
]

# Hotel brands substituted into the review questions.
HOTEL_NAMES: list[str] = [
    "如家酒店", "汉庭酒店", "7天酒店", "格林豪泰", "锦江之星",
    "维也纳酒店", "全季酒店", "桔子酒店", "亚朵酒店", "希尔顿酒店",
    "万豪酒店", "喜来登酒店", "香格里拉酒店", "凯悦酒店", "洲际酒店"
]

# Cities and nightly budget bands substituted into the recommendation questions.
HOTEL_CITIES: list[str] = ["北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "成都"]
HOTEL_BUDGETS: list[str] = ["200-300", "300-500", "500-800", "800-1200", "1200-2000"]

# Workflow 3 - hotel requests with no city or budget, plus the follow-up answers.
HOTEL_ASK_QUESTIONS: list[str] = [
    "我需要订酒店", "帮我找个酒店", "我要住酒店",
    "推荐个酒店", "找个地方住", "需要住宿",
    "帮我找个住的地方", "我要预订酒店", "找个酒店住",
    "帮我订个房间", "我想住酒店", "需要找个酒店",
    "我要找住宿", "帮我安排住宿", "我需要住的地方",
    "想订个酒店", "我要找个宾馆", "需要预订房间",
    "帮我找住宿", "我想找个酒店", "需要安排住宿",
    "我要住宿", "帮我订住宿", "想找个住的地方",
    "我需要房间", "帮我找房间", "我要订房",
    "需要找住宿", "我想预订酒店", "帮我安排酒店"
]
HOTEL_ASK_ANSWERS: list[str] = [
    "北京市中心，预算300-500元", "上海浦东新区，预算600-1000元",
    "广州天河区，预算400-600元", "深圳南山区，预算500-800元",
    "杭州西湖附近，预算400-700元", "南京市中心，预算300-500元",
    "武汉汉口，预算200-400元", "成都春熙路附近，预算300-600元",
    "重庆解放碑，预算350-550元", "天津滨海新区，预算280-450元",
    "西安钟楼附近，预算250-400元", "青岛市南区，预算400-650元",
    "大连中山区，预算350-600元", "厦门思明区，预算500-800元",
    "苏州园区，预算300-500元", "无锡新区，预算280-480元",
    "宁波江北区，预算320-520元", "温州鹿城区，预算300-500元",
    "合肥政务区，预算250-450元", "福州台江区，预算280-480元",
    "南昌红谷滩，预算200-400元", "长沙五一广场，预算250-450元",
    "贵阳观山湖区，预算200-350元", "昆明五华区，预算280-480元",
    "哈尔滨道里区，预算200-400元", "长春朝阳区，预算180-350元"
]

# Workflow 4 - travel-related small talk and general travel advice.
TRAVEL_CHAT_QUESTIONS: list[str] = [
    "你好，我是第一次使用旅行助手", "你好", "早上好", "晚上好", "您好",
    "旅行的时候需要注意什么安全问题？", "出国旅行需要准备什么？",
    "旅行保险重要吗？", "一个人旅行安全吗？", "旅行预算怎么控制？",
    "什么季节旅行最好？", "旅行必备物品有哪些？", "如何选择旅行目的地？",
    "旅行中如何省钱？", "自由行好还是跟团好？", "旅行拍照技巧有哪些？",
    "如何克服旅行中的语言障碍？", "旅行中生病了怎么办？",
    "旅行中的饮食安全怎么保证？", "如何避免旅行中的购物陷阱？",
    "旅行中如何保护个人财物？", "什么是深度旅行？", "如何制定旅行计划？",
    "旅行时如何选择合适的交通工具？", "旅行中如何与当地人交流？",
    "什么是穷游？", "背包客需要注意什么？", "旅行中如何应对突发情况？",
    "如何选择旅行伙伴？", "旅行中如何保持健康？", "什么是文化旅行？",
    "旅行中如何尊重当地文化？", "如何避免旅行疲劳？", "旅行中如何记录美好回忆？",
    "什么是生态旅行？", "旅行中如何环保？", "如何选择旅行装备？",
    "旅行中如何应对时差？", "什么是医疗旅游？", "如何选择旅行保险？",
    "旅行中如何使用手机？", "什么是美食旅行？", "如何体验当地美食？",
    "旅行中如何购买纪念品？", "什么是摄影旅行？", "如何拍出好的旅行照片？",
    "旅行中如何节约时间？", "什么是主题旅行？", "如何规避旅行风险？",
    "旅行中如何保持联系？", "什么是慢旅行？", "如何享受旅行过程？",
    "旅行回来后如何整理照片？", "如何分享旅行经历？", "旅行对人生有什么意义？",
    "如何培养旅行兴趣？", "旅行中最重要的是什么？", "你能介绍一下你的功能吗？",
    "感谢你的帮助", "你真厉害", "你的建议很有用", "谢谢你", "再见",
    "旅行助手有什么功能？", "你能帮我做什么？", "你是怎么工作的？"
]

# Workflow 5 - off-topic questions the assistant must decline.
REJECT_QUESTIONS: list[str] = [
    "帮我算一下1+1等于几？", "今天天气怎么样？", "你能帮我写一段Python代码吗？",
    "什么是人工智能？", "如何学习编程？", "推荐几本好书",
    "今天股市怎么样？", "帮我翻译一段英文", "什么是区块链？",
    "如何做红烧肉？", "推荐几部电影", "如何减肥？",
    "什么是量子计算？", "如何投资理财？", "推荐几首歌曲",
    "如何学习英语？", "什么是机器学习？", "如何写简历？",
    "推荐几个游戏", "如何护肤？", "什么是大数据？",
    "帮我解这道数学题", "什么是深度学习？", "如何学习日语？",
    "推荐几个购物网站", "如何修电脑？", "什么是云计算？",
    "如何做蛋糕？", "推荐几个音乐软件", "如何学开车？",
    "什么是5G技术？", "如何炒股？", "推荐几个新闻APP",
    "如何治感冒？", "什么是物联网？", "如何学画画？",
    "推荐几个学习网站", "如何做生意？", "什么是虚拟现实？",
    "如何写小说？", "推荐几个视频软件", "如何学跳舞？",
    "什么是增强现实？", "如何创业？", "推荐几个社交软件",
    "如何养宠物？", "什么是区块链技术？", "如何学乐器？",
    "推荐几个办公软件", "如何做运动？", "什么是元宇宙？",
    "如何写诗？", "推荐几个理财软件", "如何学摄影？",
    "什么是人脸识别？", "如何做饭？", "推荐几个购物APP",
    "如何学化妆？", "什么是自动驾驶？", "如何写作文？",
    "推荐几个健身APP", "如何种花？", "什么是智能家居？",
    "如何学书法？", "推荐几个外卖软件", "如何做手工？"
]


class DatasetGenerator:
    """Generates the synthetic seed dataset, one method per workflow.

    Each generator method returns a list of rows keyed by the Chinese field names
    used on disk, and reports progress through a shared counter so the methods can
    run concurrently.
    """

    def __init__(self) -> None:
        self.cities = CITIES
        self.names = NAMES

        # Destination pool for questions that name the city being travelled to.
        self.destinations = list(self.cities.values())
        self.destination_names = [city_info["name"] for city_info in self.destinations]

        self.city_landmarks = CITY_LANDMARKS
        self.generic_landmarks = GENERIC_LANDMARKS

        self.lock = threading.Lock()
        self.progress: dict[str, int] = {"completed": 0, "total": TOTAL_SAMPLES}

    def get_random_date(self) -> str:
        """Pick a departure date uniformly from the generation window.

        Returns:
            The date formatted as ``YYYY-MM-DD``.
        """
        days_diff = (DEPARTURE_WINDOW_END - DEPARTURE_WINDOW_START).days
        random_days = random.randint(0, days_diff)
        return (DEPARTURE_WINDOW_START + timedelta(days=random_days)).strftime(DATE_FORMAT)

    def get_random_city_info(self) -> tuple[str, str]:
        """Pick a random city to use as the user's current location.

        Returns:
            A ``(city_id, coordinate)`` pair, where the coordinate is
            ``"longitude,latitude"``.
        """
        city_id = random.choice(list(self.cities.keys()))
        return city_id, self.cities[city_id]["coord"]

    def get_city_landmarks(self, city_name: str) -> list[str]:
        """Return the landmarks of a city.

        Args:
            city_name: Display name of the city.

        Returns:
            The city's own landmarks, or the generic landmark list when the city
            has no dedicated entry.
        """
        return self.city_landmarks.get(city_name, self.generic_landmarks)

    def get_random_city_with_landmarks(self) -> tuple[str, str, str, list[str]]:
        """Pick a random city together with its landmarks.

        Returns:
            A ``(city_id, coordinate, city_name, landmarks)`` tuple.
        """
        city_id = random.choice(list(self.cities.keys()))
        city_name = self.cities[city_id]["name"]
        coord = self.cities[city_id]["coord"]
        landmarks = self.get_city_landmarks(city_name)
        return city_id, coord, city_name, landmarks

    def update_progress(self) -> None:
        """Count one finished row and reprint the progress line.

        Safe to call from several worker threads: the counter is guarded by a lock.
        """
        with self.lock:
            self.progress["completed"] += 1
            completed = self.progress["completed"]
            total = self.progress["total"]
            percentage = (completed / total) * 100
            print(f"\r进度: {completed}/{total} ({percentage:.1f}%)", end="", flush=True)

    def generate_travel_planning_no_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 1: travel planning where the destination is already given.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()
            destination_name = random.choice(self.destination_names)
            month = random.randint(1, 12)
            question_template = random.choice(TRAVEL_PLANNING_NO_ASK_QUESTIONS)

            if "{}月份" in question_template:
                question = question_template.format(month, destination_name)
            elif question_template.count("{}") >= 1:
                question = question_template.format(destination_name)
            else:
                question = question_template

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 1,
                "用户问题": question,
                "是否追问": "否",
                "追问回答": None
            })
            self.update_progress()
        return rows

    def generate_travel_planning_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 1: travel planning that needs a follow-up for the destination.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 1,
                "用户问题": random.choice(TRAVEL_PLANNING_ASK_QUESTIONS),
                "是否追问": "是",
                "追问回答": random.choice(TRAVEL_PLANNING_ASK_ANSWERS)
            })
            self.update_progress()
        return rows

    def generate_route_no_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 2: route questions that already name their endpoints.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord, _city_name, landmarks = self.get_random_city_with_landmarks()

            # Mostly real landmarks; the rest are generic ones that stand in for
            # places the geocoder cannot resolve.
            if random.random() < REAL_LANDMARK_RATIO:
                start_place = random.choice(landmarks)
                end_place = random.choice(landmarks)
            else:
                start_place = random.choice(self.generic_landmarks)
                end_place = random.choice(self.generic_landmarks)

            question_template = random.choice(ROUTE_NO_ASK_QUESTIONS)
            if question_template.count("{}") == 2:
                question = question_template.format(start_place, end_place)
            else:
                question = question_template.format(end_place)

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 2,
                "用户问题": question,
                "是否追问": "否",
                "追问回答": None
            })
            self.update_progress()
        return rows

    def generate_route_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 2: route questions whose destination needs a follow-up.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            # Draw a question and an answer from the same place category.
            questions, answers = random.choice(ROUTE_ASK_QUESTION_ANSWER_PAIRS)
            question = random.choice(questions)
            answer = random.choice(answers)

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 2,
                "用户问题": question,
                "是否追问": "是",
                "追问回答": answer
            })
            self.update_progress()
        return rows

    def generate_hotel_no_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 3: hotel recommendation and hotel review questions.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            if random.random() < HOTEL_RECOMMEND_RATIO:
                city_name = random.choice(HOTEL_CITIES)
                budget = random.choice(HOTEL_BUDGETS)
                question_template = random.choice(HOTEL_RECOMMEND_QUESTIONS)

                if "{}" in question_template and "{}元" in question_template:
                    question = question_template.format(city_name, budget)
                elif "{}" in question_template:
                    question = question_template.format(city_name)
                else:
                    question = question_template
            else:
                hotel_name = random.choice(HOTEL_NAMES)
                question = random.choice(HOTEL_REVIEW_QUESTIONS).format(hotel_name)

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 3,
                "用户问题": question,
                "是否追问": "否",
                "追问回答": None
            })
            self.update_progress()
        return rows

    def generate_hotel_ask(self, count: int) -> list[dict[str, Any]]:
        """Workflow 3: hotel requests missing the city or the budget.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 3,
                "用户问题": random.choice(HOTEL_ASK_QUESTIONS),
                "是否追问": "是",
                "追问回答": random.choice(HOTEL_ASK_ANSWERS)
            })
            self.update_progress()
        return rows

    def generate_travel_chat(self, count: int) -> list[dict[str, Any]]:
        """Workflow 4: travel-related small talk.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 4,
                "用户问题": random.choice(TRAVEL_CHAT_QUESTIONS),
                "是否追问": "否",
                "追问回答": None
            })
            self.update_progress()
        return rows

    def generate_reject(self, count: int) -> list[dict[str, Any]]:
        """Workflow 5: off-topic questions the assistant should decline.

        Args:
            count: Number of rows to generate.

        Returns:
            The generated rows.
        """
        rows: list[dict[str, Any]] = []
        for _ in range(count):
            city_id, coord = self.get_random_city_info()

            rows.append({
                "用户名字": random.choice(self.names),
                "用户所处城市": city_id,
                "出发日期": self.get_random_date(),
                "起点坐标": coord,
                "工作流": 5,
                "用户问题": random.choice(REJECT_QUESTIONS),
                "是否追问": "否",
                "追问回答": None
            })
            self.update_progress()
        return rows

    def generate_dataset(self) -> list[dict[str, Any]]:
        """Generate every workflow concurrently and save the shuffled result.

        The rows are written to ``SEED_DIR`` as
        ``travel_assistant_dataset_<timestamp>.json``, and a per-workflow summary
        is printed.

        Returns:
            All generated rows, in the same shuffled order as the saved file.
        """
        print("开始生成数据集...")

        tasks: list[tuple[Callable[[int], list[dict[str, Any]]], int]] = [
            (getattr(self, method_name), count) for method_name, count in TASK_COUNTS
        ]

        all_rows: list[dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(generate_rows, count) for generate_rows, count in tasks]

            for future in as_completed(futures):
                all_rows.extend(future.result())

        # Interleave the workflows instead of leaving them grouped by task.
        random.shuffle(all_rows)

        print(f"\n数据生成完成！总计 {len(all_rows)} 条")

        timestamp = datetime.now().strftime(FILE_TIMESTAMP_FORMAT)
        output_path = ensure_dir(SEED_DIR) / f"travel_assistant_dataset_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)

        print(f"数据已保存到: {output_path}")

        workflow_counts: dict[int, int] = {}
        for row in all_rows:
            workflow_id = row["工作流"]
            workflow_counts[workflow_id] = workflow_counts.get(workflow_id, 0) + 1

        print("\n数据统计:")
        for workflow_id, count in sorted(workflow_counts.items()):
            print(f"工作流{workflow_id}({WORKFLOW_NAMES[workflow_id]}): {count}条")

        return all_rows


def main() -> None:
    """Generate the dataset and write it to ``SEED_DIR``."""
    DatasetGenerator().generate_dataset()


if __name__ == "__main__":
    main()
