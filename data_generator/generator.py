"""数据生成器 - 生成手机商品与评价数据"""
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from .schemas import PhoneProduct, Review


# ============================================================
# 品牌与机型数据
# ============================================================

BRANDS = {
    "华为": {
        "series": ["Mate", "Pura", "nova"],
        "processors": ["麒麟9000S", "麒麟9010", "骁龙8 Gen2"],
        "price_range": (2999, 8999),
        "tags": ["商务", "拍照", "旗舰", "鸿蒙"]
    },
    "小米": {
        "series": ["小米数字", "Redmi", "POCO"],
        "processors": ["骁龙8 Gen3", "骁龙8 Gen2", "天玑9200+"],
        "price_range": (1299, 5999),
        "tags": ["性价比", "旗舰", "游戏", "快充"]
    },
    "OPPO": {
        "series": ["Find", "Reno", "K"],
        "processors": ["骁龙8 Gen3", "天玑9200+", "骁龙7 Gen3"],
        "price_range": (1999, 6999),
        "tags": ["拍照", "快充", "颜值", "轻薄"]
    },
    "vivo": {
        "series": ["X", "S", "iQOO"],
        "processors": ["天玑9300", "骁龙8 Gen3", "天玑8200"],
        "price_range": (1999, 6999),
        "tags": ["拍照", "HiFi", "游戏", "轻薄"]
    },
    "苹果": {
        "series": ["iPhone"],
        "processors": ["A17 Pro", "A16 Bionic", "A15 Bionic"],
        "price_range": (5999, 12999),
        "tags": ["iOS", "流畅", "生态", "旗舰"]
    },
    "三星": {
        "series": ["Galaxy S", "Galaxy Z", "Galaxy A"],
        "processors": ["骁龙8 Gen3", "Exynos 2400", "骁龙7 Gen1"],
        "price_range": (1999, 12999),
        "tags": ["屏幕", "折叠屏", "拍照", "商务"]
    },
    "荣耀": {
        "series": ["Magic", "数字", "X"],
        "processors": ["骁龙8 Gen3", "天玑9200+", "骁龙6 Gen1"],
        "price_range": (1299, 5999),
        "tags": ["AI", "拍照", "轻薄", "性价比"]
    },
    "一加": {
        "series": ["一加数字", "一加Ace"],
        "processors": ["骁龙8 Gen3", "天玑9200+"],
        "price_range": (2299, 5499),
        "tags": ["性能", "流畅", "快充", "旗舰"]
    }
}

SCREEN_TYPES = ["直屏", "曲面屏", "折叠屏"]
CHARGING_SPECS = ["33W快充", "67W快充", "80W快充", "100W快充", "120W快充", "150W快充", "240W快充"]
CAMERA_CONFIGS = [
    {"main": "4800万", "ultra": "1200万", "tele": None},
    {"main": "5000万", "ultra": "800万", "tele": None},
    {"main": "5000万", "ultra": "1200万", "tele": "1200万"},
    {"main": "6400万", "ultra": "800万", "tele": None},
    {"main": "1亿", "ultra": "800万", "tele": "200万"},
    {"main": "2亿", "ultra": "1200万", "tele": "1000万"},
    {"main": "4800万", "ultra": "4800万", "tele": "1200万"},
    {"main": "5000万", "ultra": "5000万", "tele": "5000万"},
]


# ============================================================
# 评价模板数据
# ============================================================

REVIEW_TEMPLATES = {
    "positive": [
        "用了一周了，整体非常满意！{feature}表现超出预期，推荐购买。",
        "这个价位能买到这样的手机，性价比真的高。{feature}很给力。",
        "从{old_brand}换过来的，体验提升明显，尤其是{feature}。",
        "颜值在线，手感也不错。{feature}是亮点，日常使用完全够用。",
        "给家人买的，反馈都说好用。{feature}确实不错，值得推荐。",
        "已经是第N次买这个品牌了，{feature}一如既往地稳定。",
        "首发就入手了，{feature}真的惊艳到我了，好评！",
        "对比了好几款，最终选了这个。{feature}没让我失望。",
        "手机收到了，迫不及待试了一下{feature}，效果很好！",
        "用了三天来评价，{feature}确实可以，日常使用很流畅。"
    ],
    "neutral": [
        "手机还行，{feature}中规中矩，对得起这个价格。",
        "整体可以，就是{feature}感觉还有提升空间。",
        "日常使用没问题，{feature}够用，但追求极致可能差点意思。",
        "性价比不错，{feature}在这个价位算可以的了。",
        "手机没什么大毛病，{feature}符合预期，好评吧。"
    ],
    "negative": [
        "手机还行，但{feature}感觉一般，有点小失望。",
        "整体中规中矩，{feature}不太满意，希望能优化。",
        "用了一段时间，{feature}确实差点意思，给个中评吧。"
    ]
}

FEATURES = ["拍照", "性能", "续航", "充电速度", "屏幕显示", "系统流畅度", "手感", "音质", "信号", "发热控制"]
OLD_BRANDS = ["华为", "小米", "OPPO", "vivo", "苹果", "三星"]
USERNAMES = [
    "数码爱好者", "科技达人", "手机玩家", "极客小白", "理性消费者",
    "颜值党", "性能控", "摄影爱好者", "商务人士", "学生党",
    "游戏达人", "数码博主", "普通用户", "老用户", "新用户"
]


def generate_review_id() -> str:
    """生成评价ID"""
    return f"REV-{uuid.uuid4().hex[:8].upper()}"


def generate_sku_id(brand: str, series: str, index: int) -> str:
    """生成商品ID"""
    brand_short = {
        "华为": "HW", "小米": "MI", "OPPO": "OP", "vivo": "VV",
        "苹果": "AP", "三星": "SS", "荣耀": "HN", "一加": "OPP"
    }.get(brand, "XX")
    return f"{brand_short}-{series}-{index:03d}"


def generate_phone_name(brand: str, series: str, is_pro: bool = False) -> str:
    """生成手机名称"""
    suffix = " Pro" if is_pro else ""
    if series == "iPhone":
        models = ["15", "15 Plus", "15 Pro", "15 Pro Max", "16", "16 Plus", "16 Pro", "16 Pro Max"]
        return f"iPhone {random.choice(models)}"
    elif series == "Galaxy S":
        models = ["24", "24+", "24 Ultra"]
        return f"Samsung Galaxy S{random.choice(models)}"
    elif series == "Galaxy Z":
        return f"Samsung Galaxy Z {'Flip' if random.random() > 0.5 else 'Fold'} 5"
    else:
        num = random.randint(10, 99)
        return f"{brand} {series} {num}{suffix}"


def generate_review(date_range: tuple, features: List[str]) -> Review:
    """生成单条评价"""
    sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.6, 0.3, 0.1])[0]
    template = random.choice(REVIEW_TEMPLATES[sentiment])

    content = template.format(
        feature=random.choice(features),
        old_brand=random.choice(OLD_BRANDS)
    )

    days_offset = random.randint(0, 365)
    review_date = datetime.now() - timedelta(days=days_offset)

    return Review(
        review_id=generate_review_id(),
        username=random.choice(USERNAMES) + str(random.randint(1, 999)),
        rating=random.choices([5, 4, 3, 2, 1], weights=[0.4, 0.35, 0.15, 0.07, 0.03])[0],
        content=content,
        date=review_date.strftime("%Y-%m-%d"),
        likes=random.randint(0, 500)
    )


def generate_single_phone(brand: str, brand_config: dict, index: int) -> PhoneProduct:
    """生成单个手机商品"""
    series = random.choice(brand_config["series"])
    is_pro = random.random() > 0.6

    # 价格
    price_min, price_max = brand_config["price_range"]
    price = round(random.uniform(price_min, price_max) / 100) * 100

    # 屏幕配置
    screen_type = random.choices(SCREEN_TYPES, weights=[0.5, 0.4, 0.1])[0]
    if screen_type == "折叠屏":
        price = max(price, 6999)  # 折叠屏价格下限

    # 摄像头配置
    camera_config = random.choice(CAMERA_CONFIGS)

    # 生成商品
    phone = PhoneProduct(
        sku_id=generate_sku_id(brand, series, index),
        name=generate_phone_name(brand, series, is_pro),
        brand=brand,
        series=series,
        price=price,
        original_price=price + random.choice([0, 200, 300, 500]) if random.random() > 0.5 else None,
        screen_type=screen_type,
        screen_size=round(random.uniform(6.1, 7.2), 2),
        screen_resolution=random.choice(["2400x1080", "2772x1240", "3200x1440", "2844x1260"]),
        refresh_rate=random.choice([60, 90, 120, 144]),
        processor=random.choice(brand_config["processors"]),
        ram=random.choice([8, 12, 16]),
        storage=random.choice([128, 256, 512, 1024]),
        camera_main=camera_config["main"],
        camera_ultra_wide=camera_config["ultra"],
        camera_telephoto=camera_config["tele"],
        camera_front=f"{random.choice([12, 16, 20, 32])}00万",
        battery=random.choice([4000, 4500, 4800, 5000, 5500, 6000]),
        charging=random.choice(CHARGING_SPECS),
        weight=round(random.uniform(170, 240), 1),
        os="iOS" if brand == "苹果" else ("HarmonyOS" if brand == "华为" else "Android"),
        reviews=[],
        tags=random.sample(brand_config["tags"], k=min(3, len(brand_config["tags"]))),
        description=f"{brand}旗舰手机，搭载{random.choice(brand_config['processors'])}处理器，{camera_config['main']}像素主摄",
        release_date=f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}"
    )

    # 生成评价
    num_reviews = random.randint(5, 10)
    review_features = random.sample(FEATURES, k=min(3, len(FEATURES)))
    phone.reviews = [generate_review((2024, 2025), review_features) for _ in range(num_reviews)]

    return phone


def generate_phones(count: int = 120) -> List[PhoneProduct]:
    """生成多款手机数据"""
    phones = []
    brands_list = list(BRANDS.keys())

    # 确保每个品牌至少有 10 款
    phones_per_brand = count // len(brands_list)
    remainder = count % len(brands_list)

    for i, (brand, config) in enumerate(BRANDS.items()):
        brand_count = phones_per_brand + (1 if i < remainder else 0)
        for j in range(brand_count):
            phone = generate_single_phone(brand, config, j + 1)
            phones.append(phone)

    random.shuffle(phones)
    return phones


def save_to_json(phones: List[PhoneProduct], output_path: str = "data/products.json"):
    """保存数据到 JSON 文件"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = [phone.model_dump() for phone in phones]

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 已生成 {len(phones)} 款手机数据，保存到 {output_path}")
    return len(phones)


def main():
    """主函数"""
    print("🚀 开始生成手机商品与评价数据...")

    # 生成数据
    phones = generate_phones(count=120)

    # 保存数据
    count = save_to_json(phones)

    # 统计信息
    brands = set(p.brand for p in phones)
    prices = [p.price for p in phones]
    total_reviews = sum(len(p.reviews) for p in phones)

    print(f"\n📊 数据统计:")
    print(f"   - 总商品数: {count}")
    print(f"   - 覆盖品牌: {', '.join(brands)}")
    print(f"   - 价格区间: {min(prices):.0f} - {max(prices):.0f} 元")
    print(f"   - 总评价数: {total_reviews}")
    print(f"   - 平均评价数: {total_reviews/count:.1f} 条/款")

    return count


if __name__ == "__main__":
    main()
