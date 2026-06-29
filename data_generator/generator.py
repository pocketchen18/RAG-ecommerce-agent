# -*- coding: utf-8 -*-
"""数据生成器 - 生成真实手机商品与评价数据"""
import sys
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# 强制 sys.stdout 和 sys.stderr 使用 UTF-8 编码，防止在 Windows GBK 终端下打印 emoji 崩溃
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from .schemas import PhoneProduct, Review
from .real_phones import REAL_PHONES


# ============================================================
# 评价模板数据 (用于生成拟真的用户评论)
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


def generate_configured_phone(base_phone: dict, config_idx: int, global_index: int) -> PhoneProduct:
    """基于真实手机模板，生成不同内存/存储配置的 SKU"""
    brand = base_phone["brand"]
    series = base_phone["series"]
    model_name = base_phone["model_name"]
    base_price = base_phone["base_price"]
    
    # 针对苹果与安卓分别生成不同存储阶梯价格
    if brand == "苹果":
        if base_price >= 9999.0: # Pro Max
            storage_options = [256, 512, 1024]
            price_offsets = [0.0, 1500.0, 3000.0]
            ram = 8
        elif "Pro" in model_name: # Pro
            storage_options = [128, 256, 512]
            price_offsets = [0.0, 1000.0, 2000.0]
            ram = 8
        else: # Standard
            storage_options = [128, 256, 512]
            price_offsets = [0.0, 1000.0, 2000.0]
            ram = 6 if "15" in model_name else 8
            
        storage = storage_options[config_idx]
        price = base_price + price_offsets[config_idx]
    else:
        # 安卓存储规则
        if base_price < 2000.0:
            ram_options = [8, 12, 16]
            storage_options = [128, 256, 512]
            price_offsets = [0.0, 300.0, 600.0]
        elif base_price < 5000.0:
            ram_options = [12, 16, 16]
            storage_options = [256, 512, 1024]
            price_offsets = [0.0, 400.0, 900.0]
        else:
            ram_options = [12, 16, 16]
            storage_options = [256, 512, 1024]
            price_offsets = [0.0, 500.0, 1200.0]
            
        ram = ram_options[config_idx]
        storage = storage_options[config_idx]
        price = base_price + price_offsets[config_idx]
        
    storage_str = f"{storage}GB" if storage < 1024 else "1TB"
    name = f"{brand} {model_name} ({ram}GB+{storage_str})"
    
    # 原价（有概率比现价略高）
    original_price = price + random.choice([0.0, 200.0, 300.0, 500.0]) if random.random() > 0.5 else None
    
    phone = PhoneProduct(
        sku_id=generate_sku_id(brand, series, global_index),
        name=name,
        brand=brand,
        series=series,
        price=price,
        original_price=original_price,
        screen_type=base_phone["screen_type"],
        screen_size=base_phone["screen_size"],
        screen_resolution=base_phone["screen_resolution"],
        refresh_rate=base_phone["refresh_rate"],
        processor=base_phone["processor"],
        ram=ram,
        storage=storage,
        camera_main=base_phone["camera_main"],
        camera_ultra_wide=base_phone["camera_ultra_wide"],
        camera_telephoto=base_phone["camera_telephoto"],
        camera_front=base_phone["camera_front"],
        battery=base_phone["battery"],
        charging=base_phone["charging"],
        weight=base_phone["weight"],
        os=base_phone["os"],
        reviews=[],
        tags=base_phone["tags"],
        description=base_phone["description"],
        release_date=base_phone["release_date"]
    )
    
    # 生成评价
    num_reviews = random.randint(6, 12)
    review_features = random.sample(FEATURES, k=min(3, len(FEATURES)))
    phone.reviews = [generate_review((2023, 2025), review_features) for _ in range(num_reviews)]
    
    return phone


def generate_phones(count: int = 120) -> List[PhoneProduct]:
    """根据真实手机模板列表生成多款手机 SKU"""
    phones = []
    
    # 循环遍历 REAL_PHONES，对每款机型生成不同配置的 SKU
    for config_idx in range(3): # 对每款机型支持生成至多 3 种存储配置
        for base_phone in REAL_PHONES:
            if len(phones) >= count:
                break
            global_idx = len(phones) + 1
            phone = generate_configured_phone(base_phone, config_idx, global_idx)
            phones.append(phone)
            
    # 随机打乱以增加数据真实度和分布性
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
