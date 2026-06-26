"""数据模型定义 - 手机商品与评价"""
from pydantic import BaseModel, Field
from typing import List, Optional


class Review(BaseModel):
    """用户评价模型"""
    review_id: str = Field(..., description="评价唯一ID")
    username: str = Field(..., description="用户名")
    rating: int = Field(..., ge=1, le=5, description="评分 1-5")
    content: str = Field(..., description="评价内容")
    date: str = Field(..., description="评价日期 YYYY-MM-DD")
    likes: int = Field(default=0, ge=0, description="点赞数")


class PhoneProduct(BaseModel):
    """手机商品模型"""
    sku_id: str = Field(..., description="商品唯一ID")
    name: str = Field(..., description="商品名称")
    brand: str = Field(..., description="品牌")
    series: Optional[str] = Field(None, description="系列")
    price: float = Field(..., gt=0, description="价格（元）")
    original_price: Optional[float] = Field(None, description="原价")

    # 基本参数
    screen_type: str = Field(..., description="屏幕类型（直屏/曲面屏/折叠屏）")
    screen_size: float = Field(..., description="屏幕尺寸（英寸）")
    screen_resolution: str = Field(default="2400x1080", description="屏幕分辨率")
    refresh_rate: int = Field(default=120, description="刷新率（Hz）")

    processor: str = Field(..., description="处理器型号")
    ram: int = Field(..., description="运行内存（GB）")
    storage: int = Field(..., description="存储容量（GB）")

    camera_main: str = Field(..., description="主摄像头像素")
    camera_ultra_wide: Optional[str] = Field(None, description="超广角像素")
    camera_telephoto: Optional[str] = Field(None, description="长焦像素")
    camera_front: str = Field(default="1600万", description="前置摄像头")

    battery: int = Field(..., description="电池容量（mAh）")
    charging: str = Field(default="67W快充", description="充电规格")
    weight: float = Field(..., description="重量（g）")
    os: str = Field(default="Android", description="操作系统")

    # 评价
    reviews: List[Review] = Field(default_factory=list, description="用户评价列表")

    # 元数据
    tags: List[str] = Field(default_factory=list, description="商品标签")
    description: Optional[str] = Field(None, description="商品描述")
    release_date: Optional[str] = Field(None, description="发布日期")

    class Config:
        json_schema_extra = {
            "example": {
                "sku_id": "HW-Mate60Pro-001",
                "name": "华为 Mate 60 Pro",
                "brand": "华为",
                "series": "Mate",
                "price": 6999,
                "screen_type": "曲面屏",
                "screen_size": 6.82,
                "processor": "麒麟9000S",
                "ram": 12,
                "storage": 256,
                "camera_main": "5000万",
                "battery": 5000,
                "charging": "88W快充",
                "weight": 225,
                "tags": ["旗舰", "拍照", "商务"]
            }
        }
