# -*- coding: utf-8 -*-
"""Единый формат данных между агентами — датакласс Business."""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Business:
    """Один бизнес, найденный в Google Maps, со всеми результатами проверок."""

    # --- базовые данные из Google Maps ---
    id: Optional[int] = None            # id в базе SQLite
    name: str = ""                      # название бизнеса
    niche: str = ""                     # ниша (салон краси, барбершоп...)
    city: str = ""                      # город поиска
    phone: str = ""                     # телефон
    address: str = ""                   # адрес
    website: str = ""                   # сайт (если указан в профиле Maps)
    instagram_url: str = ""             # ссылка на Instagram (если есть)
    rating: float = 0.0                 # рейтинг на Maps
    reviews_count: int = 0              # количество отзывов на Maps

    # --- результаты site_checker ---
    has_site: bool = False              # есть ли живой сайт
    site_quality: str = "none"          # none / unknown / old / bad / good

    # --- результаты social_checker ---
    instagram_active: bool = False      # активен ли Instagram
    followers: int = 0                  # подписчики
    posts_count: int = 0                # количество постов
    last_post_days: Optional[int] = None  # дней с последнего поста (None = неизвестно)

    # --- результаты ai_scorer ---
    ai_score: int = 0                   # оценка 0-100
    ai_priority: str = ""               # hot / warm / cold
    ai_reason: str = ""                 # объяснение оценки

    # --- служебное ---
    task_id: Optional[int] = None       # id задачи поиска

    def to_dict(self) -> dict:
        """Словарь для записи в базу/CSV."""
        return asdict(self)
