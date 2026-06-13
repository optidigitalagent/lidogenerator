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
    has_site: bool = False              # есть ли живой собственный сайт
    site_quality: str = "none"          # none (нет) / dead (не открывается) / bad / good

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

    @property
    def website_status(self) -> str:
        """Короткий статус сайта для таблицы и Telegram.

        no website  — сайта нет вообще, либо вместо сайта соцсеть/агрегатор,
                      либо указанный сайт не открывается (none / dead).
        bad website — сайт есть, но старый/слабый/неадаптивный (лид).
        good website — современный нормальный сайт (такой бизнес отсеивается).
        """
        if self.has_site and self.site_quality == "good":
            return "good website"
        if self.has_site and self.site_quality == "bad":
            return "bad website"
        return "no website"

    @property
    def is_lead(self) -> bool:
        """Лид = есть Instagram И (сайта нет ИЛИ сайт плохой).

        Бизнес с хорошим сайтом отсеивается, даже если Instagram отличный.
        Бизнес без Instagram отсеивается — связываться будем через Instagram.
        """
        if not self.instagram_url:
            return False
        return self.website_status in ("no website", "bad website")

    def to_dict(self) -> dict:
        """Словарь для записи в базу/CSV."""
        return asdict(self)
