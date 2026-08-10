# -*- coding: utf-8 -*-
"""Единый формат данных между агентами — датакласс Business."""

from dataclasses import asdict, dataclass
from typing import Optional

import config
from contactability import Contactability, ContactChannel, contactability_from_business
from website_pipeline import LeadDecision


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
    category: str = ""                  # категория из карточки, если доступна
    email: str = ""                     # email источника, если доступен
    website: str = ""                   # сайт (если указан в профиле Maps)
    instagram_url: str = ""             # ссылка на Instagram (если есть)
    google_maps_url: str = ""           # URL исходной карточки Maps
    google_place_id: str = ""           # Google Place ID, если доступен
    external_candidate_id: str = ""     # стабильный ID внешнего кандидата
    collected_at: str = ""              # время сбора ISO-8601
    rating: float = 0.0                 # рейтинг на Maps
    reviews_count: int = 0              # количество отзывов на Maps

    # --- результаты site_checker ---
    has_site: bool = False              # есть ли живой собственный сайт
    site_quality: str = "none"          # none (нет) / dead (не открывается) / bad / good

    # --- результаты website resolver / audit ---
    website_original_url: str = ""
    instagram_bio_url: str = ""
    website_resolved_url: str = ""
    website_final_url: str = ""
    website_resolution_status: str = ""
    website_resolution_source: str = ""
    website_resolution_confidence: float = 0.0
    website_resolution_evidence: str = ""
    website_resolution_error: str = ""
    website_audit_status: str = ""
    website_audit_http_status: Optional[int] = None
    website_audit_evidence: str = ""
    website_audit_error: str = ""
    lead_decision: str = ""
    lead_decision_reason: str = ""

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
        if self.site_quality == "good":
            return "good website"
        if self.site_quality == "bad":
            return "bad website"
        if self.site_quality in {"uncertain", "technical_error"}:
            return "uncertain website"
        return "no website"

    @property
    def effective_website_url(self) -> str:
        """URL selected for audit and evidence-aware reporting."""
        return self.website_resolved_url or self.website_original_url or self.website

    @property
    def contactability(self) -> Contactability:
        return contactability_from_business(self)

    @property
    def has_actionable_contact(self) -> bool:
        return self.contactability.actionable

    @property
    def preferred_contact_channel(self) -> ContactChannel | None:
        return self.contactability.preferred_channel

    @property
    def is_lead(self) -> bool:
        """Require website need and the contact allowed by the active mode."""
        if self.lead_decision:
            return self.lead_decision == LeadDecision.LEAD.value
        if config.LEAD_CONTACTABILITY_MODE == "instagram_only":
            has_contact = bool(self.instagram_url)
        else:
            has_contact = self.has_actionable_contact
        if not has_contact:
            return False
        return self.website_status in ("no website", "bad website")

    def to_dict(self) -> dict:
        """Словарь для записи в базу/CSV."""
        return asdict(self)
