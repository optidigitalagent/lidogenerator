# -*- coding: utf-8 -*-
"""Отладка: og:description с украинской локалью."""

import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

from playwright.async_api import async_playwright

SEL = 'meta[property="og:description"]'


async def main():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(locale="uk-UA")
        page = await ctx.new_page()
        await page.goto("https://www.instagram.com/ilmargo_beauty/", wait_until="domcontentloaded")
        await page.wait_for_selector(SEL, state="attached", timeout=15000)
        print(repr(await page.get_attribute(SEL, "content")))
        await browser.close()


asyncio.run(main())
