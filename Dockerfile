FROM python:3.12-slim

WORKDIR /app

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chromium + системные библиотеки для Playwright
RUN playwright install --with-deps chromium

COPY . .

CMD ["python", "bot.py"]
