# MCP-сервер поиска по кодексам РФ. Всё нужное внутри образа: тексты, индекс и модели,
# поэтому контейнеру не нужна сеть.
#
#   docker build -t laws-mcp .
#   docker run -i --rm laws-mcp        # флаг -i обязателен: обмен идёт по stdio
#
# Перед сборкой выполните подготовку (/x-prepare-system или setup_check --install):
# модели в репозиторий не входят и копируются в образ из models/.
#
# Размер образа около 3,5 ГБ, из них 2,8 ГБ — модели. Эмбеддер меньше не сделать:
# его int8-сборка даёт векторы, не совпадающие с индексом, и поиск молча промахивается.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    LAWS_ROOT=/app \
    LAWS_BACKEND=onnx \
    OMP_NUM_THREADS=4

WORKDIR /app

# рантайм без torch: запросы считаются через onnxruntime на CPU
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/laws_mcp /app/src/laws_mcp
COPY models /app/models
# тексты статей в индексе не дублируются — search.article() читает их из data/
COPY data /app/data
COPY index /app/index

CMD ["python", "-m", "laws_mcp.server"]
