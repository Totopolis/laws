"""MCP-сервер поиска по кодексам РФ (stdio).

    python -m laws_mcp.server

Настройки — через переменные окружения:
    LAWS_INDEX_DATE     снапшот индекса (по умолчанию последний)
    LAWS_BACKEND        auto | torch | onnx (в контейнере — onnx на CPU)
    LAWS_MODEL_DIR      каталог ONNX-модели для backend=onnx
"""
import json
import os

try:                                    # SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:                     # SDK 1.x, там же класс назывался FastMCP
    from mcp.server.fastmcp import FastMCP as MCPServer

from .search import Searcher

mcp = MCPServer("laws")
_searcher = None


def searcher():
    """Ленивая инициализация: модель грузится при первом запросе, а не при старте."""
    global _searcher
    if _searcher is None:
        _searcher = Searcher(date=os.getenv("LAWS_INDEX_DATE"),
                             # onnx, а не auto: сервер считает запросы на CPU и не занимает
                             # видеокарту — она нужна для построения индекса. Через
                             # LAWS_BACKEND=torch можно переключить, если GPU свободна.
                             backend=os.getenv("LAWS_BACKEND", "onnx"),
                             model_dir=os.getenv("LAWS_MODEL_DIR"))
    return _searcher


@mcp.tool()
def search_law(query: str, top_k: int = 10, codes: list[str] | None = None,
               include_repealed: bool = False, mode: str = "dense",
               rerank: bool = False) -> str:
    """Найти статьи кодексов РФ по смыслу запроса.

    query: вопрос или формулировка на естественном языке («что грозит за неуплату алиментов»).
    codes: ограничить поиск кодексами по slug (['uk-rf', 'sk-rf']); список даёт list_codes.
    include_repealed: включать статьи, утратившие силу (по умолчанию нет).
    mode: dense — поиск по смыслу, он же лучший по замерам; fts — по словам,
          для точных цитат; hybrid — слияние обоих.
    rerank: пересортировать выдачу cross-encoder'ом. Включай, когда запрос описан
          бытовыми словами, а норма названа абстрактно («сосед залил квартиру» ->
          «общие основания ответственности за причинение вреда»), или когда обычный
          поиск вернул не то. Стоит около 4 секунд, поэтому не включён по умолчанию.
    Возвращает найденные статьи с фрагментами; полный текст — через get_article.
    """
    hits = searcher().search(query, top_k=top_k, codes=codes,
                             include_repealed=include_repealed, mode=mode, rerank=rerank)
    out = [{"code": h["code"], "code_slug": h["code_slug"], "number": h["article_number"],
            "article": h["article_heading"],
            "unit": h["unit_path"], "status": h["status"], "path": h["path"],
            "fragment": h["fragment"], "score": round(h["score"], 5),
            "article_file": h["article_file"]} for h in hits]
    return json.dumps(out, ensure_ascii=False, indent=1)


@mcp.tool()
def get_article(code_slug: str, number: str) -> str:
    """Полный текст статьи по кодексу и номеру («uk-rf», «105»).

    Возвращает структуру статьи целиком: заголовок, иерархию, части и пункты по порядку.
    """
    found = searcher().find_article(code_slug, number)
    if not found:
        return json.dumps({"error": f"статья {number} не найдена в {code_slug}"},
                          ensure_ascii=False)
    out = [f["article"] for f in found if f["article"]]
    return json.dumps(out[0] if len(out) == 1 else out, ensure_ascii=False, indent=1)


@mcp.tool()
def list_codes() -> str:
    """Список кодексов в индексе: slug, полное название, число статей, дата редакции."""
    s = searcher()
    codes = [{k: c[k] for k in ("slug", "code", "articles", "redaction_date")}
             for c in s.codes()]
    return json.dumps({"snapshot": s.date, "codes": codes}, ensure_ascii=False, indent=1)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
