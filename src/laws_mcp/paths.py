"""Каталоги данных. В репозитории — относительно корня, в контейнере задаются через env.

    LAWS_ROOT       корень данных (по умолчанию корень репозитория)
    LAWS_INDEX      каталог с индексом LanceDB          (<root>/index)
    LAWS_PROCESSED  каталог с текстами кодексов         (<root>/processed или <root>/data)
    LAWS_MODEL_DIR  каталог ONNX-модели эмбеддера       (<root>/models/qwen3-emb-onnx)
    LAWS_RERANKER_DIR каталог ONNX-реранкера            (<root>/models/bge-reranker-onnx)

Раскладок две. В репозитории разработки тексты и индексы разложены по датам выгрузки
(`processed/2026-08-24/`, `index/2026-08-24/`), потому что снапшотов бывает несколько
и их сравнивают. В клиентской сборке снапшот ровно один, и папки-даты там только мешают:
данные лежат плоско — `data/` и `index/`. Код понимает обе, см. Searcher в search.py.
"""
import os
import pathlib

ROOT = pathlib.Path(os.getenv("LAWS_ROOT")
                    or pathlib.Path(__file__).resolve().parents[2])
INDEX = pathlib.Path(os.getenv("LAWS_INDEX") or ROOT / "index")
MODEL_DIR = pathlib.Path(os.getenv("LAWS_MODEL_DIR") or ROOT / "models" / "qwen3-emb-onnx")
RERANKER_DIR = pathlib.Path(os.getenv("LAWS_RERANKER_DIR")
                            or ROOT / "models" / "bge-reranker-onnx")


def _texts_dir():
    """processed/ в репозитории разработки, data/ — в клиентской сборке."""
    env = os.getenv("LAWS_PROCESSED")
    if env:
        return pathlib.Path(env)
    for name in ("processed", "data"):
        if (ROOT / name).exists():
            return ROOT / name
    return ROOT / "processed"


PROCESSED = _texts_dir()
