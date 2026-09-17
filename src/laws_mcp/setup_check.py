"""Подготовка и проверка системы поиска: зависимости, модель, индекс, CLI и MCP.

    python -m laws_mcp.setup_check              # только проверить, ничего не менять
    python -m laws_mcp.setup_check --install    # доустановить недостающее и скачать модель
    python -m laws_mcp.setup_check --install --gpu   # ещё и torch для построения индекса

Проверки идут по порядку зависимости: сначала окружение, потом данные, потом то, что
на них опирается. Первая же непройденная проверка объясняет, что делать, — дальше
идти смысла нет, поэтому такие проверки помечаются как блокирующие.
"""
import argparse
import importlib
import json
import pathlib
import shutil
import subprocess
import sys
import time

MODEL_REPO = "onnx-community/Qwen3-Embedding-0.6B-ONNX"
MODEL_FILES = ["onnx/model.onnx", "onnx/model.onnx_data",
               "tokenizer.json", "tokenizer_config.json", "config.json"]
RERANKER_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
RERANKER_FILES = ["onnx/model_int8.onnx", "tokenizer.json",
                  "tokenizer_config.json", "config.json"]
RUNTIME = ["mcp", "lancedb", "pyarrow", "numpy", "onnxruntime", "tokenizers"]
GPU_EXTRA = ["torch", "sentence_transformers", "huggingface_hub"]
PIP_NAME = {"sentence_transformers": "sentence-transformers", "huggingface_hub": "huggingface_hub"}

OK, FAIL, WARN = "  [ok]  ", "  [нет] ", "  [!]   "


class Report:
    def __init__(self):
        self.failed = []

    def ok(self, what, detail=""):
        print(f"{OK}{what}" + (f" — {detail}" if detail else ""), flush=True)

    def warn(self, what, detail=""):
        print(f"{WARN}{what}" + (f" — {detail}" if detail else ""), flush=True)

    def fail(self, what, how):
        print(f"{FAIL}{what}\n         {how}", flush=True)
        self.failed.append(what)


def has_torch():
    try:
        importlib.import_module("torch")
        return True
    except ImportError:
        return False


def pip_install(packages, extra_args=()):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *extra_args, *packages]
    print(f"  ставлю: {' '.join(packages)}", flush=True)
    return subprocess.run(cmd).returncode == 0


def check_packages(r, names, install, label, extra_args=()):
    missing = []
    for n in names:
        try:
            importlib.import_module(n)
        except ImportError:
            missing.append(PIP_NAME.get(n, n))
    if not missing:
        r.ok(f"{label}: все на месте", ", ".join(names))
        return True
    if install:
        if pip_install(missing, extra_args):
            r.ok(f"{label}: доустановлены", ", ".join(missing))
            return True
        r.fail(f"{label}: установка не удалась", f"поставьте вручную: pip install {' '.join(missing)}")
        return False
    r.fail(f"{label}: не хватает {', '.join(missing)}",
           "запустите с --install либо поставьте сами")
    return False


def check_model(r, install):
    from .paths import MODEL_DIR
    onnx = MODEL_DIR / "onnx" / "model.onnx"
    data = MODEL_DIR / "onnx" / "model.onnx_data"
    if onnx.exists() and data.exists():
        size = (onnx.stat().st_size + data.stat().st_size) / 2**30
        if size < 2.0:
            r.fail(f"модель подозрительно мала ({size:.1f} ГБ)",
                   "ожидается ~2,3 ГБ (fp32). Удалите каталог и запустите с --install")
            return False
        r.ok("ONNX-модель на месте", f"{size:.1f} ГБ, {MODEL_DIR}")
        return True
    if not install:
        r.fail("нет ONNX-модели", f"запустите с --install (скачает ~2,3 ГБ в {MODEL_DIR})")
        return False
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        if not pip_install(["huggingface_hub"]):
            r.fail("нет huggingface_hub", "pip install huggingface_hub")
            return False
        from huggingface_hub import hf_hub_download
    print(f"  качаю модель {MODEL_REPO} (~2,3 ГБ, это надолго)", flush=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for f in MODEL_FILES:
        hf_hub_download(MODEL_REPO, f, local_dir=str(MODEL_DIR))
        print(f"    {f}", flush=True)
    r.ok("модель скачана", str(MODEL_DIR))
    return True


def check_reranker(r, install):
    """Реранкер необязателен: без него поиск работает, просто хуже на расплывчатых запросах."""
    from .paths import RERANKER_DIR
    onnx = next(iter(sorted(RERANKER_DIR.rglob("model*.onnx"))), None) if RERANKER_DIR.exists() else None
    if onnx:
        r.ok("реранкер на месте", f"{onnx.stat().st_size / 2**20:.0f} МБ, {RERANKER_DIR}")
        return True
    if not install:
        r.warn("реранкера нет", "не обязателен; чтобы поставить — запустите с --install")
        return False
    from huggingface_hub import hf_hub_download
    print(f"  качаю реранкер {RERANKER_REPO} (~560 МБ)", flush=True)
    for f in RERANKER_FILES:
        hf_hub_download(RERANKER_REPO, f, local_dir=str(RERANKER_DIR))
        print(f"    {f}", flush=True)
    r.ok("реранкер скачан", str(RERANKER_DIR))
    return True


def full_repo():
    """Полный репозиторий (с выгрузкой и сборкой индекса) или клиентская сборка.

    В клиентской сборке нет ни chunker/index, ни скиллов выгрузки — советовать там
    «запустите /x-parse-codexes» бессмысленно, данные туда кладутся готовыми.
    """
    return (pathlib.Path(__file__).parent / "index.py").exists()


def check_data(r):
    """Тексты кодексов и индекс — без них искать нечего.

    Раскладки две: по датам выгрузки (репозиторий разработки) и плоская — data/ и index/
    без папок-дат (клиентская сборка). Проверяются они по-разному, иначе папки кодексов
    в data/ будут приняты за снапшоты.
    """
    from .paths import INDEX, PROCESSED
    from .search import is_flat, snapshots

    if is_flat():
        reg = PROCESSED / "index.json"
        if not reg.exists():
            r.fail(f"нет текстов кодексов в {PROCESSED}", "сборка неполная")
            return False
        reg = json.loads(reg.read_text(encoding="utf-8"))
        r.ok("тексты кодексов", f"{len(reg['codes'])} кодексов, выгрузка "
                                f"{reg.get('fetched', '?')}, {PROCESSED}")
        r.ok("векторный индекс", str(INDEX))
        return True

    proc = sorted(p.name for p in PROCESSED.iterdir() if p.is_dir()) if PROCESSED.exists() else []
    if not proc:
        r.fail("нет разобранных кодексов в processed/",
               "сначала /x-fetch-codexes, затем /x-parse-codexes" if full_repo() else
               "тексты кодексов должны лежать рядом — сборка неполная")
        return False
    r.ok("разобранные кодексы", f"снапшоты: {', '.join(proc)}")

    idx = snapshots()
    if not idx:
        # в клиентской сборке индекс поставляется готовым, собирать его там нечем
        r.fail("нет векторного индекса",
               "постройте: PYTHONPATH=src python -m laws_mcp.index (~70 мин на GPU)"
               if full_repo() else "индекс должен поставляться вместе со сборкой")
        return False
    r.ok("векторный индекс", f"снапшоты: {', '.join(idx)}")
    if idx[-1] not in proc:
        r.warn("индекс новее разбора", f"индекс {idx[-1]}, а processed для него нет")
    elif proc[-1] != idx[-1]:
        r.warn("индекс отстал от разбора",
               f"есть processed/{proc[-1]}, индекс только по {idx[-1]} — пересоберите index")
    return True


def check_search(r):
    """Один реальный запрос: модель + индекс + фильтры вместе."""
    from .search import Searcher
    t0 = time.time()
    s = Searcher(backend="onnx")
    hits = s.search("что грозит за неуплату алиментов", top_k=3)
    dt = time.time() - t0
    if not hits:
        r.fail("поиск ничего не вернул",
               "индекс пуст или не соответствует processed/")
        return False
    top = hits[0]
    r.ok("поиск работает", f"{dt:.1f} с (с загрузкой модели), топ-1: "
                           f"{top['code_slug']} ст. {top['article_number']}")
    art = s.find_article("uk-rf", "105")
    if art and art[0].get("article"):
        r.ok("чтение статей", f"uk-rf ст. 105 — {art[0]['article']['title']}")
    else:
        r.fail("статья не читается", "processed/ не соответствует индексу")
        return False
    return True


def check_quality(r):
    """Контрольные запросы: ловит рассогласование модели и индекса, которое поиск не видит."""
    from .evaluate import SEMANTIC, evaluate
    from .search import Searcher
    st = evaluate(Searcher(backend="onnx"), SEMANTIC[:8], modes=("dense",))["dense"]
    if st["recall"] < 0.9:
        hint = ("обычно это несовпадение модели и индекса — сверьте "
                "python -m laws_mcp.check_backends" if full_repo() else
                "обычно это не та сборка модели: нужна fp32, а не int8 — перекачайте с --install")
        r.fail(f"качество поиска упало: recall {st['recall']:.0%} на контрольных запросах", hint)
        return False
    r.ok("контрольные запросы", f"recall {st['recall']:.0%}, MRR {st['mrr']:.2f}")
    return True


def check_rerank(r):
    """Пересортировка должна чинить запросы, где норма названа не теми словами."""
    from .search import Searcher
    s = Searcher(backend="onnx")
    # без пересортировки ст. 1064 стоит на 9-м месте, после неё — в первой тройке
    q, want = "сосед залил квартиру кто возмещает ущерб", "1064"
    t0 = time.time()
    hits = s.search(q, top_k=5, rerank=True)
    dt = time.time() - t0
    nums = [h["article_number"] for h in hits]
    if want not in nums[:3]:
        r.fail(f"пересортировка не подняла ст. {want} в топ-3 (получено {nums[:3]})",
               "проверьте models/bge-reranker-onnx — возможно, файл модели не тот")
        return False
    r.ok("реранкер работает", f"{dt:.1f} с с загрузкой модели, «{q}» -> "
                              f"ст. {want} на {nums.index(want) + 1} месте (без него 9-е)")
    return True


def check_mcp(r):
    """Поднять сервер как подпроцесс и вызвать инструмент по протоколу."""
    import asyncio
    import os

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    src = str(pathlib.Path(__file__).resolve().parents[1])

    async def run():
        env = dict(os.environ, LAWS_BACKEND="onnx", PYTHONPATH=src, PYTHONIOENCODING="utf-8")
        params = StdioServerParameters(command=sys.executable,
                                       args=["-m", "laws_mcp.server"], env=env)
        async with stdio_client(params) as (rd, wr):
            async with ClientSession(rd, wr) as s:
                await s.initialize()
                tools = [t.name for t in (await s.list_tools()).tools]
                res = await s.call_tool("search_law", {"query": "самовольная постройка", "top_k": 1})
                return tools, json.loads(res.content[0].text)

    try:
        tools, hits = asyncio.run(run())
    except Exception as e:
        r.fail("MCP-сервер не отвечает", f"{type(e).__name__}: {str(e)[:150]}")
        return False
    if not hits:
        r.fail("MCP-сервер вернул пустой результат", "проверьте индекс")
        return False
    r.ok("MCP-сервер", f"инструменты: {', '.join(tools)}; поиск вернул {hits[0]['code_slug']} "
                       f"ст. {hits[0]['number']}")
    return True


def check_cli(r):
    """CLI как его вызывает скилл — отдельным процессом, чтобы поймать проблемы запуска."""
    src = str(pathlib.Path(__file__).resolve().parents[1])
    import os
    env = dict(os.environ, PYTHONPATH=src, PYTHONIOENCODING="utf-8")
    p = subprocess.run([sys.executable, "-m", "laws_mcp.cli", "--top=1", "--fragment=0",
                        "необходимая оборона"], capture_output=True, text=True,
                       encoding="utf-8", env=env, timeout=300)
    if p.returncode != 0 or "uk-rf" not in p.stdout:
        r.fail("CLI не работает", (p.stderr or p.stdout)[-200:].strip())
        return False
    r.ok("CLI", p.stdout.strip().splitlines()[-1].strip()[:80])
    return True


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--install", action="store_true", help="доустановить недостающее")
    ap.add_argument("--gpu", action="store_true", help="ещё и torch+sentence-transformers")
    a = ap.parse_args()

    r = Report()
    print(f"python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"свободно на диске: {shutil.disk_usage(pathlib.Path.cwd()).free / 2**30:.0f} ГБ\n")

    # блокирующие: без них дальнейшие проверки бессмысленны
    if not check_packages(r, RUNTIME, a.install, "рантайм-зависимости"):
        return finish(r)
    # GPU-часть нужна только для построения индекса, поэтому её отсутствие не блокирует
    if a.gpu:
        check_packages(r, GPU_EXTRA, a.install, "GPU-зависимости (для индексации)",
                       ("--index-url", "https://download.pytorch.org/whl/cu126"))
    elif has_torch():
        r.ok("torch на месте", "индекс можно строить здесь же")
    else:
        r.warn("torch не установлен", "для поиска не нужен; чтобы строить индекс — --install --gpu")
    if not check_model(r, a.install):
        return finish(r)
    has_rr = check_reranker(r, a.install)
    if not check_data(r):
        return finish(r)

    ok = check_search(r) and check_quality(r)
    if ok and has_rr:
        check_rerank(r)
    check_cli(r)
    check_mcp(r)
    return finish(r)


def finish(r):
    print()
    if r.failed:
        print(f"не пройдено: {len(r.failed)} — {'; '.join(r.failed)}")
        return 1
    print("система готова: модель, индекс, CLI и MCP работают")
    return 0


if __name__ == "__main__":
    sys.exit(main())
