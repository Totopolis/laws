"""Поиск по кодексам из командной строки — то же, что даёт MCP-сервер, но без него.

    python -m laws_mcp.cli "что грозит за неуплату алиментов"
    python -m laws_mcp.cli "вопрос один" "вопрос два" "вопрос три"   # см. ниже
    python -m laws_mcp.cli --code=uk-rf --top=5 "необходимая оборона"
    python -m laws_mcp.cli --article uk-rf:105 gk-rf-chast-1:222
    python -m laws_mcp.cli --json "самовольная постройка"

Запросов можно передать сколько угодно за один запуск, и так и надо делать: модель
весит 2,3 ГБ и грузится ~4 секунды, а сам поиск занимает 0,2 с. Пять отдельных вызовов
CLI — это пять загрузок модели, один вызов с пятью запросами — одна.
"""
import argparse
import json
import sys

from .search import DEFAULT_MODE, Searcher

FRAGMENT = 300


def flat(content, out=None, depth=0):
    """Текст статьи по порядку документа, с отступами по вложенности."""
    out = [] if out is None else out
    for it in content:
        if "table" in it:
            for row in it["table"]:
                out.append("  " * depth + " | ".join(str(c) for c in row if c))
        elif "content" in it:
            flat(it["content"], out, depth + 1)
        elif "text" in it:
            out.append("  " * depth + it["text"])
    return out


def show_hits(query, hits, fragment=FRAGMENT):
    print(f"\n=== {query} ===")
    if not hits:
        print("  ничего не найдено")
        return
    for i, h in enumerate(hits, 1):
        num = h["article_number"]
        print(f"{i}. {h['code_slug']} ст. {num} — {h['article_title'] or h['article_heading']}"
              f"  [{h['score']:.4f}]")
        where = " > ".join(h["path"][-2:]) if h["path"] else ""
        meta = " | ".join(x for x in (where, h["unit_path"], h["status"]) if x)
        if meta:
            print(f"   {meta}")
        if fragment:
            frag = " ".join(h["fragment"].split())[:fragment]
            print(f"   {frag}{'…' if len(h['fragment']) > fragment else ''}")


def show_article(s, ref):
    """ref — «uk-rf:105»."""
    if ":" not in ref:
        print(f"неверная ссылка «{ref}», нужно вида uk-rf:105")
        return
    slug, num = ref.split(":", 1)
    found = s.find_article(slug, num)
    if not found:
        print(f"\n=== {ref} — не найдено ===")
        return
    for f in found:
        art = f["article"]
        if not art:
            continue
        print(f"\n=== {art['code']} ===")
        print(art["heading"], f"({art['status']})")
        if art.get("path"):
            print(" > ".join(art["path"]))
        print()
        print("\n".join(flat(art.get("content") or [])))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="laws", description="поиск по кодексам РФ")
    ap.add_argument("queries", nargs="*", help="запросы (несколько — за один запуск)")
    ap.add_argument("--article", nargs="+", metavar="SLUG:НОМЕР",
                    help="показать статьи целиком, например uk-rf:105")
    ap.add_argument("--code", action="append", dest="codes", metavar="SLUG",
                    help="искать только в этих кодексах (можно повторять)")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=("dense", "fts", "hybrid"))
    ap.add_argument("--repealed", action="store_true", help="включая утратившие силу")
    ap.add_argument("--rerank", action="store_true",
                    help="пересортировать выдачу cross-encoder'ом: точнее на расплывчатых "
                         "запросах, но +4 с на запрос")
    ap.add_argument("--date", help="снапшот индекса (по умолчанию последний)")
    # onnx, а не auto: для разового запуска инициализация CUDA дороже всего остального —
    # 19,5 с на torch против 6,1 с на onnx при одном и том же запросе
    ap.add_argument("--backend", default="onnx", help="onnx (по умолчанию) | torch | auto")
    ap.add_argument("--json", action="store_true", help="вывод в JSON")
    ap.add_argument("--fragment", type=int, default=FRAGMENT, help="0 — без фрагментов")
    ap.add_argument("--codes-list", action="store_true", help="показать список кодексов")
    a = ap.parse_args()

    if not (a.queries or a.article or a.codes_list):
        ap.error("нужен хотя бы один запрос, --article или --codes-list")

    s = Searcher(date=a.date, backend=a.backend)

    if a.codes_list:
        if a.json:
            print(json.dumps(s.codes(), ensure_ascii=False, indent=1))
        else:
            for c in s.codes():
                print(f"{c['slug']:<18} {c['articles']:>4} ст.  ред. {c['redaction_date']}  {c['code']}")
        return

    if a.article:
        for ref in a.article:
            show_article(s, ref)
        return

    result = {}
    for q in a.queries:
        hits = s.search(q, top_k=a.top, codes=a.codes, include_repealed=a.repealed,
                        mode=a.mode, rerank=a.rerank)
        if a.json:
            result[q] = [{"code_slug": h["code_slug"], "number": h["article_number"],
                          "heading": h["article_heading"], "unit": h["unit_path"],
                          "status": h["status"], "path": h["path"],
                          "score": round(h["score"], 5), "file": h["article_file"]}
                         | ({"fragment": " ".join(h["fragment"].split())[:a.fragment]}
                            if a.fragment else {})
                         for h in hits]
        else:
            show_hits(q, hits, a.fragment)
    if a.json:
        print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
