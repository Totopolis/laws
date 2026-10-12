"""Поиск по индексу: векторный (Qwen3), полнотекстовый (BM25) и их слияние по RRF.

По умолчанию работает чистый dense. Ожидание было обратным — что для кодексов нужен
гибрид, — но замер на 32 контрольных запросах (evaluate.py) его не подтвердил:

    набор       dense           fts             hybrid
    смысловые   100% / 0.839    75% / 0.337     90% / 0.607
    термины     100% / 0.875    92% / 0.615    100% / 0.836   (recall@10 / MRR)

Qwen3-Embedding уверенно берёт и точные термины («преюдиция», «эмансипация»), поэтому
BM25 в слиянии только подмешивает шум. Перебор веса FTS (0.15 / 0.3 / 0.5) картину
не меняет: лучшее, что он даёт, — +0,003 MRR на терминах при −0,17 на смысловых.
Режимы fts и hybrid оставлены для запросов на точную цитату и для перепроверки.

Результаты схлопываются по статье: article- и chunk-записи одной статьи — один ответ.
"""
import numpy as np

from .paths import INDEX, PROCESSED

TABLE = "laws"
RRF_K = 60          # сглаживание рангов; классическое значение из статьи о RRF
POOL = 5            # во сколько раз глубже брать кандидатов, чем нужно ответов
FTS_WEIGHT = 0.3    # вклад BM25, если явно выбран mode="hybrid" (см. таблицу выше)
DEFAULT_MODE = "dense"
RERANK_DEPTH = 30   # кандидатов на пересортировку; глубже — дороже, а recall тот же
RERANK_WEIGHT = 1.0 # вес ранга cross-encoder'а против исходного при слиянии


def is_flat():
    """Клиентская раскладка: индекс лежит прямо в index/, без папки-даты."""
    return (INDEX / f"{TABLE}.lance").exists()


def snapshots():
    """Даты, для которых построен индекс. У плоской раскладки дат нет — один снапшот."""
    if is_flat():
        return ["-"]
    if not INDEX.exists():
        return []
    return sorted(p.name for p in INDEX.iterdir() if (p / f"{TABLE}.lance").exists())


class Searcher:
    def __init__(self, date=None, backend="auto", model_dir=None):
        import lancedb
        if is_flat():
            # в клиентской сборке снапшот один: index/ и data/ без папок-дат
            self.date, db_dir, self.texts = None, INDEX, PROCESSED
        else:
            dates = snapshots()
            if not dates:
                raise SystemExit(f"нет индекса в {INDEX} — сначала python -m laws_mcp.index")
            self.date = date or dates[-1]
            db_dir, self.texts = INDEX / self.date, PROCESSED / self.date
        self.db = lancedb.connect(db_dir)
        self.t = self.db.open_table(TABLE)
        self._emb = self._rr = self._dirs = None
        self._backend, self._model_dir = backend, model_dir

    @property
    def emb(self):
        if self._emb is None:
            from .embedder import load
            self._emb = load(self._backend, self._model_dir)
        return self._emb

    def _where(self, codes=None, include_repealed=False):
        cond = []
        if codes:
            lst = ", ".join(f"'{c}'" for c in codes)
            cond.append(f"code_slug IN ({lst})")
        if not include_repealed:
            cond.append("status = 'действует'")
        return " AND ".join(cond) or None

    @property
    def reranker(self):
        if self._rr is None:
            from .reranker import load
            self._rr = load("onnx")
        return self._rr

    def _rerank(self, query, hits):
        """Пересортировать кандидатов cross-encoder'ом, слив его ранг с исходным по RRF.

        Именно слить, а не заменить: чистый реранкер роняет recall на простых запросах
        (100% -> 95%), потому что уверенно поднимает формально похожие, но не те статьи.
        RRF с равным весом сохраняет recall и при этом чинит трудные случаи.
        """
        import numpy as np

        from .reranker import doc_text
        scores = self.reranker.score(query, [doc_text(h) for h in hits])
        rr_rank = {j: i for i, j in enumerate(np.argsort(-scores), 1)}
        for i, h in enumerate(hits):
            h["rerank_score"] = float(scores[i])
            h["score"] = 1.0 / (RRF_K + i + 1) + RERANK_WEIGHT / (RRF_K + rr_rank[i])
        return sorted(hits, key=lambda h: -h["score"])

    def search(self, query, top_k=10, codes=None, include_repealed=False, mode=DEFAULT_MODE,
               fts_weight=FTS_WEIGHT, rerank=False, rerank_depth=RERANK_DEPTH):
        where = self._where(codes, include_repealed)
        depth = max(top_k, rerank_depth if rerank else 0) * POOL
        ranked = {}

        if mode in ("hybrid", "dense"):
            qv = self.emb.encode([query], is_query=True)[0]
            q = self.t.search(qv.astype(np.float32), vector_column_name="vector").limit(depth)
            if where:
                q = q.where(where, prefilter=True)
            self._merge(ranked, q.to_list(), "dense")

        if mode in ("hybrid", "fts"):
            try:
                q = self.t.search(query, query_type="fts").limit(depth)
                if where:
                    q = q.where(where, prefilter=True)
                self._merge(ranked, q.to_list(), "fts")
            except Exception as e:                       # индекс FTS ещё не построен
                if mode == "fts":
                    raise
                self.fts_error = str(e)

        w = {"dense": 1.0, "fts": fts_weight}
        for r in ranked.values():
            r["score"] = sum(w[tag] / (RRF_K + rank) for tag, rank in r["sources"].items())
        out = sorted(ranked.values(), key=lambda r: -r["score"])
        if rerank:
            out = self._rerank(query, out[:rerank_depth])
        return out[:top_k]

    @staticmethod
    def _merge(ranked, rows, tag):
        """Схлопнуть записи одной статьи, запомнив её лучший ранг в этом источнике.

        Ранги именно минимизируются, а не суммируются: иначе длинная статья, у которой
        в выдачу попало пять чанков, обгоняла бы точное попадание за счёт количества.
        Итоговый RRF считается в search() уже по лучшим рангам.
        """
        for rank, row in enumerate(rows, 1):
            key = f"{row['code_slug']}/{row['article_file']}"
            cur = ranked.get(key)
            if cur is None:
                cur = ranked[key] = {k: row[k] for k in
                                     ("code", "code_slug", "article_number", "article_title",
                                      "article_heading", "article_file", "path", "status")}
                cur.update(score=0.0, sources={}, fragment="", unit_path="", level="", best=None)
            if cur["sources"].get(tag, rank + 1) > rank:
                cur["sources"][tag] = rank
            # фрагмент берём от записи с лучшим рангом; чанк предпочтительнее статьи
            weight = (rank, 0 if row["level"] == "chunk" else 1)
            if cur["best"] is None or weight < cur["best"]:
                cur.update(best=weight, fragment=row["text"], unit_path=row["unit_path"],
                           level=row["level"])

    def _dir(self, code_slug):
        """Папка документа в снапшоте.

        Тексты разложены по разделам (`codex/uk-rf/`, `fz/fz-127-2002/`), и раздел
        известен только реестру; в индексе лежит один code_slug. Снапшоты до разделения
        держали папки в корне — для них dir совпадает со slug'ом.
        """
        if self._dirs is None:
            self._dirs = {c["slug"]: c.get("dir", c["slug"]).strip("/") for c in self.codes()}
        return self._dirs.get(code_slug, code_slug)

    def article(self, code_slug, article_file):
        """Полный JSON статьи из снапшота — в индексе он не дублируется."""
        import json
        p = self.texts / self._dir(code_slug) / article_file
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def find_article(self, code_slug, number):
        """Статья по номеру. Имена файлов не угадываем: номер лежит в индексе.

        Номер может повторяться (в БК две статьи 242.1) — возвращаем все совпадения,
        действующая первой.
        """
        num = str(number).replace("¹", ".1").replace(",", ".").strip()
        rows = (self.t.search()
                .where(f"code_slug = '{code_slug}' AND article_number = '{num}' "
                       f"AND level = 'article'", prefilter=True)
                .select(["article_file", "status", "article_heading"]).limit(10).to_list())
        rows.sort(key=lambda r: r["status"] != "действует")
        return [dict(r, article=self.article(code_slug, r["article_file"])) for r in rows]

    def codes(self):
        import json
        reg = json.loads((self.texts / "index.json").read_text(encoding="utf-8"))
        return reg["codes"]
