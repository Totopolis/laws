"""Cross-encoder bge-reranker-v2-m3: пересортировка кандидатов после векторного поиска.

Векторный поиск сравнивает запрос и статью по отдельности — каждый превращён в вектор
заранее, и «сосед залил квартиру» не встречается с «общими основаниями ответственности
за причинение вреда». Cross-encoder читает пару целиком и потому ловит связь, которой
нет в лексическом пересечении.

В отличие от эмбеддера, реранкер **ни с чем не обязан совпадать**: он выдаёт скаляр,
а не вектор в общем с индексом пространстве. Поэтому квантование здесь допустимо —
проверять надо не косинус к эталону, а метрику поиска (evaluate.py dense dense+rerank).

Оговорка про int8: torch в fp32 даёт ровно одни и те же оценки независимо от того, как
пары разложены по батчам (разница 0,0), а int8-сборка при разном паддинге расходится
до 0,7 и переставляет близких кандидатов между собой. На агрегированной метрике
(40 запросов) это не сказывается — расходятся только соседи с почти равными оценками,
— но одиночный пример на int8 ничего не доказывает: сравнивать варианты надо метрикой.

Скорость упирается в размер модели: 568M параметров на 5 тыс. токенов — это ~5,7 TFLOP,
то есть около 3,8 с на 30 пар при int8 на 20 ядрах (примерно половина пика AVX-VNNI).
Проверено и отвергнуто: bge-reranker-base (278M) вдвое быстрее, но роняет recall
на простых запросах со 100% до 95% и почти не помогает на трудных; урезание до 20
кандидатов даёт 2,7 с ценой падения recall на трудных с 75% до 62%; ORT_ENABLE_ALL
и parallel execution не дают ничего.
"""
import numpy as np

MODEL_ID = "BAAI/bge-reranker-v2-m3"
ONNX_REPO = "onnx-community/bge-reranker-v2-m3-ONNX"
MAX_TOKENS = 512          # cross-encoder обучен на парах такой длины
CANDIDATES = 30           # сколько кандидатов от векторного поиска пересортировывать


class TorchReranker:
    """sentence-transformers на GPU — для замеров качества и офлайн-экспериментов."""

    def __init__(self, model_id=MODEL_ID, device=None, max_tokens=MAX_TOKENS):
        import torch
        from sentence_transformers import CrossEncoder
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m = CrossEncoder(model_id, device=device, max_length=max_tokens)
        self.device = device

    def score(self, query, docs, batch_size=8):
        if not docs:
            return np.zeros(0, dtype=np.float32)
        pairs = [[query, d] for d in docs]
        return np.asarray(self.m.predict(pairs, batch_size=batch_size), dtype=np.float32)


class OnnxReranker:
    """onnxruntime на CPU — то, что работает в контейнере и в CLI."""

    def __init__(self, model_dir, max_tokens=MAX_TOKENS, threads=None):
        import json
        import pathlib

        import onnxruntime as ort
        from tokenizers import Tokenizer
        d = pathlib.Path(model_dir)
        onnx = next(iter(sorted(d.rglob("model*.onnx"))), None)
        if onnx is None:
            raise SystemExit(f"нет файла реранкера в {d}")
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.pad = self.tok.token_to_id(cfg.get("pad_token") or "<pad>") or 1
        self.tok.enable_truncation(max_tokens)
        self.max_tokens = max_tokens

    def score(self, query, docs, batch_size=8):
        """Пары считаются в порядке возрастания длины, а результат возвращается в исходном.

        Батч дополняется до самой длинной пары в нём, поэтому вперемешку 35% вычислений
        уходит на паддинг. Группировка похожих по длине убирает большую часть этих потерь
        и даёт около четверти времени даром: 5,2 с -> 3,8 с на 30 парах.
        """
        if not docs:
            return np.zeros(0, dtype=np.float32)
        enc = self.tok.encode_batch([(query, d) for d in docs])
        order = sorted(range(len(enc)), key=lambda i: len(enc[i].ids))
        scores = np.zeros(len(docs), dtype=np.float32)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            scores[idx] = self._batch([enc[j] for j in idx])
        return scores

    def _batch(self, enc):
        n = max(len(e.ids) for e in enc)
        ids = np.array([e.ids + [self.pad] * (n - len(e.ids)) for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask + [0] * (n - len(e.ids)) for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = self.sess.run(None, feed)[0]
        return logits.reshape(len(enc), -1)[:, 0].astype(np.float32)


def load(backend="onnx", model_dir=None, **kw):
    from .paths import RERANKER_DIR
    if backend == "torch":
        return TorchReranker(**kw)
    return OnnxReranker(model_dir or RERANKER_DIR, **kw)


def doc_text(hit, limit=1200):
    """Что показывать реранкеру: заголовок статьи плюс найденный фрагмент.

    Заголовок нужен — во фрагменте часто одна оговорка без указания, о чём статья;
    длина режется, потому что модель всё равно видит только 512 токенов.
    """
    head = hit["article_heading"]
    body = " ".join(hit.get("fragment", "").split())
    return f"{head}. {body}"[:limit]
