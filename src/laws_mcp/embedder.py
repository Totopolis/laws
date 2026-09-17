"""Qwen3-Embedding-0.6B: два бэкенда — torch/GPU для индексации, ONNX/CPU для запросов.

У модели last-token pooling (не mean) и левый паддинг, поэтому пулинг руками писать нельзя —
для torch берётся sentence-transformers с родным конфигом, для ONNX пулинг повторён явно.
Запрос подаётся с инструкцией, документ — без неё: это требование модели, иначе
качество молча падает.
"""
import numpy as np

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
DIM = 1024
MAX_TOKENS = 2560                 # хватает на самый длинный чанк (~7 тыс. символов)
TASK = "Найди статью кодекса Российской Федерации, отвечающую на вопрос"


def query_prompt(text, task=TASK):
    """Обёртка запроса; документы эмбеддятся без неё."""
    return f"Instruct: {task}\nQuery: {text}"


def gpu_dtype(torch):
    """fp16 — только там, где он действительно быстрый.

    Начиная с Volta (sm_70) половинная точность считается на тензорных ядрах и даёт
    кратное ускорение. На Pascal (GTX 10xx, sm_6x) тензорных ядер нет, а fp16-арифметика
    урезана в 64 раза относительно fp32 — там считать нужно в fp32, иначе индексация
    растянется на сутки.
    """
    major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
    return "float16" if major >= 7 else "float32"


class TorchEmbedder:
    """sentence-transformers на GPU. Используется при построении индекса."""

    def __init__(self, model_id=MODEL_ID, device=None, dtype=None, max_tokens=MAX_TOKENS):
        import torch
        from sentence_transformers import SentenceTransformer
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = dtype or gpu_dtype(torch)
        self.dtype = dtype
        kw = {"torch_dtype": getattr(torch, dtype)} if device == "cuda" else {}
        self.m = SentenceTransformer(model_id, device=device, model_kwargs=kw,
                                     processor_kwargs={"padding_side": "left"})
        self.m.max_seq_length = max_tokens
        self.device = device

    def encode(self, texts, batch_size=8, is_query=False, show_progress=False):
        if is_query:
            texts = [query_prompt(t) for t in texts]
        v = self.m.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                          show_progress_bar=show_progress, convert_to_numpy=True)
        return v.astype(np.float32)


class OnnxEmbedder:
    """onnxruntime на CPU: то же самое для одиночных запросов в MCP-сервере.

    Модель нужна именно во float32 (2,3 ГБ). Проверено на 40 текстах против torch:
    fp32 даёт косинус 1,000 (минимум 0,998), а int8-сборка — 0,73, то есть её векторы
    лежат в другой области пространства, чем построенный на GPU индекс, и поиск
    молча промахивается. Вариант fp16 на CPU считается через Cast-ноды и не уложился
    в 10 минут на тех же 40 текстах.
    """

    def __init__(self, model_dir, max_tokens=MAX_TOKENS, threads=None):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        import pathlib, json
        d = pathlib.Path(model_dir)
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        onnx = next(iter(sorted(d.rglob("model*.onnx"))), None)
        if onnx is None:
            raise SystemExit(f"нет файла модели в {d}")
        self.sess = ort.InferenceSession(str(onnx), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        # модель экспортирована как декодер с KV-кэшем; для одного прогона подаём пустой
        self.kv = sorted(n for n in self.inputs if n.startswith("past_key_values."))
        self.kv_heads, self.kv_dim = 8, 128
        # у fp16-сборки кэш тоже fp16 — float32 она не принимает
        kv_type = next((i.type for i in self.sess.get_inputs() if i.name in self.kv), "tensor(float)")
        self.kv_dtype = np.float16 if "float16" in kv_type else np.float32
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        cfg = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
        # пулинг берёт последний токен, поэтому важно, какой именно он. Токенизатор Qwen3
        # сам дописывает <|endoftext|> — тот же токен использует sentence-transformers,
        # а eos_token из конфига (<|im_end|>) здесь ни при чём и вектор бы испортил.
        self.pad = self.tok.token_to_id(cfg.get("pad_token") or "<|endoftext|>")
        self.eos = self.pad
        self.max_tokens = max_tokens

    def encode(self, texts, batch_size=1, is_query=False, show_progress=False):
        """Тексты кодируются по одному: batch_size принимается ради общего интерфейса.

        Этот ONNX-экспорт сделан как декодер с KV-кэшем и при паддинге даёт другой вектор
        для того же текста (косинус ~0,89 между batch=1 и batch=3) — маска учитывается
        не полностью. Без паддинга результат детерминирован, а MCP-сервер и так шлёт
        по одному запросу.
        """
        if is_query:
            texts = [query_prompt(t) for t in texts]
        return np.concatenate([self._one(t) for t in texts], 0)

    def _one(self, text):
        ids = self.tok.encode(text).ids[:self.max_tokens]
        if not ids or ids[-1] != self.eos:
            ids = ids + [self.eos]
        inp = np.array([ids], dtype=np.int64)
        mask = np.ones_like(inp)
        feed = {"input_ids": inp, "attention_mask": mask}
        if "position_ids" in self.inputs:
            feed["position_ids"] = np.arange(len(ids), dtype=np.int64)[None, :]
        empty = np.zeros((1, self.kv_heads, 0, self.kv_dim), dtype=self.kv_dtype)
        feed |= {n: empty for n in self.kv}
        h = self.sess.run(["last_hidden_state"], feed)[0]     # (1, seq, hidden)
        v = h[:, -1, :].astype(np.float32)                    # last-token pooling
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def load(backend="auto", model_dir=None, **kw):
    """auto: torch, если есть CUDA, иначе ONNX (в контейнере доступен только он)."""
    from .paths import MODEL_DIR
    model_dir = model_dir or MODEL_DIR
    if backend == "torch":
        return TorchEmbedder(**kw)
    if backend == "onnx":
        return OnnxEmbedder(model_dir, **kw)
    try:
        import torch
        if torch.cuda.is_available():
            return TorchEmbedder(**kw)
    except ImportError:
        pass
    return OnnxEmbedder(model_dir, **kw)
