from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False


DEFAULT_RAG_MODEL = "google/flan-t5-large"
_MAX_INPUT_TOKENS = 512
_MAX_OUTPUT_TOKENS = 256


@dataclass
class RagResult:
    answer: str
    citations: List[str]
    sources: List[str]


_model = None
_tokenizer = None


def load_rag_model(model_name: str = DEFAULT_RAG_MODEL, device: str = "cpu"):
    global _model, _tokenizer
    if not RAG_AVAILABLE:
        raise RuntimeError("transformers not installed")

    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        _model.eval()
        if device == "cuda" and torch.cuda.is_available():
            _model.to("cuda")
        else:
            _model.to("cpu")
    return _model, _tokenizer


def unload_rag_model():
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_prompt(query: str, paragraphs: List[str], sources: List[str]) -> str:
    context_parts = []
    for i, (para, src) in enumerate(zip(paragraphs, sources), 1):
        context_parts.append(f"[{i}] (from {src})\n{para[:600]}")

    context = "\n\n".join(context_parts)

    prompt = f"""Answer the question based only on the provided context. If the context doesn't contain enough information, say "I cannot answer this from the provided documents."

Context:
{context}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query: str, results: list, top_n: int = 5) -> Optional[RagResult]:
    try:
        model, tokenizer = load_rag_model()
    except (RuntimeError, Exception) as e:
        return None

    paragraphs = []
    sources = []
    for r in results[:top_n]:
        paragraphs.append(r.content if hasattr(r, 'content') else str(r))
        source_info = r.source if hasattr(r, 'source') else "unknown"
        if hasattr(r, 'page_number') and r.page_number:
            source_info += f" (page {r.page_number})"
        sources.append(source_info)

    prompt = build_prompt(query, paragraphs, sources)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_MAX_INPUT_TOKENS,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=_MAX_OUTPUT_TOKENS,
            do_sample=False,
            temperature=0.3,
            num_beams=2,
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    citation_numbers = []
    seen = set()
    for i, (para, src) in enumerate(zip(paragraphs, sources), 1):
        pattern = re.compile(r'\[' + str(i) + r'\]', re.IGNORECASE)
        if pattern.search(answer):
            if i not in seen:
                citation_numbers.append(str(i))
                seen.add(i)

    answer_clean = re.sub(r'\[\d+\]', '', answer).strip()

    return RagResult(
        answer=answer_clean,
        citations=citation_numbers,
        sources=sources,
    )
