from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import List, Tuple
from urllib import error, request

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    """文档切片：包含切片编号和文本内容。"""

    chunk_id: int
    text: str


class DeepSeekClient:
    """DeepSeek Chat Completions 客户端。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout: int = 60,
    ):
        # 不在代码里硬编码密钥，优先从环境变量读取。
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_enabled(self) -> bool:
        """是否已配置可用的 API Key。"""
        return bool(self.api_key)

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """调用 DeepSeek 生成回答。"""
        if not self.api_key:
            raise RuntimeError("未设置 DEEPSEEK_API_KEY，无法调用 DeepSeek。")

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
        }

        req = request.Request(
            url=f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"DeepSeek API 网络错误: {exc.reason}") from exc

        try:
            return body["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"DeepSeek 返回结构异常: {body}") from exc


def load_text(file_path: str) -> str:
    """从磁盘读取原始文档。"""
    return Path(file_path).read_text(encoding="utf-8")


def split_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[Chunk]:
    """按字符窗口切分文本；overlap 用于保留跨片段上下文。"""
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap") 

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        # 使用滑动窗口切分，减少关键信息落在边界被截断的风险。
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(chunk_id=chunk_id, text=chunk_text))
            chunk_id += 1
        start += chunk_size - overlap

    return chunks


class SimpleRAG:
    """
    教学版 RAG：
    1) 召回（TF-IDF + 余弦相似度）
    2) 重排（融合召回分数 + 词项重叠分数）
    3) 生成（DeepSeek 根据上下文作答）
    """

    def __init__(self, chunks: List[Chunk], llm_client: DeepSeekClient | None = None):
        self.chunks = chunks
        self.llm_client = llm_client
        # 建立向量空间（索引阶段）。
        self.vectorizer = TfidfVectorizer()
        self.doc_vectors = self.vectorizer.fit_transform([c.text for c in chunks])

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """第一阶段召回：按余弦相似度选出 top-k 候选片段。"""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def _tokenize_for_rerank(self, text: str) -> set[str]:
        """
        轻量分词（不依赖第三方分词库）：
        - 英文/数字：按单词提取
        - 中文：按单字提取
        """
        en_tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        zh_tokens = re.findall(r"[\u4e00-\u9fff]", text)
        return set(en_tokens + zh_tokens)

    def rerank(
        self, query: str, candidates: List[Tuple[Chunk, float]], top_k: int = 3
    ) -> List[Tuple[Chunk, float, float, float]]:
        """
        第二阶段重排：
        - recall_score：第一阶段召回分
        - overlap_score：query 与 chunk 的词项重叠比例
        - final_score：融合后用于最终排序的分数
        """
        query_tokens = self._tokenize_for_rerank(query)
        if not query_tokens:
            # 极端情况（如 query 全是符号）直接回退为召回排序。
            return [
                (chunk, recall_score, recall_score, 0.0)
                for chunk, recall_score in candidates[:top_k]
            ]

        reranked: List[Tuple[Chunk, float, float, float]] = []
        for chunk, recall_score in candidates:
            chunk_tokens = self._tokenize_for_rerank(chunk.text)
            overlap = len(query_tokens & chunk_tokens) / len(query_tokens)

            # 教学目的：显式展示“召回分 + 词项匹配分”的融合方式。
            final_score = 0.7 * recall_score + 0.3 * overlap
            reranked.append((chunk, final_score, recall_score, overlap))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]

    def _build_context(self, reranked: List[Tuple[Chunk, float, float, float]]) -> str:
        """把重排后的结果格式化成上下文文本。"""
        context_parts = []
        for chunk, final_score, recall_score, overlap_score in reranked:
            context_parts.append(
                f"[Chunk {chunk.chunk_id} | rerank={final_score:.4f} | "
                f"recall={recall_score:.4f} | overlap={overlap_score:.4f}]\n"
                f"{chunk.text}"
            )
        return "\n\n".join(context_parts)

    def _generate_with_llm(
        self, query: str, reranked: List[Tuple[Chunk, float, float, float]]
    ) -> str:
        """第三阶段生成：把检索上下文喂给 DeepSeek，得到最终回答。"""
        if self.llm_client is None or not self.llm_client.is_enabled():
            return "未检测到 DEEPSEEK_API_KEY，当前仅展示检索与重排结果。"

        context_only = "\n\n".join(chunk.text for chunk, _, _, _ in reranked)
        system_prompt = (
            "你是一个严谨的问答助手。"
            "请仅依据给定上下文回答问题；"
            "如果上下文不足，请明确回答“根据当前上下文无法确定”。"
        )
        user_prompt = (
            f"问题：{query}\n\n"
            "上下文：\n"
            f"{context_only}\n\n"
            "请用简洁中文给出答案。"
        )

        try:
            return self.llm_client.chat(system_prompt=system_prompt, user_prompt=user_prompt)
        except RuntimeError as exc:
            return f"DeepSeek 调用失败：{exc}"

    def answer(self, query: str, top_k: int = 3, candidate_k: int = 10) -> str:
        # 标准流程：先召回候选，再 rerank，最后生成答案。
        candidate_k = max(candidate_k, top_k)
        candidates = self.retrieve(query, top_k=candidate_k)
        reranked = self.rerank(query, candidates, top_k=top_k)

        context = self._build_context(reranked)
        final_answer = self._generate_with_llm(query, reranked)

        return (
            f"问题：{query}\n\n"
            f"召回 + 重排结果：\n{context}\n\n"
            "最终回答（DeepSeek 生成）：\n"
            f"{final_answer}"
        )


def main() -> None:
    # 端到端流程：加载 -> 切分 -> 建索引 -> 召回 -> 重排 -> 生成。
    text = load_text("./story.txt")
    chunks = split_text(text, chunk_size=120, overlap=30)

    print(f"已加载文档，总切块数: {len(chunks)}")
    print("-" * 60)

    llm_client = DeepSeekClient()
    if llm_client.is_enabled():
        print("已检测到 DEEPSEEK_API_KEY，将调用 DeepSeek 生成最终回答。")
    else:
        print("未检测到 DEEPSEEK_API_KEY，将只展示检索与重排结果。")
    print("-" * 60)

    rag = SimpleRAG(chunks, llm_client=llm_client)

    # 固定问题用于快速验证召回、重排、生成流程。
    demo_questions = [
        "兰琦捡到的钥匙是什么颜色？",
        "为什么说第三层不存在？",
        "RAG-01是什么意思？",
        "时间14:07出现在哪里？",
    ]

    for q in demo_questions:
        print(rag.answer(q, top_k=3, candidate_k=10))
        print("=" * 80)

    while True:
        user_q = input("\n请输入问题（输入 exit 退出）：").strip()
        if user_q.lower() in {"exit", "quit"}:
            break
        print()
        print(rag.answer(user_q, top_k=3, candidate_k=10))
        print("=" * 80)


if __name__ == "__main__":
    main()
