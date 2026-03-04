import json
import logging
from typing import List, Dict, Any, Optional
import re
import httpx
from app.api.schemas import Candidate, RerankResult
from app.observability.metrics import metrics

logger = logging.getLogger("reranker-llm")

class LLMClient:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        top_n: int = 5,
        intent: Optional[str] = None,
        caller: Optional[str] = "unknown",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = 30,
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.0,
        reasoning: Optional[Dict[str, Any]] = None
    ) -> List[RerankResult]:
        if not candidates:
            return []
        metrics.record_request()
        
        # Format candidates
        candidate_text = ""
        for i, cand in enumerate(candidates):
            meta_str = f" [Context: {json.dumps(cand.metadata)}]" if cand.metadata else ""
            candidate_text += f"[{i}] (ID: {cand.id}) {cand.content}{meta_str}\n"
            
        prompt = f"""Rank candidate memories for immediate response relevance.

RANKING TARGET:
Prioritize relevance to:
1) latest user message/query,
2) most recent turn context included in query text (if present),
3) explicit communication intent.

CONTEXT:
User Query + Recent Context:
{query}
Communication Intent: {intent or 'Respond helpfully'}

CANDIDATE MEMORIES:
{candidate_text}

TASK:
Return the best {top_n} candidate indices in descending usefulness.
Prefer memories with concrete relation to the query/turn context; avoid generic identity statements unless directly relevant.

REASONING BUDGET:
If internal reasoning occurs, keep it under 300 words.

STRICT OUTPUT:
Return valid JSON ONLY in this exact shape:
{{"indices":[3,0,12]}}

EXAMPLES:
Input intent: "clarify a deployment error"
Good output: {{"indices":[4,1,8]}}
Bad output: "I think 4 is best because..." """

        try:
            logger.info(
                "Calling LLM %s for reranking %d items (caller=%s, temp=%s, max_tokens=%s)",
                model,
                len(candidates),
                caller or "unknown",
                temperature,
                max_tokens,
            )
            request_payload = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON reranker. "
                            "Never output chain-of-thought. "
                            "Never output <think>. "
                            "If internal reasoning occurs, keep it under 300 words. "
                            "Output one JSON object: {\"indices\":[...]}."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            
            if reasoning and isinstance(reasoning, dict):
                request_payload["reasoning"] = reasoning

            response = await self.client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=request_payload,
                timeout=timeout
            )
            if response.status_code == 400:
                # Some providers reject response_format; retry without it.
                request_payload.pop("response_format", None)
                response = await self.client.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=request_payload,
                    timeout=timeout
                )
            response.raise_for_status()
            result = response.json()
            finish_reason = result.get("choices", [{}])[0].get("finish_reason")
            message = result.get("choices", [{}])[0].get("message", {}) or {}
            content = message.get("content")

            # Normalize provider-specific content shapes.
            if isinstance(content, list):
                # Some providers return structured content chunks.
                parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text_val = item.get("text")
                        if isinstance(text_val, str):
                            parts.append(text_val)
                content = "\n".join(parts)
            elif content is None:
                content = ""

            # If content is empty, attempt recovery from reasoning details.
            if not str(content).strip():
                logger.warning(
                    "Reranker raw message has empty content before parsing. "
                        f"model={model}, candidates={len(candidates)}, caller={caller or 'unknown'}"
                )
                try:
                    logger.warning(
                        "Reranker raw message payload: %s",
                        json.dumps(message, ensure_ascii=False)
                    )
                    logger.warning(
                        "Reranker full completion payload: %s",
                        json.dumps(result, ensure_ascii=False)
                    )
                except Exception:
                    logger.warning("Failed to serialize raw reranker payload for logging.")
                reasoning_parts: List[str] = []
                for detail in message.get("reasoning_details", []) or []:
                    if isinstance(detail, dict):
                        text_val = detail.get("text")
                        if isinstance(text_val, str):
                            reasoning_parts.append(text_val)
                if reasoning_parts:
                    content = "\n".join(reasoning_parts)
            
            # Parse JSON array
            try:
                # Clean up markdown code blocks if present
                raw_content = content or ""
                clean_content = raw_content.replace("```json", "").replace("```", "").strip()
                
                # Remove <think> tags and their content (for reasoning models)
                clean_content = re.sub(r'<think>.*?</think>', '', clean_content, flags=re.DOTALL).strip()

                indices: List[int] = []
                parsed: Any = None
                try:
                    parsed = json.loads(clean_content)
                except json.JSONDecodeError:
                    parsed = None

                # Accept multiple structured variants to reduce brittle failures.
                if isinstance(parsed, list):
                    # Either [0, 2, 1] or [{"index": 0}, ...]
                    if parsed and isinstance(parsed[0], dict):
                        for item in parsed:
                            if isinstance(item, dict):
                                idx = item.get("index")
                                if isinstance(idx, int):
                                    indices.append(idx)
                    else:
                        indices = [i for i in parsed if isinstance(i, int)]
                elif isinstance(parsed, dict):
                    for key in ("indices", "ranked_indices", "ranking", "order"):
                        value = parsed.get(key)
                        if isinstance(value, list):
                            if value and isinstance(value[0], dict):
                                for item in value:
                                    if isinstance(item, dict) and isinstance(item.get("index"), int):
                                        indices.append(item["index"])
                            else:
                                indices = [i for i in value if isinstance(i, int)]
                            break

                # Fallback: extract first JSON-like numeric list from free text.
                if not indices:
                    array_match = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", clean_content)
                    if array_match:
                        try:
                            extracted = json.loads(array_match.group(0))
                            if isinstance(extracted, list):
                                indices = [i for i in extracted if isinstance(i, int)]
                        except Exception:
                            pass

                # Final fallback: if the model only emitted indices inside <think>,
                # try the same extraction on the raw (pre-think-strip) content.
                if not indices and raw_content:
                    raw_array_match = re.search(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", raw_content)
                    if raw_array_match:
                        try:
                            extracted = json.loads(raw_array_match.group(0))
                            if isinstance(extracted, list):
                                indices = [i for i in extracted if isinstance(i, int)]
                                logger.warning(
                                    "Recovered reranker indices from raw model content "
                                    "(array was inside reasoning text)."
                                )
                        except Exception:
                            pass

                # Heuristic fallback for reasoning-only answers:
                # extract memory ids from the final reasoning segment (often where the shortlist appears).
                if not indices and raw_content:
                    lower = raw_content.lower()
                    start_idx = max(lower.rfind("thus"), lower.rfind("therefore"), lower.rfind("maybe"))
                    segment = raw_content[start_idx:] if start_idx >= 0 else raw_content[-1200:]
                    mentions = re.findall(r"\b(?:memory|candidate)\s+(\d+)\b", segment, flags=re.IGNORECASE)
                    if mentions:
                        seen = set()
                        recovered: List[int] = []
                        for m in mentions:
                            idx = int(m)
                            if idx not in seen:
                                seen.add(idx)
                                recovered.append(idx)
                        if recovered:
                            indices = recovered
                            logger.warning(
                                "Recovered reranker indices from reasoning text fallback: %s",
                                indices[:top_n]
                            )

                if not indices:
                    logger.warning(
                        "LLM reranker response could not be parsed into indices. "
                        f"Response snippet: {clean_content[:300]!r}; "
                        f"content_len={len(raw_content)}; finish_reason={finish_reason!r}; caller={caller or 'unknown'}"
                    )
                    try:
                        logger.warning(
                            "Reranker raw message payload on parse failure: %s",
                            json.dumps(message, ensure_ascii=False)
                        )
                        logger.warning(
                            "Reranker full completion payload on parse failure: %s",
                            json.dumps(result, ensure_ascii=False)
                        )
                    except Exception:
                        logger.warning("Failed to serialize reranker payload on parse failure.")
                    reason = "empty_indices" if isinstance(parsed, (list, dict)) else "parse_failure"
                    metrics.record_fallback(reason, model or "", base_url or "")
                    return self._heuristic_fallback(query=query, candidates=candidates, top_n=top_n)
                
                results = []
                for rank, idx in enumerate(indices):
                    if isinstance(idx, int) and 0 <= idx < len(candidates):
                        cand = candidates[idx]
                        # Score is purely rank-based for now: 1.0 down to 0.5
                        score = 1.0 - (0.5 * (rank / max(len(indices), 1)))
                        results.append(RerankResult(
                            id=cand.id,
                            score=score,
                            index=idx,
                            content=cand.content
                        ))

                final_results = results[:top_n]
                if final_results:
                    logger.info(
                        "Reranker success: selected %d/%d candidates. picks=%s",
                        len(final_results),
                        len(candidates),
                        [
                            {"id": r.id, "index": r.index, "score": round(r.score, 4)}
                            for r in final_results
                        ],
                    )
                else:
                    logger.warning(
                        "Reranker produced parsed indices but final result set is empty. "
                        "indices=%s candidates=%d top_n=%d",
                        indices,
                        len(candidates),
                        top_n,
                    )

                if not final_results:
                    # Model returned indices but none mapped to valid candidates.
                    metrics.record_fallback("id_mismatch", model or "", base_url or "")
                    return self._heuristic_fallback(query=query, candidates=candidates, top_n=top_n)
                metrics.record_success()
                return final_results
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM output: {content}")
                metrics.record_fallback("parse_failure", model or "", base_url or "")
                return self._heuristic_fallback(query=query, candidates=candidates, top_n=top_n)
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            metrics.record_fallback("empty_indices", model or "", base_url or "")
            return self._heuristic_fallback(query=query, candidates=candidates, top_n=top_n)

    def _heuristic_fallback(
        self,
        *,
        query: str,
        candidates: List[Candidate],
        top_n: int,
    ) -> List[RerankResult]:
        """Deterministic lightweight fallback to avoid empty successful responses."""
        try:
            query_terms = set(re.findall(r"[a-z0-9]+", (query or "").lower()))
            scored: List[tuple[int, float, Candidate]] = []
            for idx, cand in enumerate(candidates):
                content = (cand.content or "").lower()
                terms = set(re.findall(r"[a-z0-9]+", content))
                overlap = len(query_terms & terms)
                # Small tie-breaker by shorter distance to front (stable-ish ordering)
                score = float(overlap) + (1.0 / float(idx + 1))
                scored.append((idx, score, cand))
            scored.sort(key=lambda x: x[1], reverse=True)
            selected = scored[: max(1, min(top_n, len(scored)))]
            results = [
                RerankResult(
                    id=item[2].id,
                    score=max(0.1, item[1]),
                    index=item[0],
                    content=item[2].content,
                )
                for item in selected
            ]
            logger.info(
                "Reranker heuristic fallback selected %d/%d candidates.",
                len(results),
                len(candidates),
            )
            return results
        except Exception:
            # Final safety net: preserve original order.
            safe_n = max(1, min(top_n, len(candidates)))
            return [
                RerankResult(id=c.id, score=1.0 - (i * 0.01), index=i, content=c.content)
                for i, c in enumerate(candidates[:safe_n])
            ]

llm_client = LLMClient()
