import asyncio
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from services.llm import generate_paimon_answer
from services.memory import get_recent_conversations, save_conversation
from services.retrieval import retrieve_knowledge, retrieve_personality
from services.translation import translate_to_chinese


class PaimonState(TypedDict, total=False):
    mode: Literal["chat", "translate"]
    user_id: str
    text: str
    history: list[dict]
    knowledge_context: str
    personality_context: str
    has_kqm_context: bool
    response: str


def _format_retrieval_chunks(
    chunks: list[dict],
    label: str
) -> str:
    parts = []

    for i, chunk in enumerate(
        chunks,
        start=1
    ):
        metadata = [
            f"{label} {i}",
            f"Source: {chunk['source']}",
        ]

        if chunk.get("category"):
            metadata.append(
                f"Category: {chunk['category']}"
            )

        if chunk.get("file"):
            metadata.append(
                f"File: {chunk['file']}"
            )

        if chunk.get("section"):
            metadata.append(
                f"Section: {chunk['section']}"
            )

        metadata.append(
            f"Chunk: {chunk.get('chunk_number')}"
        )

        metadata.append(
            f"Similarity: {chunk['score']:.3f}"
        )

        parts.append(
            "\n".join(metadata)
            + "\n\n"
            + chunk["text"]
        )

    return "\n\n".join(parts)


def route_mode(state: PaimonState) -> str:
    return state.get("mode", "chat")


async def load_memory_node(state: PaimonState) -> dict:
    history = await asyncio.to_thread(
        get_recent_conversations,
        state["user_id"],
    )
    return {"history": history}


async def retrieve_context_node(
    state: PaimonState
) -> dict:
    question = state["text"]

    knowledge_task = asyncio.to_thread(
        retrieve_knowledge,
        question,
        5,
    )

    personality_task = asyncio.to_thread(
        retrieve_personality,
        question,
        5,
    )

    knowledge_chunks, personality_chunks = (
        await asyncio.gather(
            knowledge_task,
            personality_task,
        )
    )

    has_kqm_context = any(
        "kqm" in chunk.get("source", "").lower()
        and chunk.get("score", 0) >= 0.50
        for chunk in knowledge_chunks
    )

    return {
        "knowledge_context": (
            _format_retrieval_chunks(
                knowledge_chunks,
                "KNOWLEDGE SOURCE",
            )
        ),
        "personality_context": (
            _format_retrieval_chunks(
                personality_chunks,
                "STYLE EXAMPLE",
            )
        ),
        "has_kqm_context": has_kqm_context,
    }


async def generate_chat_node(
    state: PaimonState
) -> dict:
    answer = await generate_paimon_answer(
        question=state["text"],
        knowledge_context=state.get(
            "knowledge_context",
            ""
        ),
        personality_context=state.get(
            "personality_context",
            ""
        ),
        history=state.get(
            "history",
            []
        ),
        has_kqm_context=state.get(
            "has_kqm_context",
            False
        ),
    )

    return {
        "response": answer
    }


async def save_memory_node(state: PaimonState) -> dict:
    await asyncio.to_thread(
        save_conversation,
        state["user_id"],
        state["text"],
        state.get("response", ""),
    )
    return {}


async def translate_node(state: PaimonState) -> dict:
    translation = await translate_to_chinese(state["text"])
    return {"response": translation}


builder = StateGraph(PaimonState)

builder.add_node("load_memory", load_memory_node)
builder.add_node("retrieve_context", retrieve_context_node)
builder.add_node("generate_chat", generate_chat_node)
builder.add_node("save_memory", save_memory_node)
builder.add_node("translate", translate_node)

builder.add_conditional_edges(
    START,
    route_mode,
    {
        "chat": "load_memory",
        "translate": "translate",
    },
)

builder.add_edge("load_memory", "retrieve_context")
builder.add_edge("retrieve_context", "generate_chat")
builder.add_edge("generate_chat", "save_memory")
builder.add_edge("save_memory", END)
builder.add_edge("translate", END)

paimon_graph = builder.compile()


async def run_paimon_chat(user_id: str, question: str) -> str:
    result = await paimon_graph.ainvoke(
        {
            "mode": "chat",
            "user_id": user_id,
            "text": question,
        }
    )
    return result.get("response", "")


async def run_translation(text: str) -> str:
    result = await paimon_graph.ainvoke(
        {
            "mode": "translate",
            "user_id": "translation",
            "text": text,
        }
    )
    return result.get("response", "")
