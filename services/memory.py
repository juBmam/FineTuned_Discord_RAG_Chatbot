from collections import defaultdict, deque


MAX_HISTORY = 3

# Per-user, in-process memory. Each entry is one user/assistant exchange.
# This intentionally resets whenever the bot process/container restarts.
_memory = defaultdict(lambda: deque(maxlen=MAX_HISTORY))


def get_recent_conversations(user_id: str) -> list[dict]:
    """Return up to the last three user/assistant exchanges, oldest first."""
    return list(_memory[user_id])


def save_conversation(
    user_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Save one exchange in local memory, automatically keeping only the latest three."""
    _memory[user_id].append(
        {
            "user": user_message,
            "assistant": assistant_message,
        }
    )
