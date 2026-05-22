"""Conversation repository implementations."""

from smartour.domain.conversation import Conversation
from smartour.infrastructure.database import SQLiteDatabase


class InMemoryConversationRepository:
    """
    Process-local in-memory conversation repository.
    """

    def __init__(self) -> None:
        """
        Initialize the repository.
        """
        self.conversations: dict[str, Conversation] = {}

    async def save(self, conversation: Conversation) -> None:
        """
        Save a conversation.

        Args:
            conversation: The conversation to save.
        """
        self.conversations[conversation.id] = conversation.model_copy(deep=True)

    async def get(self, conversation_id: str) -> Conversation | None:
        """
        Return a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation when found.
        """
        conversation = self.conversations.get(conversation_id)
        if conversation is None:
            return None
        return conversation.model_copy(deep=True)

    async def list_by_user(self, user_id: str) -> list[Conversation]:
        """
        Return conversations owned by a user.

        Args:
            user_id: The owner user ID.

        Returns:
            The user's conversations sorted by creation time descending.
        """
        conversations = [
            conversation.model_copy(deep=True)
            for conversation in self.conversations.values()
            if conversation.user_id == user_id
        ]
        return sorted(
            conversations,
            key=lambda conversation: conversation.created_at,
            reverse=True,
        )


class SQLiteConversationRepository:
    """
    SQLite-backed conversation repository.
    """

    def __init__(self, database: SQLiteDatabase) -> None:
        """
        Initialize the repository.

        Args:
            database: The SQLite database.
        """
        self.database = database

    async def save(self, conversation: Conversation) -> None:
        """
        Save a conversation.

        Args:
            conversation: The conversation to save.
        """
        async with self.database.connect() as connection:
            await connection.execute(
                """
                INSERT INTO conversations (
                    id, user_id, state, payload, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    state = excluded.state,
                    payload = excluded.payload,
                    updated_at = excluded.updated_at
                """,
                (
                    conversation.id,
                    conversation.user_id,
                    conversation.state.value,
                    conversation.model_dump_json(),
                    conversation.created_at.isoformat(),
                    conversation.updated_at.isoformat(),
                ),
            )

    async def get(self, conversation_id: str) -> Conversation | None:
        """
        Return a conversation by ID.

        Args:
            conversation_id: The conversation ID.

        Returns:
            The conversation when found.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                "SELECT payload FROM conversations WHERE id = ?",
                (conversation_id,),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            return None
        return Conversation.model_validate_json(row["payload"])

    async def list_by_user(self, user_id: str) -> list[Conversation]:
        """
        Return conversations owned by a user.

        Args:
            user_id: The owner user ID.

        Returns:
            The user's conversations sorted by creation time descending.
        """
        async with (
            self.database.connect() as connection,
            connection.execute(
                """
                SELECT payload FROM conversations
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ) as cursor,
        ):
            rows = await cursor.fetchall()
        return [Conversation.model_validate_json(row["payload"]) for row in rows]
