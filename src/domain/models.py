"""Domain models for agent responses."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentResponse:
    """Represents a response from the RAG agent."""

    input: str
    output: str
    source_documents: int
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Returns True if the operation was successful."""
        return self.error is None


# Backward compatibility alias
ClassificationResult = AgentResponse
