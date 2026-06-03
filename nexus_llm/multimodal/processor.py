"""Base processor abstraction for Nexus-LLM multimodal support."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseProcessor(ABC):
    """Abstract base class for multimodal data processors.

    All concrete processors (image, audio, text, etc.) must implement
    ``process``, ``validate_input``, and ``get_info``.
    """

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return a result.

        Args:
            data: Input data whose type depends on the concrete processor.

        Returns:
            Processed result.

        Raises:
            ValueError: If input validation fails.
        """
        ...

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """Check whether the input data is valid for this processor.

        Args:
            data: Input data to validate.

        Returns:
            ``True`` if the data is valid, ``False`` otherwise.
        """
        ...

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Return metadata about this processor.

        Returns:
            Dict with keys: ``name``, ``modality``, ``description``,
            ``version``.
        """
        ...

    def __repr__(self) -> str:
        info = self.get_info()
        return f"{self.__class__.__name__}(name={info.get('name')!r}, modality={info.get('modality')!r})"
