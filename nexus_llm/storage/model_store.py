"""Nexus-LLM Model Metadata Storage.

Manages persistent storage of model metadata including registration,
search, update, and deletion of model records. Tracks model loading
states, capabilities, and performance characteristics.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from nexus_llm.storage.database import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class ModelRecord:
    """A persisted model metadata record.

    Attributes:
        id: Auto-incremented database ID.
        name: Short model name (unique identifier).
        full_name: Full model name (e.g., HuggingFace repo ID).
        model_type: Type of model (causal_lm, seq2seq, etc.).
        size: Model size label (e.g., '7B', '13B').
        parameter_count: Exact parameter count.
        context_length: Maximum context window length.
        device: Compute device (auto, cuda, cpu).
        precision: Weight precision (fp16, fp32, int8, int4).
        description: Human-readable model description.
        license: Model license identifier.
        local_path: Local filesystem path to model weights.
        is_loaded: Whether the model is currently in memory.
        tags: Categorization tags.
        created_at: Registration timestamp.
        updated_at: Last update timestamp.
        metadata: Additional metadata as JSON.
    """

    id: Optional[int] = None
    name: str = ""
    full_name: str = ""
    model_type: str = "causal_lm"
    size: str = ""
    parameter_count: Optional[int] = None
    context_length: int = 2048
    device: str = "auto"
    precision: str = "fp16"
    description: str = ""
    license: str = ""
    local_path: Optional[str] = None
    is_loaded: bool = False
    tags: List[str] = field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelStore:
    """Persistent storage for model metadata.

    Provides CRUD operations and search capabilities for registered
    models, with support for filtering by type, size, device, and tags.

    Attributes:
        db: Database manager instance.
    """

    def __init__(self, db: DatabaseManager) -> None:
        """Initialize the model store.

        Args:
            db: DatabaseManager instance for database access.
        """
        self.db = db

    def register_model(self, record: ModelRecord) -> ModelRecord:
        """Register a new model in the store.

        Args:
            record: ModelRecord with model information.

        Returns:
            The registered ModelRecord with ID assigned.

        Raises:
            ValueError: If a model with the same name already exists.
        """
        existing = self.get_model_by_name(record.name)
        if existing is not None:
            raise ValueError(f"Model '{record.name}' is already registered")

        now = datetime.now().isoformat()
        tags_json = json.dumps(record.tags)
        metadata_json = json.dumps(record.metadata)

        cursor = self.db.execute(
            """INSERT INTO models
               (name, full_name, model_type, size, parameter_count, context_length,
                device, precision, description, license, local_path, is_loaded,
                tags, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.name,
                record.full_name,
                record.model_type,
                record.size,
                record.parameter_count,
                record.context_length,
                record.device,
                record.precision,
                record.description,
                record.license,
                record.local_path,
                1 if record.is_loaded else 0,
                tags_json,
                now,
                now,
                metadata_json,
            ),
        )
        record.id = cursor.lastrowid
        record.created_at = now
        record.updated_at = now

        logger.info(f"Registered model '{record.name}' (id={record.id})")
        return record

    def get_model(self, model_id: int) -> Optional[ModelRecord]:
        """Retrieve a model by its database ID.

        Args:
            model_id: The auto-incremented database ID.

        Returns:
            ModelRecord if found, None otherwise.
        """
        row = self.db.fetch_one("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            return None
        return self._row_to_model(row)

    def get_model_by_name(self, name: str) -> Optional[ModelRecord]:
        """Retrieve a model by its name.

        Args:
            name: The unique model name.

        Returns:
            ModelRecord if found, None otherwise.
        """
        row = self.db.fetch_one("SELECT * FROM models WHERE name = ?", (name,))
        if row is None:
            return None
        return self._row_to_model(row)

    def list_models(
        self,
        model_type: Optional[str] = None,
        is_loaded: Optional[bool] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        tag: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[ModelRecord], int]:
        """List models with optional filtering and pagination.

        Args:
            model_type: Filter by model type.
            is_loaded: Filter by loaded status.
            device: Filter by compute device.
            precision: Filter by weight precision.
            tag: Filter by tag (matches if tag is in the tags list).
            limit: Maximum models to return.
            offset: Number of models to skip.

        Returns:
            Tuple of (list of ModelRecord, total count).
        """
        conditions = []
        params: List[Any] = []

        if model_type:
            conditions.append("model_type = ?")
            params.append(model_type)
        if is_loaded is not None:
            conditions.append("is_loaded = ?")
            params.append(1 if is_loaded else 0)
        if device:
            conditions.append("device = ?")
            params.append(device)
        if precision:
            conditions.append("precision = ?")
            params.append(precision)
        if tag:
            conditions.append("tags LIKE ?")
            params.append(f'%"{tag}"%')

        where = " AND ".join(conditions) if conditions else "1=1"

        count = self.db.fetch_value(
            f"SELECT COUNT(*) FROM models WHERE {where}",
            tuple(params),
        ) or 0

        rows = self.db.fetch_all(
            f"SELECT * FROM models WHERE {where} ORDER BY name ASC LIMIT ? OFFSET ?",
            tuple(params + [limit, offset]),
        )

        models = [self._row_to_model(row) for row in rows]
        return models, count

    def update_model(
        self,
        name: str,
        is_loaded: Optional[bool] = None,
        local_path: Optional[str] = None,
        device: Optional[str] = None,
        precision: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> bool:
        """Update a model's attributes.

        Args:
            name: The model name to update.
            is_loaded: New loaded status.
            local_path: New local path.
            device: New compute device.
            precision: New weight precision.
            tags: New tags list (replaces existing).
            metadata: New metadata (merged with existing).
            description: New description.

        Returns:
            True if the model was updated.
        """
        updates: List[str] = []
        params: List[Any] = []

        if is_loaded is not None:
            updates.append("is_loaded = ?")
            params.append(1 if is_loaded else 0)
        if local_path is not None:
            updates.append("local_path = ?")
            params.append(local_path)
        if device is not None:
            updates.append("device = ?")
            params.append(device)
        if precision is not None:
            updates.append("precision = ?")
            params.append(precision)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if description is not None:
            updates.append("description = ?")
            params.append(description)
        if metadata is not None:
            existing = self.get_model_by_name(name)
            if existing:
                merged = {**existing.metadata, **metadata}
                updates.append("metadata = ?")
                params.append(json.dumps(merged))

        if not updates:
            return False

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(name)

        self.db.execute(
            f"UPDATE models SET {', '.join(updates)} WHERE name = ?",
            tuple(params),
        )
        return True

    def delete_model(self, name: str) -> bool:
        """Delete a model from the store.

        Args:
            name: The model name to delete.

        Returns:
            True if the model was deleted.
        """
        cursor = self.db.execute("DELETE FROM models WHERE name = ?", (name,))
        if cursor.rowcount > 0:
            logger.info(f"Deleted model '{name}'")
            return True
        return False

    def mark_loaded(self, name: str) -> bool:
        """Mark a model as currently loaded in memory.

        Args:
            name: The model name.

        Returns:
            True if the model was updated.
        """
        return self.update_model(name, is_loaded=True)

    def mark_unloaded(self, name: str) -> bool:
        """Mark a model as not currently loaded in memory.

        Args:
            name: The model name.

        Returns:
            True if the model was updated.
        """
        return self.update_model(name, is_loaded=False)

    def get_loaded_models(self) -> List[ModelRecord]:
        """Get all currently loaded models.

        Returns:
            List of ModelRecord for loaded models.
        """
        rows = self.db.fetch_all(
            "SELECT * FROM models WHERE is_loaded = 1 ORDER BY name ASC"
        )
        return [self._row_to_model(row) for row in rows]

    def get_models_by_type(self, model_type: str) -> List[ModelRecord]:
        """Get all models of a specific type.

        Args:
            model_type: The model type to filter by.

        Returns:
            List of matching ModelRecord objects.
        """
        rows = self.db.fetch_all(
            "SELECT * FROM models WHERE model_type = ? ORDER BY name ASC",
            (model_type,),
        )
        return [self._row_to_model(row) for row in rows]

    def search_models(self, query: str, limit: int = 20) -> List[ModelRecord]:
        """Search models by name, full_name, or description.

        Args:
            query: Search query string.
            limit: Maximum results to return.

        Returns:
            List of matching ModelRecord objects.
        """
        pattern = f"%{query}%"
        rows = self.db.fetch_all(
            """SELECT * FROM models
               WHERE name LIKE ? OR full_name LIKE ? OR description LIKE ?
               ORDER BY name ASC LIMIT ?""",
            (pattern, pattern, pattern, limit),
        )
        return [self._row_to_model(row) for row in rows]

    def get_model_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics about registered models.

        Returns:
            Dictionary with model statistics.
        """
        total = self.db.fetch_value("SELECT COUNT(*) FROM models") or 0
        loaded = self.db.fetch_value("SELECT COUNT(*) FROM models WHERE is_loaded = 1") or 0
        types = self.db.fetch_all(
            "SELECT model_type, COUNT(*) as count FROM models GROUP BY model_type"
        )

        total_params = self.db.fetch_value(
            "SELECT SUM(parameter_count) FROM models WHERE parameter_count IS NOT NULL"
        )

        return {
            "total_models": total,
            "loaded_models": loaded,
            "model_types": {row["model_type"]: row["count"] for row in types},
            "total_parameters": total_params or 0,
        }

    def _row_to_model(self, row: Dict[str, Any]) -> ModelRecord:
        """Convert a database row to a ModelRecord."""
        tags = row.get("tags", "[]")
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = []

        metadata = row.get("metadata", "{}")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return ModelRecord(
            id=row.get("id"),
            name=row.get("name", ""),
            full_name=row.get("full_name", ""),
            model_type=row.get("model_type", "causal_lm"),
            size=row.get("size", ""),
            parameter_count=row.get("parameter_count"),
            context_length=row.get("context_length", 2048),
            device=row.get("device", "auto"),
            precision=row.get("precision", "fp16"),
            description=row.get("description", ""),
            license=row.get("license", ""),
            local_path=row.get("local_path"),
            is_loaded=bool(row.get("is_loaded", 0)),
            tags=tags,
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
            metadata=metadata,
        )
