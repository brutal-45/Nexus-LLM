"""Nexus-LLM Workflow Errors.

Provides custom exception classes for workflow-related errors,
including node execution failures, validation errors, and
scheduling errors.
"""

from typing import Any, Dict, Optional


class WorkflowError(Exception):
    """Base exception for all workflow errors.

    Attributes:
        workflow_id: ID of the workflow that caused the error.
        details: Additional error details.
    """

    def __init__(
        self,
        message: str,
        workflow_id: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.workflow_id = workflow_id
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "workflow_id": self.workflow_id,
            "details": self.details,
        }


class NodeExecutionError(WorkflowError):
    """Raised when a workflow node fails during execution.

    Attributes:
        node_id: ID of the node that failed.
    """

    def __init__(
        self,
        message: str,
        node_id: str = "",
        workflow_id: str = "",
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message, workflow_id=workflow_id)
        self.node_id = node_id
        self.cause = cause
        self.details["node_id"] = node_id
        if cause:
            self.details["cause"] = str(cause)


class WorkflowValidationError(WorkflowError):
    """Raised when a workflow definition fails validation.

    Attributes:
        validation_errors: List of validation error descriptions.
    """

    def __init__(
        self,
        message: str,
        validation_errors: Optional[list] = None,
        workflow_id: str = "",
    ) -> None:
        super().__init__(message, workflow_id=workflow_id)
        self.validation_errors = validation_errors or []
        self.details["validation_errors"] = self.validation_errors


class WorkflowTimeoutError(WorkflowError):
    """Raised when a workflow execution exceeds its timeout.

    Attributes:
        timeout_seconds: The timeout value that was exceeded.
        elapsed_seconds: Actual elapsed time.
    """

    def __init__(
        self,
        message: str,
        timeout_seconds: float = 0.0,
        elapsed_seconds: float = 0.0,
        workflow_id: str = "",
    ) -> None:
        super().__init__(message, workflow_id=workflow_id)
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds
        self.details["timeout_seconds"] = timeout_seconds
        self.details["elapsed_seconds"] = elapsed_seconds


class CycleDetectedError(WorkflowError):
    """Raised when a cycle is detected in the workflow graph."""

    def __init__(self, message: str = "Workflow contains a cycle", workflow_id: str = "") -> None:
        super().__init__(message, workflow_id=workflow_id)


class ScheduleError(WorkflowError):
    """Raised when a scheduling operation fails."""

    def __init__(
        self,
        message: str,
        schedule_id: str = "",
        workflow_id: str = "",
    ) -> None:
        super().__init__(message, workflow_id=workflow_id)
        self.schedule_id = schedule_id
        self.details["schedule_id"] = schedule_id


class SerializationError(WorkflowError):
    """Raised when workflow serialization or deserialization fails."""

    def __init__(self, message: str, format: str = "", workflow_id: str = "") -> None:
        super().__init__(message, workflow_id=workflow_id)
        self.format = format
        self.details["format"] = format
