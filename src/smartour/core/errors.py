"""Application error types for Smartour."""


class SmartourError(Exception):
    """
    Base error for Smartour application failures.
    """


class ExternalServiceError(SmartourError):
    """
    Error raised when an external service request fails.
    """

    def __init__(
        self, service: str, message: str, status_code: int | None = None
    ) -> None:
        """
        Initialize an external service error.

        Args:
            service: The external service name.
            message: The sanitized error message.
            status_code: The HTTP status code when available.
        """
        self.service = service
        self.status_code = status_code
        super().__init__(message)


class PlanningInputError(SmartourError):
    """
    Error raised when itinerary planning cannot start from current inputs.
    """


class RateLimitError(SmartourError):
    """
    Error raised when a request exceeds configured rate limits.
    """
