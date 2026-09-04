class HotelDataValidationError(ValueError):
    """Raised when a hotel dataset violates the local data contract."""

    def __init__(self, message: str, *, path: str | None = None):
        self.path = path
        prefix = f"{path}: " if path else ""
        super().__init__(f"{prefix}{message}")


class HotelPreferenceValidationError(ValueError):
    """Raised when session preferences violate the public input contract."""

    def __init__(self, message: str, *, path: str | None = None):
        self.path = path
        prefix = f"{path}: " if path else ""
        super().__init__(f"{prefix}{message}")


class HotelGeminiError(RuntimeError):
    """Raised when a Gemini call or response cannot be used safely."""


class HotelHybridValidationError(ValueError):
    """Raised when a hybrid proposal batch has no valid JSON structure."""
