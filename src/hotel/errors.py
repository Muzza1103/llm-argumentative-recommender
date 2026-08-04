class HotelDataValidationError(ValueError):
    """Raised when a hotel dataset violates the local data contract."""

    def __init__(self, message: str, *, path: str | None = None):
        self.path = path
        prefix = f"{path}: " if path else ""
        super().__init__(f"{prefix}{message}")
