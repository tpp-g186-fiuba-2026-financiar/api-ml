class ApiMlError(Exception):
    pass


class ModelNotLoadedError(ApiMlError):
    pass


class InvalidFeaturesError(ApiMlError):
    pass


class DataUnavailableError(ApiMlError):
    pass


class NotEnoughDataError(ApiMlError):
    pass


class UnknownModelError(ApiMlError):
    pass


class StaleArtifactError(ApiMlError):
    """El artefacto persistido se entreno con un feature set distinto al actual."""
