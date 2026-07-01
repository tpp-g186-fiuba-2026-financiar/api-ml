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
