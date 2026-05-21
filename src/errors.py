class ApiMlError(Exception):
    pass


class ModelNotLoadedError(ApiMlError):
    pass


class InvalidFeaturesError(ApiMlError):
    pass
