class MainException(Exception):
    """
    Parent class for custom exceptions, informs about problems
    with varius errors.
    """

    def __init__(self, text):
        super().__init__(text)

    def __str__(self):
        return "{}".format(super().__str__())


class CollectDataException(MainException):
    """
    Exception informs about no data on the website at the moment.
    """

    def __init__(self, text):
        super().__init__(text)


class ForbiddenValue(MainException):
    """Exception informs that prediction made negative values"""

    def __init__(self, text):
        super().__init__(text)
