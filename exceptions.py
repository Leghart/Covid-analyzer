class MainException(Exception):
    """
    Parent class for custom exceptions, informs about problems
    with varius errors.
    """

    def __init__(self, text, description):
        super().__init__(text)
        self.description = description

    def __str__(self):
        return '{}: {}'.format(super().__str__(), self.description)


class CollectDataException(MainException):
    """
    Exception informs about no data on the website at the moment.
    """

    def __init__(self, text):
        super().__init__(text, 'Data wasnt uploaded yet. Please try later ...')
