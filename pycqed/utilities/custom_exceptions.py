# -------------------------------------------
# Customized exceptions for better maintainability
# -------------------------------------------


class InterfaceMethodException(Exception):
    """
    Raised when the interface method is not implemented.
    """


class IdentifierFeedlineException(Exception):
    """
    Raised when (feedline) identifier is not correctly handled.
    """


class EnumNotDefinedException(Exception):
    """
    Raised when undefined enum is detected.
    """
