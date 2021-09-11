class ExtorchException(Exception):
    pass


class InvalidValueException(ExtorchException):
    pass


class InvalidConfigException(ExtorchException):
    pass


def expect(bool_expr: bool, message: str = "", exception_type: Exception = ExtorchException):
    if not bool_expr:
        raise exception_type(message)
