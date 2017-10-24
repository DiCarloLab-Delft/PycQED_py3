from qcodes import validators as vals

class CaseInsensitiveEnum(vals.Validator):
    """
    requires one of a provided set of values, but checks case-insensitive
    eg. Enum(val1, val2, val3)
    """

    def __init__(self, *values):
        if not len(values):
            raise TypeError('Enum needs at least one value')

        self._values = set()
        for v in values:
            if isinstance(v, str):
                self._values.add(v.lower())
            else:
                self._values.add(v)

    def validate(self, value, context=''):
        try:
            if isinstance(value, str):
                if value.lower() not in self._values:
                    raise ValueError('{} is not in {}; {}'.format(
                        repr(value), repr(self._values), context))
            else:
                if value not in self._values:
                    raise ValueError('{} is not in {}; {}'.format(
                        repr(value), repr(self._values), context))

        except TypeError as e:  # in case of unhashable (mutable) type
            e.args = e.args + ('error looking for {} in {}; {}'.format(
                repr(value), repr(self._values), context),)
            raise

    def __repr__(self):
        return '<Enum: {}>'.format(repr(self._values))
