# from: https://blog.mosthege.net/2016/11/12/json-deserialization-of-nested-objects/

import json

class JsonConvert(object):
    mappings = {}

    @classmethod
    def class_mapper(clsself, d):
        for keys, cls in clsself.mappings.items():
            if keys.issuperset(d.keys()):  # are all required arguments present?
                return cls(**d)
        else:
            # Raise exception instead of silently returning None
            raise ValueError('Unable to find a matching class for object: {!s}'.format(d))

    @classmethod
    def complex_handler(clsself, Obj):
        if hasattr(Obj, '__dict__'):
            return Obj.__dict__
        else:
            raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj)))

    @classmethod
    def register(clsself, cls):
        clsself.mappings[frozenset(tuple([attr for attr, val in cls().__dict__.items()]))] = cls
        return cls

    @classmethod
    def ToJSON(clsself, obj):
        return json.dumps(obj.__dict__, default=clsself.complex_handler, indent=4)

    @classmethod
    def FromJSON(clsself, json_str):
        return json.loads(json_str, object_hook=clsself.class_mapper)

    @classmethod
    def ToFile(clsself, obj, path):
        with open(path, 'w') as jfile:
            jfile.writelines([clsself.ToJSON(obj)])
        return path

    @classmethod
    def FromFile(clsself, filepath):
        result = None
        with open(filepath, 'r') as jfile:
            result = clsself.FromJSON(jfile.read())
        return result


@JsonConvert.register
class Employee(object):
    def __init__(self, Name: int = None, Age: int = None):
        self.Name = Name
        self.Age = Age
        return


@JsonConvert.register
class Company(object):
    def __init__(self, Name: str = "", Employees: [Employee] = None):
        self.Name = Name
        self.Employees = [] if Employees is None else Employees
        return

