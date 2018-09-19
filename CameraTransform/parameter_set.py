
STATE_DEFAULT = 0
STATE_USER_SET = 1
STATE_FIT = 2
STATE_ESTIMATE = 3

TYPE_INTRINSIC = 1 << 1
TYPE_EXTRINSIC1 = 1 << 2
TYPE_EXTRINSIC2 = 1 << 3
TYPE_DISTORTION = 1 << 4


class Parameter(object):
    __slots__ = ["value", "range", "state", "type", "default", "callback"]

    def __init__(self, value=None, range=None, default=None, state=None, type=TYPE_INTRINSIC, callback=None):
        self.value = value
        if range is not None:
            self.range = range
        else:
            self.range = (None, None)
        self.default = default
        if state is None:
            if value is None:
                self.state = STATE_DEFAULT
            else:
                self.state = STATE_USER_SET
        self.type = type
        self.callback = callback


class DefaultAccess(object):
    parameters = {}

    def __init__(self, parameter_list):
        self.parameters = parameter_list

    def __getattr__(self, item):
        if item in self.parameters:
            return self.parameters[item].default

    def __setattr__(self, key, value):
        if key in self.parameters:
            parameter_obj = self.parameters[key]
            parameter_obj.default = value
            if parameter_obj.value is None and parameter_obj.callback is not None:
                parameter_obj.callback()
        else:
            return object.__setattr__(self, key, value)


class ParameterSet(object):
    parameters = {}

    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.defaults = DefaultAccess(self.parameters)

    def __getattr__(self, item):
        if item in self.parameters:
            parameter_obj = self.parameters[item]
            if parameter_obj.value is not None:
                return parameter_obj.value
            return parameter_obj.default

    def __setattr__(self, key, value):
        if key in self.parameters:
            parameter_obj = self.parameters[key]
            parameter_obj.value = value
            parameter_obj.state = STATE_USER_SET
            if parameter_obj.callback is not None:
                parameter_obj.callback()
        else:
            return object.__setattr__(self, key, value)

    def get_fit_parameters(self):
        fit_param_names = []
        for name, param in self.parameters.items():
            if param.state != STATE_USER_SET:
                fit_param_names.append(name)
        return fit_param_names

    def set_fit_parameters(self, names, values):
        callbacks = set()
        for n, v in zip(names, values):
            self.parameters[n].value = v
            self.parameters[n].state = STATE_FIT
            if self.parameters[n].callback is not None:
                callbacks.add(self.parameters[n].callback)
        for call in callbacks:
            call()

    def get_parameter_defaults(self, names):
        return [self.parameters[n].default for n in names]

    def get_parameter_ranges(self, names):
        return [self.parameters[n].range for n in names]


class ClassWithParameterSet(object):
    parameters = None

    def __getattr__(self, item):
        if self.parameters is not None:
            if item == "defaults" or item in self.parameters.parameters:
                return getattr(self.parameters, item)

    def __setattr__(self, key, value):
        if self.parameters is not None and key in self.parameters.parameters:
            return setattr(self.parameters, key, value)
        return object.__setattr__(self, key, value)
