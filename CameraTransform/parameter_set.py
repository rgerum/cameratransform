import numpy as np

STATE_DEFAULT = 0
STATE_USER_SET = 1
STATE_FIT = 2
STATE_ESTIMATE = 3

TYPE_INTRINSIC = 1 << 1
TYPE_EXTRINSIC1 = 1 << 2
TYPE_EXTRINSIC2 = 1 << 3
TYPE_DISTORTION = 1 << 4
TYPE_GPS = 1 << 5

TYPE_EXTRINSIC = TYPE_EXTRINSIC1 | TYPE_EXTRINSIC2


class Parameter(object):
    __slots__ = ["value", "range", "state", "type", "default", "callback", "std", "mean"]

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

    def sample(self):
        if self.mean is not None:
            self.value = np.random.normal(self.mean, self.std)

    def set_to_mean(self):
        if self.mean is not None:
            self.value = self.mean

    def set_stats(self, mean, std):
        self.mean = mean
        self.value = mean
        self.std = std


class DefaultAccess(object):
    parameters = {}

    def __init__(self, parameter_list):
        self.parameters = parameter_list

    def __getattr__(self, item):
        if item in self.parameters:
            return self.parameters[item].default
        return object.__getattribute__(self, item)

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
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if key in self.parameters:
            parameter_obj = self.parameters[key]
            if isinstance(value, tuple):
                parameter_obj.set_stats(*value)
            else:
                parameter_obj.mean = None
                parameter_obj.value = value
            parameter_obj.state = STATE_USER_SET
            if parameter_obj.callback is not None:
                parameter_obj.callback()
        else:
            return object.__setattr__(self, key, value)

    def get_fit_parameters(self, type=None):
        fit_param_names = []
        for name, param in self.parameters.items():
            # if a type is given only use the parameters of this type
            if type is not None and not param.type & type:
                continue
            if param.state != STATE_USER_SET or param.value is None:
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
        return object.__getattribute__(self, item)

    def __setattr__(self, key, value):
        if self.parameters is not None and key in self.parameters.parameters:
            return setattr(self.parameters, key, value)
        return object.__setattr__(self, key, value)

    def sample(self):
        callbacks = set()
        for name, parameter_obj in self.parameters.parameters.items():
            parameter_obj.sample()
            if parameter_obj.callback is not None:
                callbacks.add(parameter_obj.callback)
        for call in callbacks:
            call()

    def set_to_mean(self):
        callbacks = set()
        for name, parameter_obj in self.parameters.parameters.items():
            parameter_obj.set_to_mean()
            if parameter_obj.callback is not None:
                callbacks.add(parameter_obj.callback)
        for call in callbacks:
            call()
