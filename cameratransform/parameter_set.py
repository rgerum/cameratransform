#!/usr/bin/env python
# -*- coding: utf-8 -*-
# parameter_set.py

# Copyright (c) 2017-2021, Richard Gerum
#
# This file is part of the cameratransform package.
#
# cameratransform is free software: you can redistribute it and/or modify
# it under the terms of the MIT licence.
#
# cameratransform is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
#
# You should have received a copy of the license
# along with cameratransform. If not, see <https://opensource.org/licenses/MIT>

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .statistic import metropolis, plotTrace, Model, printTraceSummary


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
        self.mean = None
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
    trace = None
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

    def set_fit_parameters(self, names, values=None):
        if isinstance(names, dict):
            iterator = names.items()
        else:
            iterator = zip(names, values)
        callbacks = set()
        for n, v in iterator:
            if n in self.parameters:
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

    log_prob = None
    additional_parameters = None
    info_plot_functions = None

    def __init__(self):
        self.log_prob = []
        self.additional_parameters = []
        self.info_plot_functions = []

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
        if self.parameters.trace is not None:
            parameter_set = dict(self.parameters.trace.loc[np.random.randint(len(self.parameters.trace))])
            prob = parameter_set["probability"]
            del parameter_set["probability"]
            self.parameters.set_fit_parameters(parameter_set.keys(), parameter_set.values())
            return prob
        else:
            callbacks = set()
            for name, parameter_obj in self.parameters.parameters.items():
                parameter_obj.sample()
                if parameter_obj.callback is not None:
                    callbacks.add(parameter_obj.callback)
            for call in callbacks:
                call()

    def set_to_mean(self):
        if self.parameters.trace is not None:
            most_probable_index = self.parameters.trace["probability"].idxmax()
            parameter_set = dict(self.parameters.trace.loc[most_probable_index])
            if "probability" in parameter_set:
                del parameter_set["probability"]
            self.parameters.set_fit_parameters(parameter_set.keys(), parameter_set.values())
        else:
            callbacks = set()
            for name, parameter_obj in self.parameters.parameters.items():
                parameter_obj.set_to_mean()
                if parameter_obj.callback is not None:
                    callbacks.add(parameter_obj.callback)
            for call in callbacks:
                call()

    def set_trace(self, trace):
        self.parameters.trace = trace

    def addCustomoLogProbability(self, logProbability, additional_parameters=None):
        """
        Add a custom term to the camera probability used for fitting. It takes a function that should return the
        logprobability of the observables with respect to the current camera parameters.

        Parameters
        ----------
        logProbability : function
            the function that returns a legitimized probability.
        """
        self.log_prob.append(logProbability)
        if additional_parameters is not None:
            self.additional_parameters += list(additional_parameters)

    def clearLogProbability(self):
        self.log_prob = []
        self.additional_parameters = []

    def _getLogProbability_raw(self):
        """
        The same as getLogProbability, but ZeroProbability is returned as np.nan
        """
        prob = np.sum([logProb() for logProb in self.log_prob])
        return prob

    def getLogProbability(self):
        """
        Gives the sum of all terms of the log probability. This function is used for sampling and fitting.
        """
        prob = np.sum([logProb() for logProb in self.log_prob])
        return prob if not np.isnan(prob) else -np.inf

    def fit(self, parameter, **kwargs):
        estimates = []
        names = []
        ranges = []
        for param in parameter:
            names.append(param.__name__)
            estimates.append(param.value[()])
            ranges.append([param.parents.get("lower"), param.parents.get("upper")])

        def getLogProb(position):
            self.parameters.set_fit_parameters(names, position)
            return self.getLogProbability()#{n: p for n, p in zip(parameter_names, position)})

        trys = 0
        max_tries = 1000
        while np.isinf(getLogProb(estimates)) and trys < max_tries:
            estimates = [param.random()[()] for param in parameter]
            trys += 1
        if trys >= max_tries:
            raise ValueError("Could not find a starting position with non-zero probability.")

        #names = self.parameters.get_fit_parameters(param_type)
        #ranges = self.parameters.get_parameter_ranges(names)
        #estimates = self.parameters.get_parameter_defaults(names)
        if "iterations" in kwargs:
            kwargs["options"] = dict(maxiter=kwargs["iterations"])
            del kwargs["iterations"]

        def cost(p):
            self.parameters.set_fit_parameters(names, p)
            return -self.getLogProbability()

        p = minimize(cost, estimates, bounds=ranges, **kwargs)
        self.parameters.set_fit_parameters(names, p["x"])
        return p

    def metropolis(self, parameter, step=1, iterations=1e5, burn=0.1, disable_bar=False, print_trace=True):
        start = []
        parameter_names = []
        additional_parameter_names = []
        ranges = []
        for param in parameter:
            parameter_names.append(param.__name__)
            start.append(param.value[()])
            ranges.append([param.parents.get("lower"), param.parents.get("upper")])
        for param in self.additional_parameters:
            additional_parameter_names.append(param.__name__)
            start.append(param.value[()])
            ranges.append([param.parents.get("lower"), param.parents.get("upper")])
        start = np.array(start)
        step = step*np.array([p.step for p in parameter])

        def getLogProb(position):
            self.parameters.set_fit_parameters(parameter_names, position[:len(parameter_names)])
            for param, value in zip(self.additional_parameters, position[len(parameter_names):]):
                param.set_value(value)
            return self.getLogProbability()

        trys = 0
        max_tries = 1000
        while np.isinf(getLogProb(start)) and trys < max_tries:
            start = [param.random()[()] for param in parameter+self.additional_parameters]
            trys += 1
        if trys >= max_tries:
            raise ValueError("Could not find a starting position with non-zero probability.")

        trace = metropolis(getLogProb, start, step=step, iterations=iterations, burn=burn, disable_bar=disable_bar, ranges=ranges)

        # convert the trace to a pandas dataframe
        trace = pd.DataFrame(trace, columns=list(parameter_names)+list(additional_parameter_names)+["probability"])
        if print_trace:
            print(trace)
        self.set_trace(trace)
        self.set_to_mean()
        return trace

    def fridge(self, parameter, iterations=10000, **kwargs):
        if 1:
            import mock
            import sys
            # mock pymc.ZeroProbability as this is the only direct import of pymc that Bayesianfridge makes
            sys.modules.update((mod_name, mock.MagicMock()) for mod_name in ["pymc", "pymc.ZeroProbability"])
            from bayesianfridge import sample

            # we create our own model mimicking a pymc model
            model = Model(parameter+self.additional_parameters, self.getLogProbability)

        else:
            import pymc
            from bayesianfridge import sample

            param_dict = {str(p): p for p in parameter}
            additional_param_dict = {str(p): p for p in self.additional_parameters}

            @ pymc.observed
            def Ylike(value=1, param_dict=param_dict, additional_param_dict=additional_param_dict):
                self.parameters.set_fit_parameters(param_dict.keys(), param_dict.values())
                return self._getLogProbability_raw()

            model = pymc.Model(parameter + self.additional_parameters + [Ylike])

        samples, marglike = sample(model, int(iterations), **kwargs)

        columns = [p.__name__ for p in parameter + self.additional_parameters]

        data = np.array([samples[c] for c in columns]).T
        probability = []
        import tqdm
        for values in tqdm.tqdm(data):
            self.parameters.set_fit_parameters(columns, values)
            logprob = self.getLogProbability()
            probability.append(logprob)
        trace = pd.DataFrame(np.hstack((data, np.array(probability)[:, None])), columns=columns + ["probability"])

        self.set_trace(trace)
        self.set_to_mean()

        return trace

    def plotTrace(self, **kwargs):
        """
        Generate a trace plot (matplotlib window) of the current trace of the camera.
        """
        plotTrace(self.parameters.trace, **kwargs)

    def plotFitInformation(self, image=None):
        import matplotlib.pyplot as plt
        if image is not None:
            plt.imshow(image)
        for func in self.info_plot_functions:
            func()
        plt.xlim(0, self.image_width_px)
        plt.ylim(self.image_height_px, 0)

    def printTraceSummary(self):
        printTraceSummary(self.parameters.trace)
