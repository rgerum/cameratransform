#!/usr/bin/env python
# -*- coding: utf-8 -*-
# statistic.py

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
from scipy import stats
from math import log10, floor
import tqdm

from scipy.stats import truncnorm

def print_mean_std(x, y):
    digits = -int(floor(log10(abs(y))))
    return str(round(x, digits)) + "±" + str(round(y, 1+digits))


class normal(np.ndarray):
    def __new__(cls, sigma):
        return np.ndarray.__new__(cls, (0,))

    def __init__(self, sigma):
        self.sigma = sigma

    def __add__(self, other):
        try:
            return np.random.normal(other, self.sigma, other.shape)
        except AttributeError:
            return np.random.normal(other, self.sigma)

    def __radd__(self, other):
        return self.__add__(other)


class normal_bounded(np.ndarray):
    def __new__(cls, sigma, min, max):
        return np.ndarray.__new__(cls, (0,))

    def __init__(self, sigma, min, max):
        self.sigma = sigma
        self.min = min
        self.max = max

    def __add__(self, other):
        try:
            return stats.truncnorm.rvs((self.min-other)/self.sigma, (self.max-other)/self.sigma, other, self.sigma, size=other.shape)
        except AttributeError:
            return stats.truncnorm.rvs((self.min-other)/self.sigma, (self.max-other)/self.sigma, other, self.sigma)

    def __radd__(self, other):
        return self.__add__(other)


def metropolis(getLogProb, start, step=1, iterations=1e5, burn=0.1, prior_trace=None, disable_bar=False, ranges=None):
    if burn < 1:
        burn = int(iterations*burn)
    else:
        burn = int(burn)

    N = len(start)
    accepted = 0
    rejected = 0
    trace = []

    if ranges is None:
        ranges = np.ones((N, 1)) *np.array([-np.inf, np.inf])[None,:]
    else:
        ranges = np.array(ranges, dtype=float)
        ranges[:,0][np.isnan(ranges[:,0])] = -np.inf
        ranges[:,1][np.isnan(ranges[:,1])] = np.inf
    step = np.array(step)

    adaptive_scale_factor = 1
    tuning = True

    if prior_trace is not None:
        next_prior_trace = list(prior_trace.loc[np.random.randint(len(prior_trace))])[:-1]
    else:
        next_prior_trace = []

    # initialize the start position
    last_pos = start
    last_prob = getLogProb(list(last_pos) + next_prior_trace)
    # iterate to sample
    with tqdm.trange(int(iterations), disable=disable_bar) as t:
        for i in t:
            if prior_trace is not None:
                next_prior_trace = list(prior_trace.loc[np.random.randint(len(prior_trace))])[:-1]
            else:
                next_prior_trace = []

            # draw a new position
            # next_pos = last_pos + np.random.normal(0, step*adaptive_scale_factor, N)
            next_pos = last_pos + adaptive_scale_factor*step*truncnorm((ranges[:,0]-last_pos)/step, (ranges[:,1]-last_pos)/step).rvs()
            # get the probability
            next_prob = getLogProb(list(next_pos) + next_prior_trace)
            # calculate the acceptance ratio
            ratio = next_prob - last_prob
            if np.isinf(next_prob) and np.isinf(last_prob):
                ratio = 0
            # accept depending on the ratio (>1 means accept always, 0 never)
            r = np.random.rand()
            if ratio >= 0 or r < np.exp(ratio):
                # count accepted values
                accepted += 1
                # store position
                last_pos, last_prob = next_pos, next_prob
            else:
                rejected += 1

            # add to trace after skipping the first points
            if i > burn:
                trace.append(list(last_pos) + next_prior_trace + [last_prob])
            else:
                if i > 100 and i % 100 == 0 and tuning:
                    acc_rate = accepted / (accepted + rejected)
                    # Switch statement
                    if acc_rate < 0.001:
                        # reduce by 90 percent
                        adaptive_scale_factor *= 0.1
                    elif acc_rate < 0.05:
                        # reduce by 50 percent
                        adaptive_scale_factor *= 0.5
                    elif acc_rate < 0.2:
                        # reduce by ten percent
                        adaptive_scale_factor *= 0.9
                    elif acc_rate > 0.95:
                        # increase by factor of ten
                        adaptive_scale_factor *= 10.0
                    elif acc_rate > 0.75:
                        # increase by double
                        adaptive_scale_factor *= 2.0
                    elif acc_rate > 0.5:
                        # increase by ten percent
                        adaptive_scale_factor *= 1.1
                    else:
                        pass
                    t.set_postfix(acc_rate=acc_rate, factor=adaptive_scale_factor)
                    accepted = 0
                    rejected = 0
            if i % 1000 == 0 and accepted != 0:
                acc_rate = accepted / (accepted + rejected)
                t.set_postfix(acc_rate=acc_rate, factor=adaptive_scale_factor)

    return trace


def plotTrace(trace, N=None, show_mean_median=True, axes=None, just_distributions=False, skip=1):
    from scipy.stats import gaussian_kde
    import matplotlib.pyplot as plt

    def getAxes(name, N, width):
        try:
            trace_ax_dict = plt.gcf().trace_ax_dict
        except AttributeError:
            trace_ax_dict = dict(N=N, next_index=0)
            plt.gcf().trace_ax_dict = trace_ax_dict
        if name not in trace_ax_dict:
            index = trace_ax_dict["next_index"]
            ax1 = plt.subplot(trace_ax_dict["N"], width, index * width + 1, label=name)
            if width == 1:
                trace_ax_dict[name] = ax1
                trace_ax_dict["next_index"] += 1
                return ax1
            if index == 0:
                ax2 = plt.subplot(trace_ax_dict["N"], width, index * width + 2, label=name+"_B")
                trace_ax_dict["top_left"] = ax2
            else:
                ax2 = plt.subplot(trace_ax_dict["N"], width, index * width + 2, sharex=trace_ax_dict["top_left"], label=name+"_B")
            trace_ax_dict[name] = (ax1, ax2)
            trace_ax_dict["next_index"] += 1
            return ax1, ax2
        return trace_ax_dict[name]

    try:
        most_probable_index = trace["probability"].idxmax()
    except KeyError:
        most_probable_index = 0

    columns = [col for col in trace.columns if col != "probability"]
    if N is None:
        N = len(columns)

    if axes is None:
        plt.gcf().getAxes = getAxes

    for index, name in enumerate(columns):
        if index > N-1:
            continue
        data = trace[name]

        if just_distributions:
            if axes is None:
                ax1 = getAxes(name, N, 1)
            else:
                ax1 = axes[index * 2]
        else:
            if axes is None:
                ax1, ax2 = getAxes(name, N, 2)
            else:
                ax1, ax2 = axes[index*2:(index+1)*2]

        plt.sca(ax1)
        #plt.title(name)
        x = np.linspace(min(data), max(data), 1000)
        try:
            y = gaussian_kde(data[::skip])(x)
            plt.plot(x, y, "-")
            #plt.ylim(top=max([plt.gca().get_ylim()[0], np.max(y) * 1.1]))
        except Exception as err:
            print(err)
            pass
        #plt.ylim(bottom=0)
        plt.ylabel("frequency")
        plt.xlabel(name)
        if show_mean_median:
            plt.axvline(data[most_probable_index], color="r")
            plt.axvline(np.mean(data), color="k")

        if not just_distributions:
            plt.sca(ax2)
            plt.title(name)
            plt.plot(data[::skip])
            if show_mean_median:
                plt.axhline(data[most_probable_index], color="r")
                plt.axhline(np.mean(data), color="k")
            plt.ylabel("sampled value")
    #plt.tight_layout()
    return trace


def printTraceSummary(trace, logarithmic=False):
    print("Trace %d" % len(trace))
    for index, name in enumerate(trace.columns[:-1]):
        if logarithmic is not False and logarithmic[index]:
            data = np.exp(trace[name])
        else:
            data = trace[name]
        print(name, print_mean_std(np.mean(data), np.std(data)))

def get_all_pymc_parameters(par):
    import pymc
    parameters = []
    if isinstance(par, pymc.Stochastic):
        parameters += [par]
        for parent in par.parents.values():
            parameters += get_all_pymc_parameters(parent)
    return parameters

class FitParameter:
    __name__ = ""
    value = None
    distribution = None
    observed = False
    dtype = float

    def __init__(self, name, distribution=None, lower=None, upper=None, step=1, value=None, mean=None, std=None):
        self.__name__ = name
        if distribution is not None:
            self.distribution = distribution
        elif lower is not None and upper is not None:
            self.parents = dict(lower=lower, upper=upper)
            self.distribution = stats.uniform(loc=lower, scale=(upper-lower))
        elif mean is not None and std is not None:
            self.parents = dict(mean=mean, std=std)
            self.distribution = stats.norm(loc=mean, scale=std)
        else:
            raise ValueError("No valid distribution supplied")
        self.step = step
        self.value = np.array(value)

    def random(self):
        return self.distribution.rvs()

    def set_value(self, value):
        self.value = np.array(value)

    def logp(self):
        return self.distribution.logpdf(self.value)

    def __str__(self):
        return self.__name__


class Model:
    def __init__(self, variables, logp):
        self.variables = variables
        self.logp_func = logp

    def draw_from_prior(self):
        for variable in self.variables:
            variable.set_value(variable.random())

    def __getattr__(self, item):
        if item == "logp":
            return self.logp_func()
        return object.__getattribute__(self, item)
