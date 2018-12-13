import numpy as np
from scipy import stats
from math import log10, floor
import matplotlib.pyplot as plt
import tqdm


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


def metropolis(getLogProb, start, step=1, iterations=1e5, burn=0.1, prior_trace=None):
    if burn < 1:
        burn = int(iterations*burn)
    else:
        burn = int(burn)

    N = len(start)
    accepted = 0
    rejected = 0
    trace = []

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
    with tqdm.trange(int(iterations)) as t:
        for i in t:
            if prior_trace is not None:
                next_prior_trace = list(prior_trace.loc[np.random.randint(len(prior_trace))])[:-1]
            else:
                next_prior_trace = []

            # draw a new position
            next_pos = last_pos + np.random.normal(0, step*adaptive_scale_factor, N)
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


def plotTrace(trace, N=None, show_mean_median=True, axes=None):
    from scipy.stats import gaussian_kde

    def getAxes(name, N):
        try:
            ax_dict = plt.gcf().trace_ax_dict
        except AttributeError:
            ax_dict = dict(N=N, next_index=0)
            plt.gcf().trace_ax_dict = ax_dict
        if name not in ax_dict:
            index = ax_dict["next_index"]
            ax1 = plt.subplot(ax_dict["N"], 2, index * 2 + 1)
            if index == 0:
                ax2 = plt.subplot(ax_dict["N"], 2, index * 2 + 2)
                ax_dict["top_left"] = ax2
            else:
                ax2 = plt.subplot(ax_dict["N"], 2, index * 2 + 2, sharex=ax_dict["top_left"])
            ax_dict[name] = (ax1, ax2)
            ax_dict["next_index"] += 1
            return ax1, ax2
        return ax_dict[name]

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

        if axes is None:
            ax1, ax2 = getAxes(name, N)
        else:
            ax1, ax2 = axes[index*2:(index+1)*2]

        plt.sca(ax1)
        plt.title(name)
        x = np.linspace(min(data), max(data), 1000)
        try:
            y = gaussian_kde(data)(x)
            plt.plot(x, y, "-")
            plt.ylim(top=max([plt.gca().get_ylim()[0], np.max(y) * 1.1]))
        except Exception as err:
            print(err)
            pass
        plt.ylim(bottom=0)
        plt.ylabel("frequency")
        if show_mean_median:
            plt.axvline(data[most_probable_index], color="r")
            plt.axvline(np.mean(data), color="k")

        plt.sca(ax2)
        plt.title(name)
        plt.plot(data)
        if show_mean_median:
            plt.axhline(data[most_probable_index], color="r")
            plt.axhline(np.mean(data), color="k")
        plt.ylabel("sampled value")
    #plt.tight_layout()
    return trace


def printTraceSummary(trace, logarithmic=False):
    print("Trace %d" % len(trace))
    for index, name in enumerate(trace.columns[:-1]):
        if logarithmic[index]:
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
