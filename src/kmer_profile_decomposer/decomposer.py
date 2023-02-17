from __future__ import annotations
from dataclasses import dataclass, field
import functools
import itertools
from typing import Sequence, Mapping, Literal
from scipy.optimize import least_squares
from scipy.optimize._optimize import OptimizeResult
from scipy.stats import nbinom
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@dataclass
class _KmerProfileModelParameters:
    error_dispersion: float
    error_weight: float
    haploid_depth: float
    peak_dispersion_bias: float
    peak_weights: Sequence[float]

    def pack(self) -> Sequence[float]:
        packed_parameters = [
            self.error_dispersion,
            self.error_weight,
            self.haploid_depth,
            self.peak_dispersion_bias,
            *self.peak_weights,
        ]
        return packed_parameters

    @classmethod
    def unpack(cls, packed_parameters: Sequence[float]):
        (
            error_dispersion,
            error_weight,
            haploid_depth,
            peak_dispersion_bias,
            *peak_weights,
        ) = packed_parameters
        peaks = len(packed_parameters)
        return cls(
            error_dispersion=error_dispersion,
            error_weight=error_weight,
            haploid_depth=haploid_depth,
            peak_dispersion_bias=peak_dispersion_bias,
            peak_weights=peak_weights,
        )

    def get_peak_parameters(self, peak: int) -> tuple[float, float, float]:
        depth = peak * self.haploid_depth
        dispersion = depth / self.peak_dispersion_bias
        r = dispersion
        mu = depth
        p = r / (r + mu)
        weight = self.peak_weights[peak - 1]
        return (r, p, weight)

    def get_error_parameters(self) -> tuple[float, float, float]:
        mu = 1
        r = self.error_dispersion
        p = r / (r + mu)
        weight = self.error_weight
        return (r, p, weight)


@dataclass
class KmerProfileModel:
    depths: NDArray
    counts: NDArray
    peaks: int = 4
    min_depth: float = 1
    max_depth: float = float("inf")

    @property
    def _depths(self) -> NDArray:
        return self.depths[
            (self.depths >= self.min_depth) & (self.depths <= self.max_depth)
        ]

    @property
    def _counts(self) -> NDArray:
        return self.counts[
            (self.depths >= self.min_depth) & (self.depths <= self.max_depth)
        ]

    @functools.cached_property
    def _total_counts(self) -> int:
        return self.counts.sum()

    def _get_log_likelihood(self, r, p) -> float:
        depths = self._depths
        log_likelihood = nbinom.logpmf(depths, r, p)
        if np.isnan(log_likelihood).sum() > 0:
            raise ValueError()
        return log_likelihood

    def _get_component_log_counts(self, r, p, weight) -> NDArray:
        log_likelihood = self._get_log_likelihood(r, p)
        return np.log(self._total_counts * weight) + log_likelihood

    def _get_component_counts(self, r, p, weight) -> NDArray:
        counts = np.exp(self._get_component_log_counts(r, p, weight))
        if np.isnan(counts).sum() > 0:
            raise ValueError()
        # if counts.min() == 0:
        #     raise ValueError()
        return counts

    def _get_error_counts(self, parameters: _KmerProfileModelParameters) -> NDArray:
        r, p, weight = parameters.get_error_parameters()
        return self._get_component_counts(r, p, weight)

    def _get_error_log_counts(self, parameters: _KmerProfileModelParameters) -> NDArray:
        r, p, weight = parameters.get_error_parameters()
        return self._get_component_log_counts(r, p, weight)

    def _get_peak_counts(
        self, parameters: _KmerProfileModelParameters, peak: int
    ) -> NDArray:
        r, p, weight = parameters.get_peak_parameters(peak)
        return self._get_component_counts(r, p, weight)

    def _get_peak_log_counts(
        self, parameters: _KmerProfileModelParameters, peak: int
    ) -> NDArray:
        r, p, weight = parameters.get_peak_parameters(peak)
        return self._get_component_log_counts(r, p, weight)

    def _get_predicted_counts(self, parameters: _KmerProfileModelParameters) -> NDArray:
        predicted_counts = self._get_error_counts(parameters)
        for peak in range(1, self.peaks + 1):
            predicted_counts += self._get_peak_counts(parameters, peak)
        return predicted_counts

    def _get_residuals(
        self,
        parameters: _KmerProfileModelParameters,
        scale: Literal["linear", "log"] = "linear",
    ) -> NDArray:
        predicted_counts = self._get_predicted_counts(parameters)
        if scale == "linear":
            return self._counts - predicted_counts
        elif scale == "log":
            return np.log(self._counts) - np.log(predicted_counts)

    def _linear_residual_function(self, packed_parameters, *args, **kw) -> NDArray:
        parameters = _KmerProfileModelParameters.unpack(packed_parameters)
        return self._get_residuals(parameters, scale="linear")

    def _log_residual_function(self, packed_parameters, *args, **kw) -> NDArray:
        parameters = _KmerProfileModelParameters.unpack(packed_parameters)
        return self._get_residuals(parameters, scale="log")

    def fit(
        self,
        haploid_depth: float,
        residuals: Literal["linear", "log"] = "linear",
        initial_parameters={},
        least_squares_kw={},
    ) -> "_KmerProfileModelFitResult":
        default_initial_parameters = dict(
            error_dispersion=0.15,
            error_weight=1,
            peak_dispersion_bias=0.5,
            peak_weights=np.ones(self.peaks) / (self.peaks + 1) / 2,
        )
        _initial_parameters = _KmerProfileModelParameters(
            haploid_depth=haploid_depth,
            **{
                parameter: initial_parameters.get(parameter)
                if parameter in initial_parameters
                else default_value
                for parameter, default_value in default_initial_parameters.items()
            },
        )
        lower_bounds = [
            1e-10,  # error_dispersion
            1e-3,  # error_weight
            self.min_depth,  # haploid_depth
            0.001,  # peak_dispersion_bias
        ] + [
            1e-10
        ] * self.peaks  # peak_weights
        upper_bounds = [
            1,  # error_dispersion
            10000,  # error_weight
            10000,  # haploid_depth
            1000,  # peak_dispersion_bias
        ] + [
            1
        ] * self.peaks  # peak_weights
        bounds = (lower_bounds, upper_bounds)
        if residuals == "linear":
            residual_function = self._linear_residual_function
        elif residuals == "log":
            residual_function = self._log_residual_function
        else:
            raise ValueError()
        least_squares_result = least_squares(
            residual_function,
            _initial_parameters.pack(),
            bounds=bounds,
            **least_squares_kw,
        )
        fitted_parameters = _KmerProfileModelParameters.unpack(least_squares_result.x)
        fitted_model = type(self)(self.depths, self.counts, self.peaks)
        predicted_counts = fitted_model._get_predicted_counts(fitted_parameters)
        error_counts = fitted_model._get_error_counts(fitted_parameters)
        peak_counts = {
            peak: fitted_model._get_peak_counts(fitted_parameters, peak)
            for peak in range(1, self.peaks + 1)
        }
        predicted_log_counts = np.log(predicted_counts)
        error_log_counts = fitted_model._get_error_log_counts(fitted_parameters)
        error_log_probablities = (error_log_counts - predicted_log_counts) / np.log(10)
        peak_log_probablities = {}
        for peak in range(1, self.peaks + 1):
            log_counts = fitted_model._get_peak_log_counts(fitted_parameters, peak)
            peak_log_probablities[peak] = (log_counts - predicted_log_counts) / np.log(
                10
            )

        model_result = _KmerProfileModelFitResult(
            wrapped=least_squares_result,
            peaks=self.peaks,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            parameters=fitted_parameters,
            depths=self.depths,
            observed_counts=self.counts,
            predicted_counts=predicted_counts,
            error_counts=error_counts,
            peak_counts=peak_counts,
            error_log_probablities=error_log_probablities,
            peak_log_probablities=peak_log_probablities,
        )

        return model_result


@dataclass
class _KmerProfileModelFitResult:
    wrapped: OptimizeResult = field(repr=False)
    peaks: int = field()
    min_depth: float = field()
    max_depth: float = field()
    parameters: _KmerProfileModelParameters = field()
    depths: NDArray = field(repr=False)
    observed_counts: NDArray = field(repr=False)
    predicted_counts: NDArray = field(repr=False)
    error_counts: NDArray = field(repr=False)
    peak_counts: Mapping[int, NDArray] = field(repr=False)
    error_log_probablities: NDArray = field(repr=False)
    peak_log_probablities: Mapping[int, NDArray] = field(repr=False)

    def _get_color(self, copy_number: int) -> str:
        if copy_number == 0:
            return "tab:red"
        else:
            return f"C{(copy_number)//2}"

    def _get_line_style(self, copy_number: int) -> str:
        if copy_number == 0:
            return ":"
        elif copy_number % 2 == 0:
            return "-"
        else:
            return "--"

    def _get_label(self, copy_number: int) -> str:
        if copy_number == 0:
            return "Errors"
        else:
            return f"Copy number = {copy_number}"

    def get_log_probablity_dataframe(
        self, max_depth: float | None = None
    ) -> pd.DataFrame:
        _max_depth: float
        if max_depth is None:
            _max_depth = self.max_depth
        else:
            _max_depth = max_depth

        columns: dict[str, NDArray] = {}
        row_selection: NDArray = self.depths <= self.max_depth
        columns["depth"] = self.depths[row_selection]
        columns["CN=0"] = self.error_log_probablities[row_selection]
        for peak in range(1, self.peaks + 1):
            columns[f"CN={peak}"] = self.peak_log_probablities[peak][row_selection]
        df = pd.DataFrame(columns)
        return df

    def plot_model(self, ax: Axes, scale: Literal["linear", "log"] = "linear") -> None:
        depths = self.depths
        ax.plot(self.depths, self.observed_counts, color="k", label="Observed", lw=2)
        ax.plot(
            self.depths,
            self.predicted_counts,
            color="lightgray",
            label="Predicted",
            lw=2,
        )
        ax.plot(
            depths,
            self.error_counts,
            color=self._get_color(0),
            label=self._get_label(0),
            ls=self._get_line_style(0),
        )
        for peak in range(1, self.peaks + 1):
            peak_counts = self.peak_counts[peak]
            color = self._get_color(peak)
            linestyle = self._get_line_style(peak)
            label = self._get_label(peak)
            ax.plot(depths, peak_counts, color=color, ls=linestyle, label=label)

        ax.set_xlim(0, self.max_depth)
        ax.set_xlabel("Coverage depth")
        ax.set_ylabel("$\mathit{k}$-mer count")
        ax.legend(loc="upper right")
        ymin = 1000 if scale == "log" else 0
        ymax = max(counts.max() for counts in self.peak_counts.values()) * 1.1
        ax.set_ylim(ymin, ymax)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_title("$\mathit{k}$-mer profile " + f"({scale} scale)")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def plot_probablity(
        self, ax: Axes, scale: Literal["linear", "log"] = "linear"
    ) -> None:
        depths = self.depths[self.depths <= self.max_depth]
        prob = self.error_log_probablities[self.depths <= self.max_depth]
        if scale == "linear":
            prob = np.power(10, prob)
        ax.plot(
            depths,
            prob,
            color=self._get_color(0),
            ls=self._get_line_style(0),
            label=self._get_label(0),
            clip_on=False,
        )
        for peak in range(1, self.peaks + 1):
            prob = self.peak_log_probablities[peak][self.depths <= self.max_depth]
            if scale == "linear":
                prob = np.power(10, prob)
            ax.plot(
                depths,
                prob,
                color=self._get_color(peak),
                ls=self._get_line_style(peak),
                label=self._get_label(peak),
                clip_on=False,
            )
        ax.set_xlim(0, self.max_depth)
        if scale == "linear":
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(top=0)
        ax.set_xlabel("Coverage depth")
        ylabel = "Probablity" if scale == "linear" else "$\log_{10}$ probablity"
        ax.set_ylabel(ylabel)
        ax.legend(loc="lower right")
        title = (
            "Copy number probablities"
            if scale == "linear"
            else "Copy number probablities (log scale)"
        )
        ax.set_title(title)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
