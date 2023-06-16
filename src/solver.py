from typing import List, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor

"""
In order to work with the RPL please follow the examples below.
Create a class with a static method called `run_iteration` that takes the arguments
shown below. The method should return a list of lists of floats. Each list of floats
represents a color in RGB space. The length of the list should be equal to the `run_size`.

For submission, please submit youre whole class, and make sure it runs with the
code in `src/experiment.py`.
"""


def ratioSolver(input_colors, targetrgb):
    """Input colors: assumed to be # colors x rgb"""
    ratios = np.linalg.solve(input_colors.transpose(1, 0), targetrgb)
    return ratios


# Linear interpolation based on datapoints and values (defaults to finding weighted mean, r = 0.5)
# NOTE: this gets the lerped value of x
def lerp(x1, x2, v1, v2, r=0.5):
    v = v1 + r * (v2 - v1)
    return (x1 * (v2 - v) + x2 * (v - v1)) / (v2 - v1)


class RFSolver:
    @staticmethod
    def run_iteration(
        input_colors: Optional[List[List[float]]] = None,
        previous_experiment_ratios: Optional[List[List[float]]] = [],
        run_size: int = 96,
        previous_grades=[],
        sample_size=100,
    ) -> List[List[float]]:
        """Try finding optimal points by fitting random forest model

        Parameters
        ----------
        input_colors : Optional[List[List[float]]], optional
            The three input colors [magenta, cyan, yellow], by default None
        previous_experiment_ratios : Optional[List[List[float]]], optional
            The previous ratios of mixing the colors, by default None
        run_size : int, optional
            How many new points to generate, by default 96

        Returns
        -------
        List[List[float]]
           The output ratios to mix the input colors with.
        """
        # TODO: find best run_size samples to test given the previous experiments
        # NOTE: We use a hack -- allow for negative ratios :3
        # TODO: optimization should account for the actual RGB weights of each color (we want to uniformly sample the RGB space not the ratio space)

        # If there is no history, then we start with grid search of rgb space
        input_colors = np.array(input_colors)
        if len(previous_experiment_ratios) == 0:
            interval = 255 / (run_size + 1)
            rgb_samples = np.array(
                [
                    [interval * i, interval * i, interval * i]
                    for i in range(1, run_size + 1)
                ]
            )
        # If we already found solution (exact 0) then we're done
        elif len(np.where(previous_grades == 0)[0]) > 0:
            return previous_experiment_ratios[np.where(previous_grades == 0)[0][0]]
        # Edge case: history of 1 (score is meaningless -- just grid search between the two intervals)
        elif len(previous_experiment_ratios) == 1:
            previous_experiment_ratios = np.array(previous_experiment_ratios)
            previous_color = previous_experiment_ratios @ input_colors

            rgb_samples = []
            for channel in previous_color:
                if run_size == 1:
                    # Choose larger side to sample
                    if channel < 255 / 2:
                        rgb_samples.append((255 + channel) / 2)
                    else:
                        rgb_samples.append(channel / 2)
                else:
                    size0 = int(run_size // 2)
                    size1 = int(run_size // 2 + run_size % 2)
                    channel_samples = []

                    interval0 = channel / (size0 + 1)
                    interval1 = (255 - channel) / (size1 + 1)
                    channel_samples = np.array(
                        [interval0 * i for i in range(1, size0 + 1)]
                        + [interval1 * i + channel for i in range(1, size1 + 1)]
                    )
                    rgb_samples.append(channel_samples)

            if run_size == 1:
                rgb_samples = np.array(rgb_samples)
            else:
                rgb_samples = np.stack(channel_samples, axis=1)
        else:
            ### Otherwise: use random forest predictions
            assert len(previous_experiment_ratios) == len(previous_grades)

            previous_experiment_ratios = np.array(previous_experiment_ratios)
            previous_grades = np.array(previous_grades)
            previous_colors = np.einsum(
                "ij,jk->ik", previous_experiment_ratios, input_colors
            )  # H x RGB

            # Train random forest classifier on history data
            regr = RandomForestRegressor()
            regr.fit(previous_colors, previous_grades)

            # Use channel ub/lb to define the search space
            # Random sample between ub and lb and predict using RF model
            rgb_candidates = np.random.uniform(
                [0, 0, 0], [255, 255, 255], size=(sample_size, 3)
            )
            pred_scores = regr.predict(rgb_candidates)
            rgb_samples = rgb_candidates[np.argsort(pred_scores)[:run_size]]

        # Back out the ratios to generate each rgb sample
        # For each rgb sample solve 3x3 linear system
        ratios = []
        for sample in rgb_samples:
            ratios.append(ratioSolver(input_colors, sample).tolist())
        return ratios


class InterpolationSolver:
    @staticmethod
    def run_iteration(
        input_colors: Optional[List[List[float]]] = None,
        previous_experiment_ratios: Optional[List[List[float]]] = [],
        run_size: int = 96,
        previous_grades=[],
        sample_size=100,
    ) -> List[List[float]]:
        """Runs a single iteration of interpolation search
        This is basically binary search with a linear assumption on the reward function to motivate the sampling

        Parameters
        ----------
        input_colors : Optional[List[List[float]]], optional
            The three input colors [magenta, cyan, yellow], by default None
        previous_experiment_ratios : Optional[List[List[float]]], optional
            The previous ratios of mixing the colors, by default None
        run_size : int, optional
            How many new points to generate, by default 96

        Returns
        -------
        List[List[float]]
           The output ratios to mix the input colors with.
        """
        # TODO: find best run_size samples to test given the previous experiments
        # NOTE: We use a hack -- allow for negative ratios :3
        # TODO: optimization should account for the actual RGB weights of each color (we want to uniformly sample the RGB space not the ratio space)

        # If there is no history, then we start with grid search of rgb space
        input_colors = np.array(input_colors)
        if len(previous_experiment_ratios) == 0:
            interval = 255 / (run_size + 1)
            rgb_samples = np.array(
                [
                    [interval * i, interval * i, interval * i]
                    for i in range(1, run_size + 1)
                ]
            )
        # If we already found solution (exact 0) then we're done
        elif len(np.where(previous_grades == 0)[0]) > 0:
            return previous_experiment_ratios[np.where(previous_grades == 0)[0][0]]
        # Edge case: history of 1 (score is meaningless -- just grid search between the two intervals)
        elif len(previous_experiment_ratios) == 1:
            previous_experiment_ratios = np.array(previous_experiment_ratios)
            previous_color = previous_experiment_ratios @ input_colors

            rgb_samples = []
            for channel in previous_color:
                if run_size == 1:
                    # Choose larger side to sample
                    if channel < 255 / 2:
                        rgb_samples.append((255 + channel) / 2)
                    else:
                        rgb_samples.append(channel / 2)
                else:
                    size0 = int(run_size // 2)
                    size1 = int(run_size // 2 + run_size % 2)
                    channel_samples = []

                    interval0 = channel / (size0 + 1)
                    interval1 = (255 - channel) / (size1 + 1)
                    channel_samples = np.array(
                        [interval0 * i for i in range(1, size0 + 1)]
                        + [interval1 * i + channel for i in range(1, size1 + 1)]
                    )
                    rgb_samples.append(channel_samples)

            if run_size == 1:
                rgb_samples = np.array(rgb_samples)
            else:
                rgb_samples = np.stack(channel_samples, axis=1)
        else:
            ### Otherwise: interpolation search (?) over each color channel
            # Sort the history based on rgb values for each channel
            assert len(previous_experiment_ratios) == len(previous_grades)

            previous_experiment_ratios = np.array(previous_experiment_ratios)
            previous_grades = np.array(previous_grades)
            previous_colors = np.einsum(
                "ij,jk->ik", previous_experiment_ratios, input_colors
            )  # H x RGB

            # Train random forest classifier on history data
            regr = RandomForestRegressor()
            regr.fit(previous_colors, previous_grades)

            ## NOTE: gradients are almost always going from negative to positive (changes based on other channel preds though!)
            # If all positive/all negative gradients, then we know that the target is bounded by endpoints
            # Otherwise: we find the points where the gradient flips sign
            # Compute gradients
            rgb_lb = []
            rgb_ub = []
            for channelhistory in np.split(previous_colors, 3, axis=1):
                channelhistory = channelhistory.flatten()
                channel_sort = np.argsort(channelhistory)
                channel_grades = previous_grades[channel_sort]
                channel_history_sorted = channelhistory[channel_sort]
                channel_gradient = np.diff(channel_grades)

                if np.all(channel_gradient < 0):
                    lb = channel_history_sorted[-2]
                    # lb = lerp(channel_history_sorted[-2], channel_history_sorted[-1],
                    #           channel_grades[-2], channel_grades[-1])
                    ub = 255
                elif np.all(channel_gradient > 0):
                    lb = 0
                    ub = channel_history_sorted[1]
                    # ub = lerp(channel_history_sorted[0], channel_history_sorted[1],
                    #           channel_grades[0], channel_grades[1])
                else:
                    max_neg_position = np.max(np.where(channel_gradient < 0)[0])
                    min_pos_position = np.min(np.where(channel_gradient >= 0)[0])

                    if min_pos_position < max_neg_position:
                        max_neg_position = 0
                        min_pos_position = 0
                        foundstart = False
                        for i in range(len(channel_gradient) - 1):
                            if (
                                channel_gradient[i] < 0
                                and channel_gradient[i + 1] >= 0
                                and not foundstart
                            ):
                                max_neg_position = i
                                foundstart = True
                            elif (
                                channel_gradient[i] < 0 and channel_gradient[i + 1] >= 0
                            ):
                                min_pos_position = i

                    # Bounds are lerped between min and mid, mid and max
                    lb = channel_history_sorted[max_neg_position]
                    ub = channel_history_sorted[min_pos_position + 1]
                    # lb = lerp(channel_history_sorted[max_neg_position], channel_history_sorted[min_pos_position],
                    #           channel_grades[max_neg_position], channel_grades[min_pos_position])
                    # ub = lerp(channel_history_sorted[min_pos_position], channel_history_sorted[min_pos_position+1],
                    #           channel_grades[min_pos_position], channel_grades[min_pos_position+1])
                rgb_lb.append(lb)
                rgb_ub.append(ub)

            # Use channel ub/lb to define the search space
            # Random sample between ub and lb and predict using RF model
            rgb_ub = np.array(rgb_ub)
            rgb_lb = np.array(rgb_lb)
            rgb_candidates = np.random.uniform(rgb_lb, rgb_ub, size=(sample_size, 3))
            # intervals = (rgb_ub - rgb_lb)/(sample_size + 1)
            # rgb_candidates = np.array([rgb_lb + intervals * i for i in range(1, sample_size + 1)])
            pred_scores = regr.predict(rgb_candidates)
            rgb_samples = rgb_candidates[np.argsort(pred_scores)[:run_size]]

        # Back out the ratios to generate each rgb sample
        # For each rgb sample solve 3x3 linear system
        ratios = []
        for sample in rgb_samples:
            ratios.append(ratioSolver(input_colors, sample).tolist())
        return ratios


class DummySolver:
    @staticmethod
    def run_iteration(
        input_colors: Optional[List[List[float]]] = None,
        previous_experiment_ratios: Optional[List[List[float]]] = None,
        run_size: int = 96,
        **kwargs,
    ) -> List[List[float]]:
        """Runs a single iteration of grid search.

        Parameters
        ----------
        input_colors : Optional[List[List[float]]], optional
            The three input colors [magenta, cyan, yellow], by default None
        previous_experiment_ratios : Optional[List[List[float]]], optional
            The previous ratios of mixing the colors, by default None
        run_size : int, optional
            How many new points to generate, by default 96

        Returns
        -------
        List[List[float]]
           The output ratios to mix the input colors with.
        """
        return [[1.0, 0.0, 0.0] for _ in range(run_size)]


class GridSearchSolver:
    @staticmethod
    def run_iteration(
        input_colors: Optional[List[List[float]]] = None,
        previous_experiment_ratios: Optional[List[List[float]]] = None,
        run_size: int = 96,
        **kwargs,
    ) -> List[List[float]]:
        """Runs a single iteration of grid search.

        Parameters
        ----------
        input_colors : Optional[List[List[float]]], optional
            The three input colors [magenta, cyan, yellow], by default None
        previous_experiment_ratios : Optional[List[List[float]]], optional
            The previous ratios of mixing the colors, by default None
        run_size : int, optional
            How many new points to generate, by default 96

        Returns
        -------
        List[List[float]]
           The output ratios to mix the input colors with.
        """
        # Calculate the number of points per axis
        num_points_per_axis = int(round(run_size ** (1 / 3)))

        # Calculate the step size for each axis
        step_size = 1 / (num_points_per_axis - 1)

        # Generate the grid points
        points = []
        for x in range(num_points_per_axis):
            for y in range(num_points_per_axis):
                for z in range(num_points_per_axis):
                    point = [x * step_size, y * step_size, z * step_size]
                    points.append(point)

        # Return the sampled points
        return points[:run_size]
