from typing import List, Optional


"""
In order to work with the RPL please follow the examples below.
Create a class with a static method called `run_iteration` that takes the arguments
shown below. The method should return a list of lists of floats. Each list of floats
represents a color in RGB space. The length of the list should be equal to the `run_size`.

For submission, please submit youre whole class, and make sure it runs with the
code in `src/experiment.py`.
"""


class DummySolver:
    @staticmethod
    def run_iteration(
        target_color: List[float],
        input_colors: List[List[float]],
        previous_experiment_colors: Optional[List[List[float]]] = None,
        run_size: int = 96,
        **kwargs,
    ) -> List[List[float]]:
        return [[1.0, 0.0, 0.0] for _ in range(run_size)]


class GridSearchSolver:
    @staticmethod
    def run_iteration(
        target_color: List[float],
        input_colors: List[List[float]],
        previous_experiment_colors: Optional[List[List[float]]] = None,
        run_size: int = 96,
        **kwargs,
    ) -> List[List[float]]:
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
