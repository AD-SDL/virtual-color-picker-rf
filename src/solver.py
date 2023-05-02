from typing import List, Optional


class DummySolver:
    @staticmethod
    def run_iteration(
        target_color: List[float],
        input_colors: List[List[float]],
        previous_experiment_colors: Optional[List[List[float]]] = None,
        pop_size: int = 96,
    ) -> List[List[float]]:
        return [[1.0, 0.0, 0.0] for _ in range(pop_size)]


class GridSearchSolver:
    @staticmethod
    def run_iteration(
        target_color: List[float],
        input_colors: List[List[float]],
        previous_experiment_colors: Optional[List[List[float]]] = None,
        pop_size: int = 96,
    ) -> List[List[float]]:
        # Calculate the number of points per axis
        num_points_per_axis = int(round(pop_size ** (1 / 3)))

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
        return points[:pop_size]
