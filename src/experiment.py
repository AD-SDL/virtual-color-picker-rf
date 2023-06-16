from argparse import ArgumentParser
from typing import List

import numpy as np

from solver import InterpolationSolver, GridSearchSolver, RFSolver  # noqa
from utils import (
    grade_color,
    grade_experiment,
    mix_simulated_ratios,
    visualize_final_run,
    visualize_mid_run,
)


def find_best_color(experiment: List[List[int]], target_color: List[int]) -> List[int]:
    """
    Find the best color from an experiment and return its rgb value.
    """
    experiment_grades = np.array(grade_experiment(experiment, target_color))

    return experiment[np.argmin(experiment_grades)]


def run(
    target_color: List[int],
    input_colors: List[List[int]],
    experiment_budget: int = 96,
    visualize: bool = False,
    run_size: int = 2,
):
    num_trials = 0
    previous_experiment_ratios = []
    previous_grades = []
    best_color = None
    best_diff = float("inf")
    while num_trials < experiment_budget:
        # Run the experiment
        experiment_ratios = RFSolver.run_iteration(
            input_colors,
            previous_experiment_ratios,
            run_size=run_size,
            previous_grades=previous_grades,
        )
        # Since we are simulating, we need to mix the ratios with the input colors
        experiment_colors = [
            mix_simulated_ratios(ratios, input_colors) for ratios in experiment_ratios
        ]

        # I am not using this for any solver, but you can by passing it as a kwarg
        experiment_grades = grade_experiment(experiment_colors, target_color)  # noqa
        trial_best_color = find_best_color(experiment_colors, target_color)
        color_diff = grade_color(trial_best_color, target_color)
        if color_diff < best_diff:
            best_color = trial_best_color
            best_diff = color_diff

        # Update the experiment budget
        num_trials += len(experiment_colors)

        # Update the previous experiment colors, grades
        previous_experiment_ratios += experiment_ratios
        previous_grades += experiment_grades

        if visualize:
            # NOTE: solver_out_dim is (run_size, 3)
            # NOTE: This appears unecessarily complicated, but it comes from using a CV2
            # function to grade each plate from a picture, feel free to adapt with your
            # own function
            visualize_mid_run(
                experiment_colors, trial_best_color, target_color, (run_size, 3)
            )

    if visualize:
        visualize_final_run(best_color, target_color, best_diff)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_budget",
        type=int,
        default=96,
        help="The total number of color mixings allowed",
    )
    parser.add_argument(
        "--target_color",
        metavar="N",
        type=int,
        nargs=3,
        help="a list of 3 integers representing the target color in rgb",
        default=[101, 173, 95],
    )
    parser.add_argument(
        "--run_size", type=int, default=2, help="The number of wells to fill per iteration of the solver"
    )
    parser.add_argument(
        "--random_target", action="store_true", help="Use a random rgb color"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the experiment with a popup window each time the loop runs",
    )
    args = parser.parse_args()

    target_color = args.target_color
    if args.random_target:
        target_color = np.random.randint(0, 256, size=3).tolist()

    experiment_budget = args.experiment_budget
    visualize = args.visualize
    run_size = args.run_size

    # These colors are from the RPL color picker setup do not change.
    input_colors = [
        [255, 5, 123],  # magenta printer ink
        [0, 99, 183],  # cyan printer ink
        [240, 203, 0],  # yellow printer ink
    ]

    run(target_color, input_colors, experiment_budget, visualize, run_size)
