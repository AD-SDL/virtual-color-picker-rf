from argparse import ArgumentParser
from typing import List

import numpy as np

from solver import DummySolver, GridSearchSolver  # noqa
from utils import grade_color, grade_experiment, visualize_final_run, visualize_mid_run


def find_best_color(experiment: List[List[int]], target_color: List[int]) -> List[int]:
    experiment_grades = np.array(grade_experiment(experiment, target_color))

    return experiment[np.argmax(experiment_grades)]


def run(
    target_color: List[int],
    input_colors: List[List[int]],
    experiment_budget: int = 96,
    visualize: bool = False,
):
    num_trials = 0
    previous_experiment_colors = []
    best_color = None
    best_diff = float("inf")
    while num_trials < experiment_budget:
        # Run the experiment
        experiment_colors = GridSearchSolver.run_iteration(
            target_color, input_colors, previous_experiment_colors, pop_size=8
        )

        experiment_grades = grade_experiment(experiment_colors, target_color)
        trial_best_color = find_best_color(experiment_colors, target_color)
        color_diff = grade_color(trial_best_color, target_color)
        if color_diff < best_diff:
            best_color = trial_best_color
            best_diff = color_diff

        print(experiment_grades)

        # Update the experiment budget
        num_trials += len(experiment_colors)

        # Update the previous experiment colors
        previous_experiment_colors += experiment_colors

        # Visualize
        if visualize:
            visualize_mid_run(experiment_colors, trial_best_color, target_color, (8, 3))

    if visualize:
        visualize_final_run(best_color, target_color, best_diff)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_budget", type=int, default=96)
    parser.add_argument(
        "--target_color",
        metavar="N",
        type=int,
        nargs=3,
        help="a list of 3 integers",
        default=[101, 173, 95],
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    target_color = args.target_color
    experiment_budget = args.experiment_budget
    visualize = args.visualize

    # TODO fix this for RPL colors
    input_colors = [
        [1.0, 0, 0],
        [0, 1.0, 0],
        [0, 0, 1.0],
    ]

    run(target_color, input_colors, experiment_budget, visualize)
