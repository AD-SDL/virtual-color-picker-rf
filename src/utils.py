from typing import List

import matplotlib.pyplot as plt
import numpy as np
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

# https://python-colormath.readthedocs.io/en/latest/color_objects.html
from colormath.color_objects import LabColor, sRGBColor


def grade_color(color_rgb: List[int], target_rgb: List[int]) -> float:
    """Grade a color based on its distance from the target color."""
    if not isinstance(color_rgb, sRGBColor):
        color_rgb = sRGBColor(
            *color_rgb, is_upscaled=True if max(color_rgb) > 1 else False
        )
    if not isinstance(target_rgb, sRGBColor):
        target_rgb = sRGBColor(
            *target_rgb, is_upscaled=True if max(target_rgb) > 1 else False
        )

    color_lab = convert_color(color_rgb, LabColor)
    target_lab = convert_color(target_rgb, LabColor)
    delta_e = delta_e_cie2000(color_lab, target_lab)
    return delta_e


def grade_experiment(population: List[List[int]], target_rgb: List[int]) -> List[float]:
    """Grade a set of colors based on their distance from the target color."""
    return [grade_color(color_rgb, target_rgb) for color_rgb in population]


def reshape_to_plate_dims(input_colors: np.ndarray) -> np.ndarray:
    num_rows = input_colors.shape[0]
    padding_size = 12 - (num_rows % 12)
    padding = np.ones((padding_size, 3))
    # Pad the original matrix with ones
    vis_plate = np.concatenate([input_colors, padding], axis=0)
    # Reshape the padded matrix into the desired shape
    vis_plate = np.reshape(vis_plate, (-1, 12, 3))

    return vis_plate


def visualize_mid_run(
    pred_current_plate,
    cur_best_color,
    target_color,
    solver_out_dim,
    actual_current_plate=None,
):
    f, axarr = plt.subplots(2, 2)
    # set figure size to 10x10
    f.set_figheight(10)
    f.set_figwidth(10)
    predicted_current_plate = np.array(pred_current_plate).reshape(*solver_out_dim)
    vis_predicted_current_plate = reshape_to_plate_dims(predicted_current_plate)
    if actual_current_plate is not None:
        actual_current_plate = np.array(actual_current_plate).reshape(*solver_out_dim)
        vis_actual_current_plate = reshape_to_plate_dims(actual_current_plate)
        note = ""

    else:
        vis_actual_current_plate = np.ones_like(vis_predicted_current_plate)
        note = " (No measured plate provided)"

    axarr[1][0].imshow(vis_predicted_current_plate)
    axarr[1][0].set_title("Predicted plate colors")
    axarr[1][1].imshow(vis_actual_current_plate)
    axarr[1][1].set_title(f"Measured plate colors{note}")
    axarr[0][1].imshow([[target_color]])
    axarr[0][1].set_title("Target Color")
    axarr[0][0].imshow([[cur_best_color]])
    axarr[0][0].set_title("Experiment best color")
    plt.show()


def visualize_final_run(current_best_color, target_color, cur_best_diff):
    f, axarr = plt.subplots(1, 2)
    # set figure size to 10x10
    f.set_figheight(10)
    f.set_figwidth(10)
    axarr[0].imshow([[current_best_color]])
    axarr[0].set_title("Experiment best color, difference: " + str(cur_best_diff))
    axarr[1].imshow([[target_color]])
    axarr[1].set_title("Target Color")

    plt.show()
