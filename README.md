# Auto-Labs 2023 Assignment 3 

This is the repository for the third assignment of the Auto-Labs 2023 course.

## Installation 

```
conda create -n virtual-color-picker python=3.9
conda activate virtual-color-picker
pip install -r requirements.txt
```

## Usage
> **Run from src directory**

Example usage: 
```
# From the src directory
python experiment.py --visualize --random_target --experiment_budget 96 --run_size 6
```

Help: 
```
usage: experiment.py [-h] [--experiment_budget EXPERIMENT_BUDGET] [--target_color N N N] [--run_size RUN_SIZE] [--random_target] [--visualize]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_budget EXPERIMENT_BUDGET
                        The total number of color mixings allowed
  --target_color N N N  a list of 3 integers representing the target color in rgb
  --run_size RUN_SIZE   The number of wells to fill per iteration of the solver 
  --random_target       Use a random rgb color
  --visualize           Visualize the experiment with a popup window each time the loop runs
```

## Things to do

- Experiment with different run_sizes, experiment_budgets, and target colors 
  - try in low resource settings by setting experiment budgets < 16 and see how close on average you can get to the target color
  - try to see how many runs it takes to get to the target color within a delta_e of 6 (or 10, or 20) 
  - try to find a color that 'fools' the algorithm and is hard to make 
- Try to see if you can track how much of each color is used in the experiment 
  - assume a cell in a wellplate holds 300 uL of liquid, and that we are filling them to capacity every time, 
    try and add some code that tracks how much of each color is used in the experiment and tells you at the very end of the script 
- Once you have tracking, assume that we have 10mL of each color and as you run the experiments, throw an error if you see that the experiment will use more than 10mL of any color

  
