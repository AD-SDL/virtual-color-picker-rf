# Auto-Labs 2023 Assignment 3 

This is the repository for the third assignment of the Auto-Labs 2023 course.

## Installation 

```
conda create -n color-picker-assignment python=3.9
conda activate color-picker-assignment
pip install -r requirements.txt
```

## Usage
> **Run from src directory**

Example usage: 
```
python experiment.py --visualize --random_target
```

Help: 
```
usage: experiment.py [-h] [--experiment_budget EXPERIMENT_BUDGET]
                     [--target_color N N N] [--random_target] [--visualize]

optional arguments:
  -h, --help            show this help message and exit
  --experiment_budget EXPERIMENT_BUDGET
                        The total number of color mixings allowed
  --target_color N N N  a list of 3 integers representing the target color in
                        rgb
  --random_target       Use a random rgb color
  --visualize           Visualize the experiment with a popup window each
                        time the loop runs
```

## Assignment
Create your own solver in the `solver.py` file and plug it into the experiment.py file on line 37. Make sure it follows the same interface as the other solvers.

Method signature for `SOLVER.run_iteration`: 
```
def run_iteration(
        input_colors: Optional[List[List[float]]] = None,
        previous_experiment_colors: Optional[List[List[float]]] = None,
        run_size: int = 96,
        **kwargs,
    ) -> List[List[float]]:
```

Please note, the solver should be stateless. The state of the solver should be able to be inferred from the previous experiment colors. This allows for integration with globus tools :slightly_smiling_face:. 
