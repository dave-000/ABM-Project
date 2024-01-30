## NOTE - group project
The mesa model might give you problems, with the error message:

FutureWarning: The AgentSet is experimental. It may be changed or removed in any and all future releases, including patch releases.
We would love to hear what you think about this new feature. If you have any thoughts, share them with us here: https://github.com/projectmesa/mesa/discussions/1919       
  self._agents: AgentSet = AgentSet(agents, model)

If that is the case, go inside the environment folder, look for the mesa library packages, find the "agent.py" script, and comment line 118-125, they should be these lines:

if not self.__class__.agentset_experimental_warning_given:
   self.__class__.agentset_experimental_warning_given = True
   warnings.warn(
       "The AgentSet is experimental. It may be changed or removed in any and all future releases, including patch releases.\n"
       "We would love to hear what you think about this new feature. If you have any thoughts, share them with us here: https://github.com/projectmesa/mesa/discussions/1919",
       FutureWarning,
       stacklevel=2,
   )

Once they are commented, everything should work.

## Flood Adaptation - minimal mesa model

### Introduction
This directory contains a minimal agent-based model (ABM) implemented in Python, focused on simulating household adaptation to flood events in a social network context. It uses the Mesa framework for ABM and incorporates geographical data processing for flood depth and damage calculations.

### Installation
To set up the project environment, follow these steps:
1. Make sure you have installed a recent Python version, like 3.11 or 3.12.
2. Install the latest Mesa version (2.1.5 or above) with `pip install -U mesa`
2. Clone the repository to your local machine.
3. Install required dependencies:
   ```bash
   pip install -U geopandas shapely rasterio networkx
   ```

### File descriptions
The `model` directory contains the actual Python code for the minimal model. It has the following files:
- `agents.py`: Defines the `Households` agent class, each representing a household in the model. These agents have attributes related to flood depth and damage, and their behavior is influenced by these factors. This script is crucial for modeling the impact of flooding on individual households.
- `functions.py`: Contains utility functions for the model, including setting initial values, calculating flood damage, and processing geographical data. These functions are essential for data handling and mathematical calculations within the model.
- `model.py`: The central script that sets up and runs the simulation. It integrates the agents, geographical data, and network structures to simulate the complex interactions and adaptations of households to flooding scenarios.
- `demo.ipynb`: A Jupyter notebook titled "Flood Adaptation: Minimal Model". It demonstrates running a model and analyzing and plotting some results.
There is also a directory `input_data` that contains the geographical data used in the model. You don't have to touch it, but it's used in the code and there if you want to take a look.

### Usage
Threat this as a starting point, and feel free to modify, add or remove any components and files you find useful.
