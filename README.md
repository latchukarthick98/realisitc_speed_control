# Realistic Speed Control In Traffic Simulation

This repository contains an environment for simulating road networks using [SUMO (Simulation of Urban MObility)](https://www.eclipse.org/sumo/) and integrating it with [OpenAI Gym](https://gym.openai.com/) for reinforcement learning. The simulation is managed through the Traffic Control Interface ([TraCI](https://sumo.dlr.de/docs/TraCI.html)), allowing interaction with SUMO from Python.

## Features

- **SUMO Road Networks**: A collection of road network configurations compatible with SUMO.

- **OpenAI Gym Integration**: Wrappers to use the SUMO simulation environment as a Gym environment, enabling compatibility with reinforcement learning libraries.

- **TraCI Interface**: Utilizes TraCI to communicate with the SUMO simulation, providing control over traffic simulation elements.

- **Reinforcement Learning Algorithm**: Implements the Soft Actor-Critic (SAC) algorithm for training an agent to optimize traffic flow.

## Getting Started

### Prerequisites

- [SUMO](https://www.eclipse.org/sumo/) installed on your system.

- Python 3.10 or later (The version used for this research is `3.10.6`).

- Install required Python packages using:

  ```bash
  pip install -r requirements.txt
  ```

### Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/latchukarthick98/realisitc_speed_control.git
   ```

2. Navigate to the project directory:

   ```bash
   cd realistic_speed_control
   ```

3. To Train the agent:

   ```bash
   python main.py --env-name sumo --alpha 0.2 --save-id 12 --cuda --automatic_entropy_tuning True
   ```

   Or

   Use the `train.ipynb` notebook

   This will start the training with SAC algorithm in the SUMO environment.


## Acknowledgments

- Special thanks to the SUMO and OpenAI Gym communities for their contributions.


## Authors

- Lakshman Karthik Ramkumar
- Tian Zhao (tzhao@uwm.edu)
## Contact

For inquiries, please contact [latchukarthick98@gmail.com].