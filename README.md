## Time Varying Discount
---

Discount factor in Deep Q-Networks serves a dual role:

- it explicitly specifies some intertemporal preferences (discounting the future)
- it implicitly encodes confidence on bootstrapping from function approximator (weighing the past)

The time varying discount, or myopic schedule, is a practical method to weigh earlier experience less during the myopic fraction training period. The experiment results demonstrate that the simple myopia scheme is a robust and effective way to improve performance for DRL algorithms.

To see more details, feel free to checkout this [blogpost](https://yuhao-wan.github.io/blog/exploring-gamma/). 

This work (and this repo) is ongoing. Stay tuned for more principled way to adjust discount factor throughout training.

## Prerequisites
---

1. Create and activate a virtual environment

2. Install TensorFlow if you haven’t

2. Install OpenAI Baselines package

	```
	git clone https://github.com/openai/baselines.git
	cd baselines
	pip install -e .
	```

3. Install requirements of time-varying-discount

	```
	git clone https://github.com/Yuhao-Wan/time-varying-discount.git
	cd time-varying-discount
	pip install -r requirements.txt
	```

4. (optional) To use MuJoCo, follow the installation guide at [mujoco-py](https://github.com/openai/mujoco-py).


## Training Models
---

### Example 1. DQN with Time Varying Discount

To run the Baselines DQN with modification of initial myopia scheme on one of the Gridworld environments (designed using [pycolab](https://github.com/deepmind/pycolab)), run the following commands:

	
	cd experiments
	python train_dense.py seed myopic_fraction final_discount path_name gpu
	
For example, one could run:
	
	python train_dense.py 1 0.2 0.99 02099 0

No worries if you don’t have gpu, just put ```0``` for ```gpu```, and it will use your cpu to train. 

More modifications:

In ```train_dense.py```, you can also modify the path via ```dirs```, or any parameters specified in ```kwargs```.

### Example 2. PPO with MuJoCo HalfCheetah

To run the Baselines PPO with modification of varying lambda scheme on MuJoCo environments, run the following commands:

```
cd experiments
python mujoco_ascend.py gpu env_id seed lambda_fraction final_lambda path_name
``` 

For example, one could run:

```
python mujoco_ascend.py 0 HalfCheetah-v2 1 0.2 0.95 02095
```

No worries if you don’t have gpu, just put ```0``` for ```gpu```, and it will use your cpu to train. 

More modifications:

In ```mujoco_ascend.py```, you can also modify the path via ```dirs```, or any parameters specified in ```kwargs```.

## Loading and visualizing saved models
---
To visualize the saved model for the DQN with myopia on Gridworld environment, run the following commands:

```
cd experiments
python enjoy_dense.py
```

## Loading and visualizing learning curves
---
To plot learning curves, run the following command:

```
python plot.py
```

The plot figure will be saved at the location specified by ```dirs``` in ```plot.py```.