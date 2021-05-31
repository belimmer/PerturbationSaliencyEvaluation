# Atari (Deep Reinforcement Learning agents)

This subfolder contains the code to run our experiments on four different reinforcement learning agents (Breakout, Pac-Man, Space Invaders, and Frostbite). 
It also contains our implementation of the 4 different perturbation-based XAI approaches. 

# How to replicate our experiments:
## HIGHLIGHT State Selection:
To find HIGHLIGHT states you have to run *`highlights_stream_generator`*, which generates a *stream* folder that contains a stream of states for the agent you specified within *`highlights_stream_generator`*.
Afterward, you can use *`highlights_state_selection`* to find the HIGHLIGHT states within this stream according to the HIGHLIGHT-DIV algorithm(https://arxiv.org/abs/2005.08874).
We committed the HIGHLIGHT states that we used during our experiments in *`HIGHLIGHTS_states`*.

## Parameter Tuning:
To find suiting parameter ranges for the different segmentation algorithms we used *`segmentation_test.py`* to visually inspect different segmentation.
After that, you can use *`parameter_test.py`* to calculate which parameter combination for the saliency map approach that you specified within the script achieves the highest insertion metric score across the states in *`HIGHLIGHTS_states`*.
The results are saved in *`parameter_results`* and we committed our results there.

## Insertion Metric and Run-time Testing:
Run *`insertion_metric/insertion_metric_main.py`* to calculate the insertion metric and run-time for 1000 states of each game for all the saliency map approaches.
The results are saved in *`insertion_metric/results`*.
Then you can use *`insertion_metric/insertion_metric_plot.py`* to plot the results and save the figures in *`insertion_metric/figures`*. 
We committed our results and figures in the corresponding folders.
*`insertion_metric/times.py`* calculates the mean run-time per saliency based on the results in *`insertion_metric/results`*.

## Sanity Checks:
Run *`sanity_checks/sanity_checks_main.py`* to run the sanity check on 1000 states of each game for all the saliency map approaches.
The results are saved in *`sanity_checks/results`*.
Then you can use *`sanity_checks/sanity_checks_plot.py`* to plot the results and save the figures in *`sanity_checks/figures`*. 
We committed our results and figures in the corresponding folders.
To finetune the similarity metrics, you can use *`sanity_checks/sanity_check_check.py`* which calculates the mean similarity of randomly generated saliency maps.

## Example Images:
Use *`ImageGeneration.py`* to create example saliency maps for the states in *`HIGHLIGHTS_states`*.
The results will be saved in *`output_highlight_states`* and will also include example saliency maps for cascadingly randomized agents (which are used in the sanity checks).
We also committed the example saliency maps that we generated in this folder.

## Retraining the agents.
We committed the agents that we used in *`models`*. 
If you want to retrain them in the same way as us then please use the following commit of the OpenAi
baselines repository:
https://github.com/openai/baselines/commits/9ee399f5b20cd70ac0a871927a6cf043b478193f

We slightly adjusted the reward function, such that the agents use the actual in-game reward instead of a clipped reward.
Moreover, we scaled the reward such that the minimum reward is 1. 
To achieve this, we modified the *ClipRewardEnv* class of *baselines/commons/atari_wrappers.py*.
For MsPacman the adjustment looked like this (for the other agents just replace 10 with the minimum possible reward in those games):
   ```python
   class ClipRewardEnv(gym.RewardWrapper):
       def __init__(self, env):
           gym.RewardWrapper.__init__(self, env)

       def reward(self, reward):
           return reward/10
   ```

# Short Description of each component
- *`models`* contains the trained models
- *`custum_atari_wrapper`* contains our simplified implementation of the OpenAI Atari wrapper

- *`custom_lime.py`* contains our implementation of the LIME approach (https://arxiv.org/abs/1602.04938, https://github.com/marcotcr/lime)
- *`custom_occlusion_sensitivity.py`* contains our implementation of the occlusion sensitivity approach (https://arxiv.org/abs/1311.2901, https://github.com/sicara/tf-explain).
- *`greydanus.py`* contains our implementation of the noise sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari), with the original perturbation (i.e. the image is perturbed with a gaussian blur)
- *`custom_greydanus.py`* contains the implementation of our adjustments to the Noise Sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari), where the image is perturbed with circles of black color instead of noise.
- *`rise.py`* contains our implementation of the RISE approach and the implementation of the insertion metric (https://arxiv.org/abs/1806.07421, https://github.com/eclique/RISE)
- *`explanation.py`* mainly contains the explainer class which is used to generate the different perturbation based saliency maps

- *`highlights_stream_generator`* creates a stream of gameplay states for a given agent, which is used to identify HIGHLIGHT states
- *`highlights_state_selection`* contains code to select highlight game states(https://arxiv.org/abs/2005.08874) from the stream generated by *`highlights_stream_generator`*
- *`HIGHLIGHTS_states`* contains example game states, selected by the HIGHLIGHTS-DIV algorithm (https://arxiv.org/abs/2005.08874). These game states are used for the parameter tests and to generate example saliency maps.
- *`ImageGeneration.py`* generates example saliency maps for the highlight states in *`HIGHLIGHTS_states`*
- *`output_highlight_states`* contains the example saliency maps created by *`ImageGeneration.py`*

- *`segmentation_test.py`* is used for visually testing LIME with different segmentation algorithms
- *`parameter_test.py`* runs insertion  tests on the HIGHLIGHT images in *`HIGHLIGHTS_states`* to find the best parameters for each approach
- *`parameter_results`* contains the results obtained by *`parameter_test.py`*


- *`insertion_metric`* contains the scripts and results for the insertion metric tests

- *`sanity_checks`* contains the scripts and results for the sanity checks (https://arxiv.org/abs/1810.03292)

