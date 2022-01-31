# Atari (Deep Reinforcement Learning agents)

This subfolder contains the code to run our experiments on four different reinforcement learning agents (Breakout, Pac-Man, Space Invaders, and Frostbite). 
It also contains our implementation of the 5 different perturbation-based XAI approaches. 

# How to replicate our experiments:
## HIGHLIGHT State Selection:
To find HIGHLIGHT states you have to run *`highlights_stream_generator.py`*, which generates a *stream* folder that contains a stream of states for the agent you specified within *`highlights_stream_generator.py`*.
Afterward, you can use *`highlights_state_selection.py`* to find the HIGHLIGHT states within this stream according to the HIGHLIGHT-DIV algorithm(https://scholar.harvard.edu/files/oamir/files/highlightsmain.pdf).
We committed the HIGHLIGHT states that we used for our generated saliency map images in *`HIGHLIGHTS_states`*.

## Parameter Tuning:
The directory *`parameter_search`* contains the modules we used for finding parameters that work well for each saliency approach.

- The subfolder *`parameter_search/Verification`* deals with finding the most suitable variants of the insertion metric. 
*`full_occllusion_search.py`* calculates the insertion metric for a full stream of 1000 saliency maps created by 26 different occlusion sensitivity parameter combinations.
The raw results are saved under *`insertion_metric/results/pacman`*.
*`full_occlusion_evaluation.py`* processes and combines those raw results, calculating a mean and standard deviation value for each parameter combination.
These processed results are saved under *`parameter_search/Verification/occlusion_results_...`*, where the last term describes how the single insertion metric results were processed.
This allows us to choose the normalization methods that result in the lowest standard deviation.

- *`parameter_search/Verification`* also contains several subsets of states of the pacman agent which were either randomly selected by *`select_random_states.py`* or chosen the HIGHLIGHTS modules mentioned above.
We used *`highlights_occlusion_search.py`*, to run insertion metric evaluations on those subsets for the aforementioned 26 different occlusion sensitivity parameters.
The raw results are saved under *`parameter_results`* in the subfolder of each state subset. 
Now *`highlights_occlusion_evaluation.py`* is used to process and combine the raw results for each of those subfolders analogously to *`full_occlusion_evaluation.py`*.
In addition, *`highlights_occlusion_evaluation.py`* calculates the correlation of the parameter rankings defined by the processed results of each state subset with the rating based on the full results obtained by *`full_occlusion_evaluation.py`*.
The correlation results are saved under *`correlation_values_....csv`* and are used to determine which state subset is best suited to represent the full stream in a parameter search.
 
- After finding suitable variants of the insertion metric and a state subset that represents the full stream in the subdirectory *`parameter_search/Verification`*, we used *`parameter_search/parameter_test.py`* to test a wide variety of parameters for all saliency map approaches.
The raw results are saved under *`parameter_search/paramter_results`* as *`best_parameters_black.npy`* and *`best_parameters_random.npy`*.
*`parameter_search/parameter_test_combiner`* processes and combines those results in readable .csv files.
The final combined ranking of the parameter combinations is contained in *`final_parameters_results.csv`*.

## Insertion Metric:

The directory *`insertion_metric`* contains the scripts and results for our insertion metric tests.
Run *`insertion_metric/insertion_metric_main.py`* to calculate the insertion metric for 1000 states of each game for all saliency map approaches.
The raw results are saved in *`insertion_metric/results`*.
Then you can use *`insertion_metric/insertion_metric_plot.py`* to process and combine the raw insertion metric results.
These results are saved in  *`insertion_metric/results`* under *`pacman_mean_AUCS.csv`* for instance.
To make those results more readable, we used *`readable_table.py`*.
The resulting data frames (e.g. *`pacman_cleaned.csv`*) are saved in  *`insertion_metric/results`*.
Our final results are also committed there.

## Sanity Checks:
The directory *`sanity_checks`* contains the scripts and results for the sanity checks (https://arxiv.org/abs/1810.03292).
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

# Other Scripts

## Saliency map generation methods
- *`explanation.py`* mainly contains the explainer class which is used to generate the different perturbation based saliency maps
- *`custom_lime.py`* contains our implementation of the LIME approach (https://arxiv.org/abs/1602.04938, https://github.com/marcotcr/lime)
- *`custom_occlusion_sensitivity.py`* contains our implementation of the occlusion sensitivity approach (https://arxiv.org/abs/1311.2901, https://github.com/sicara/tf-explain).
- *`greydanus.py`* contains our implementation of the noise sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari), with the original perturbation (i.e. the image is perturbed with a gaussian blur)
- *`custom_greydanus.py`* contains the implementation of our adjustments to the Noise Sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari), where the image is perturbed with circles of black color instead of noise.
- *`rise.py`* contains our implementation of the RISE approach and the implementation of the insertion metric (https://arxiv.org/abs/1806.07421, https://github.com/eclique/RISE)
- *`sarfa.py`* contains our implementation of the SARFA approach (https://arxiv.org/abs/1912.12191, https://github.com/nikaashpuri/sarfa-saliency)


## Assorted
- *`custum_atari_wrapper.py`* contains our simplified implementation of the OpenAI Atari wrapper.
- *`evaluation_utils.py`* contains utility functions used for evaluating our results.
- *`used_parameters.py`* stores the saliency map parameters which we used in our final sanity checks (*`sanity_checks/sanity_checks_main.py`*), image generation (*`ImageGeneration.py`*) and insertion metric experiments (*`insertion_metric/insertion_metric_main.py`*, even though we still manually inserted the parameters there). 
To keep the runtime reasonable (1 Week for the insertion metric and the sanity checks each), we used the best parameters for each saliency approach that needed less than 3 seconds for a single saliency map.
