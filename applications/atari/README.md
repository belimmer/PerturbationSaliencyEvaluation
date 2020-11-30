# Atari (reinforcement learning models)

This subfolder contains the code to create saliency maps for three different reinforcement learning models (Breakout, Pac-Man, and Space Invaders). It also contains the code for the 4 different perturbation based XAI approaches. 
- *`HIGHLIGHTS_states`* contains Pac-Man game states, selected by the HIGHLIGHTS-algorithm (https://arxiv.org/abs/2005.08874). These game states can be used as input of the explanation approaches.
- *`figures`* contains saved  results and the corresponding graphs of the insertion metric, for all three atari models
- *`models`* contains the trained models
- *`output_highlight_states`* contains sample saliency maps for Pac-Man game states
- *`breakout_main.py`* contains the code, which simulates Breakout games and creates saliency maps/insertion metrics for it 
- *`custum_atari_wrapper`* contains a custom implementation of the atari wrapper
- *`custom_greydanus.py`* contains a custom implementation of the noise sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari). This is the code after the adjustments (i.e. the image is perturbed with a black circle and not gaussian blur)
- *`custom_lime.py`* contains a custom implementation of the LIME approach (https://arxiv.org/abs/1602.04938, https://github.com/marcotcr/lime)
- *`custom_occlusion_sensitivity.py`* contains a custom implementation of the occlusion sensitivity approach (https://arxiv.org/abs/1311.2901, https://github.com/sicara/tf-explain). The approach was adjusted to use a black square instead of a grey one
- *`explanation.py`* contains the code to simulate Pac-Man games and create saliency maps/insertion metrics for them
- *`greydanus.py`* contains the code of the noise sensitivity approach (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari), before the adjustments (i.e. the image is perturbed with a gaussian blur)
- *`highlights_state_selection`* contains code to select highlight game states (https://arxiv.org/abs/2005.08874)
- *`insertion_metric_plot.py`* contains the code to create the insertion metric plots from the calculated insertion metric values, saved in *figures*
- *`main_atari.py`* contains code to simulate Pac-Man games
- *`rise.py`* contains the custom implementation of the RISE approach and the implementation of the insertion metric (https://arxiv.org/abs/1806.07421, https://github.com/eclique/RISE)
- *`sanity_checks.py`* contains the code to apply the sanity checks (https://arxiv.org/abs/1810.03292) on the saliency maps for the Pac-Man model
- *`space_invaders_main.py`*  contains the code, which simulates Space Invaders games and creates saliency maps/insertion metrics for it 
# How to use:
Running *`breakout_main.py`* or *`space_invaders_main.py`* will simulate either a breakout or a space invaders game of 1001 steps. A saliency map is created for every frame and used to calculate the insertion metric. The resulting scores are saved in the *`figures/backup_breakout`* or *`figures/backup_space_invaders`* folder. The computed insertion scores can be visualized using *`insertion_metric_plot.py`*.

Running *`explanation.py`* will simulate a Pac-Man game of 1001 steps if the *simulate_game* flag is set to true. Otherwise, saliency maps are created for the game states in the *`HIGHLIGHTS_states`* folder and saved in the *`output_highlight_states`* folder. 
In both cases, the generated saliency maps are used to compute the insertion metric. Resulting graphs for single frames are saved to the corresponding *`figures`* folders. The values of the insertion metric are also saved to the *`backup_predictions`* folder and can be visualized using *`insertion_metric_plot.py`*.

Running *`sanity_checks.py`* will create sanity check metrics for the Pac-Man model and save the resulting graphs in the *`figures/sanity_checks`* folder.

