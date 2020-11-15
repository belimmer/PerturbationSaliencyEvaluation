# Atari (reinforcement learning models)

This subfolder contains the code to create saliency maps for three different reinforcement learning models (Breakout, Pac-Man, and Space Invaders). It also contains the code for the 4 different perturbation based XAI approaches. 
- *HIGHLIGHTS_states* contains Pac-Man game states, selected by the HIGHLIGHTS-algorithm (https://arxiv.org/abs/2005.08874). These game states can be used as input of the explanation approaches.
- *figures* contains saved  results and the corresponding graphs of the insertion metric, for all three atari models
- *models* contains the trained models
- *output_highlight_states* contains sample saliency maps for Pac-Man game states
- *breakout_main.py* contains the code, which simulates Breakout games and creates saliency maps/insertion metrics for it 
- *custum_atari_wrapper* contains a custom implementation of the atari wrapper
- *custom_greydanus.py* contains a custom implementation of the noise sensitivity approach (https://arxiv.org/abs/1711.00138). This is the code after the adjustments (i.e. the image is perturbed with a black circle and not gaussian blur)
- *custom_lime.py* contains a custom implementation of the LIME approach (https://arxiv.org/abs/1602.04938)
- *custom_occlusion_sensitivity.py* contains a custom implementation of the occlusion sensitivity approach (https://arxiv.org/abs/1311.2901). The approach was adjusted to use a black square instead of a grey one
- *explanation.py* contains the code to simulate Pac-Man games and create saliency maps/insertion metrics for them
- *greydanus.py* contains the code of the noise sensitivity approach (https://arxiv.org/abs/1711.00138), before the adjustments (i.e. the image is perturbed with a gaussian blur)
- *highlights_state_selection* contains code to select highlight game states (https://arxiv.org/abs/2005.08874)
- *insertion_metric_plot.py* contains the code to create the insertion metric plots from the calculated insertion metric values, saved in *figures*
- *main_atari.py* contains code to simulate Pac-Man games
- *rise.py* contains the custom implementation of the RISE approach and the implementation of the insertion metric (https://arxiv.org/abs/1806.07421)
- *sanity_checks.py* contains the code to apply the sanity checks (https://arxiv.org/abs/1810.03292) on the saliency maps for the Pac-Man model
- *space_invaders_main.py*  contains the code, which simulates Space Invaders games and creates saliency maps/insertion metrics for it 

