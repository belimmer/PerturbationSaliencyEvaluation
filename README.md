# Benchmarking Perturbation-based Saliency Maps for Explaining DeepReinforcement Learning Agents

This repository contains the implementation for the paper "Benchmarking Perturbation-based Saliency Maps for Explaining Atari Agents" (https://arxiv.org/abs/2101.07312). This paper implements and compares 4 different perturbation based XAI approaches:

 - Occlusion Sensitivity (https://arxiv.org/abs/1311.2901, https://github.com/sicara/tf-explain)
 - Noise Sensitivity (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari)
 - RISE (https://arxiv.org/abs/1806.07421, https://github.com/eclique/RISE)
 - LIME (https://arxiv.org/abs/1602.04938, https://github.com/marcotcr/lime)

In addition to creating saliency maps, the code is used to run sanity checks (https://arxiv.org/abs/1810.03292), an insertion metric (https://arxiv.org/abs/1806.07421) and a run-time analysis for the different saliency map approaches.

# Subfolders
The repository is split into two subfolders: *`applications/affectnet`* and *`applications/atari`*

 - *`applications/affectnet`* contains an emotion recognition model and the code to generate saliency maps for it 
 - *`applications/atari`* contains 4 trained atari agents (Pac-Man, Breakout, Space Invaders and Frostbite), the code to generate saliency maps for them, and the comparisons metrics (sanity checks and insertion metric)

