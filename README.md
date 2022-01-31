# Benchmarking Perturbation-based Saliency Maps for Explaining Atari Agents

This repository contains the official implementation for the paper "Benchmarking Perturbation-based Saliency Maps for Explaining Atari Agents" (https://arxiv.org/abs/2101.07312). This paper implements and compares 5 different perturbation based XAI approaches:

 - Occlusion Sensitivity (https://arxiv.org/abs/1311.2901, https://github.com/sicara/tf-explain)
 - Noise Sensitivity (https://arxiv.org/abs/1711.00138, https://github.com/greydanus/visualize_atari)
 - RISE (https://arxiv.org/abs/1806.07421, https://github.com/eclique/RISE)
 - LIME (https://arxiv.org/abs/1602.04938, https://github.com/marcotcr/lime)
 - SARFA (https://arxiv.org/abs/1912.12191, https://github.com/nikaashpuri/sarfa-saliency)

In addition to creating saliency maps, the code is used to run sanity checks (https://arxiv.org/abs/1810.03292), an insertion metric (https://arxiv.org/abs/1806.07421).

# Subfolders
The repository is split into two subfolders: *`applications/affectnet`* and *`applications/atari`*

 - *`applications/atari`* contains 4 trained atari agents (Pac-Man, Breakout, Space Invaders, and Frostbite), the code to generate saliency maps for them, and the comparisons metrics (sanity checks and insertion metric).
 - *`applications/affectnet`* is deprecated. It contains an emotion recognition model and the code to generate saliency maps for it. 

Each subfolder contains an additional readme-file with more detailed information on how to use the code for the respective application.