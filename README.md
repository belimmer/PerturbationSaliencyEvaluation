# Benchmarking Perturbation-based Saliency Maps for Explaining DeepReinforcement Learning Agents

This repository contains the implementation for the paper "Benchmarking Perturbation-based Saliency Maps for Explaining DeepReinforcement Learning Agents". This paper implements and compares 4 different pertubation based XAI approaches:

 - Occlusion Sensitivity (https://arxiv.org/abs/1311.2901)
 - Noise Sensitivity (https://arxiv.org/abs/1711.00138)
 - RISE (https://arxiv.org/abs/1806.07421)
 - LIME (https://arxiv.org/abs/1602.04938)

In addition to creating saliency maps, sanity checks (https://arxiv.org/abs/1810.03292), an insertion metric (https://arxiv.org/abs/1806.07421) and a run-time analysis were deployed to compare the different approaches.

# Subfolders
The repository is split into two subfolders: *applications/affectnet* and *applications/atari*

 - applications/affectnet contains an emotion recognition model and the code to generate saliency maps for it 
 - applications/atari contains 3 trained atari agents (Pac-Man, Breakout and Spaceinvaders), the code to generate saliency maps for them and the comparisons metrics (sanity checks and insertion metric)

