# Affectnet (Emotion Recognition Model)

This subfolder contains the code to create saliency maps for an emotion recognition model.

 - *`cc_free_images`* contains sample images, which can be used as input to the model
 - *`explanations`* contains saliency maps, generated for a subset of the images saved in *`cc_free_images`*
 - *`figures`* contains the resulting graphs of the insertion metric, produced for the images in *`cc_free_images`*
 - *`main_affectnet.py`* contains the code to load and preprocess the images and model. Besides producing saliency maps, the insertion metrics and average run-times are also computed here. The actual code for the explanation approaches and insertion metric can be found in *`applications/atari`*.
 - *`vgg_face_batch_norm.py`* contains the structure of the model
 - *`vgg_face_batch_norm_e_74_l_1.18.h5`* is the trained emotional recognition model
 -*`vgg_face_no_softmax.h5`* is the trained emotional recognition model without the softmax activation layer
# How to use:
Running *`main_affectnet.py`* will create saliency maps for the 8 images in *`cc_free_images`* and save the resulting images in the *`explanations`* folder. When creating the saliency maps the corresponding insertion metrics are computed and saved in the *`figures`* folder. After computing the saliency maps, the average runtime of all approaches is displayed in the terminal. To add new images, save them in the *`cc_free_images`* folder and add the name to the *images* array. 
