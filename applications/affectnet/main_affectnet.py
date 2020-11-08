from applications.affectnet.vgg_face_batch_norm import get_model, get_preprocess
import cv2
from applications.atari.explanation import explainer, create_saliency_image, create_lime_image
import rise
import timeit
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import keras

# define preprocessing function for the model
preprocess_function = get_preprocess()


def preprocess_image(image_path):
    """
    Reads and preprocesses images.

    Args:
        image_path: path from where to read the image

    Returns:
        image_processed: preprocessed version of the image
    """
    image_read = cv2.imread(image_path)
    image_rsz = cv2.resize(image_read, (target_size[0], target_size[1]))
    image_converted = cv2.cvtColor(image_rsz, cv2.COLOR_BGR2RGB)
    image_processed = preprocess_function(image_converted)
    return image_processed


if __name__ == '__main__':
    model = get_model()
    target_size = (224, 224)
    model.load_weights('vgg_face_batch_norm_e_74_l_1.18.h5')
    idx_of_layer_to_change = -1
    model.layers[idx_of_layer_to_change].activation = keras.activations.linear
    model.save('vgg_face_no_softmax.h5')

    no_softmax_model = keras.models.load_model('vgg_face_no_softmax.h5')

    directory = 'cc_free_images'
    images = ['neutral_cc_crop.png', 'happy_cc_crop.png', 'sad_cc_crop.png', 'surprise_cc_crop.png', 'fear_cc_crop.png',
              'disgust_cc_crop.png', 'anger_cc_crop.png', 'contempt_cc_crop.png']

    for image_name in images:
        image_path = os.path.join(directory,image_name)
        image = preprocess_image(image_path)
        my_input = np.expand_dims(image, axis=0)
        prediction_val = model.predict(my_input)
        prediction = np.argmax(prediction_val)
        print(image_name + ": " + str(prediction))
        pass
    pass

    output_directory = "explanations/"
    my_explainer = explainer(model=model)
    my_no_softmax_explainer = explainer(model=no_softmax_model)
    occlusion_time = []
    lime_time = []
    greydanus_time = []
    greydanus_time_no_softmax = []
    rise_time = []
    for image_name in images:
        img = preprocess_image(os.path.join(directory, image_name))
        org_img = cv2.imread(os.path.join(directory, image_name))
        name = image_name.split(".")[0]

        start = timeit.default_timer()
        saliency_map = my_explainer.generate_occlusion_explanation(input=img, patch_size=20)
        stop = timeit.default_timer()
        occlusion_time.append(stop - start)
        create_saliency_image(saliency_map=saliency_map, image=org_img,
                              output_path=output_directory + "occlusion_explanation_" + name, cmap="viridis")

        insertion = rise.CausalMetric(model=model, mode='ins', step=img.shape[0],
                                    substrate_fn=rise.custom_blur)
        score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="occl")
        print("Occlusion Sensitivity: " + name + " AUC: " + str(rise.auc(score)))

        start = timeit.default_timer()
        saliency_map = my_explainer.generate_greydanus_explanation(input=img, r=20)
        stop = timeit.default_timer()
        greydanus_time.append(stop - start)

        start = timeit.default_timer()
        saliency_map_no__softmax = my_no_softmax_explainer.generate_greydanus_explanation(input=img, r=20)
        stop = timeit.default_timer()
        greydanus_time_no_softmax.append(stop - start)

        create_saliency_image(saliency_map=saliency_map, image=org_img,
                              output_path=output_directory + "greydanus_explanation_v2_" + name, cmap="viridis")
        create_saliency_image(saliency_map=saliency_map_no__softmax, image=org_img,
                              output_path=output_directory + "greydanus_explanation_nosoftmax_" + name, cmap="viridis")

        score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="noise")
        print("Noise Sensitivity: " + name + " AUC: " + str(rise.auc(score)))

        score = insertion.single_run(img_tensor=img, explanation=saliency_map_no__softmax, name=name,
                                     approach="noise_no_softmax")
        print("Noise Sensitivity no softmax: " + name + " AUC: " + str(rise.auc(score)))

        start = timeit.default_timer()
        explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=True, input=img,
                                                                                hide_img=False)
        stop = timeit.default_timer()
        lime_time.append(stop - start)
        plt.axis('off')
        plt.imshow(mark_boundaries(explanation/2 + 0.5, mask))
        plt.savefig(output_directory + "lime_explanation_" + name + ".png")

        # insertion game works with positive explanations only, the code could probably be adjusted to cover
        # positives and negatives explanations
        explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=True, input=img,
                                                                                hide_img=False, positive_only=True)
        score = insertion.single_run(img_tensor=img, explanation=ranked_mask, name=name, approach="lime")
        print("LIME: " + name + " AUC: " + str(rise.auc(score)))

        start = timeit.default_timer()
        saliency_map = my_explainer.generate_rise_prediction(input=img, probability=0.4, use_softmax=False)
        stop = timeit.default_timer()
        rise_time.append(stop - start)
        create_saliency_image(saliency_map=saliency_map, image=org_img,
                              output_path=output_directory + "rise_explanation_" + name)
        score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="rise")
        print("RISE: " + name + " AUC: " + str(rise.auc(score)))

    print("Occlusion average Time: ")
    print(sum(occlusion_time) / len(occlusion_time))
    print("LIME average Time:")
    print(sum(lime_time) / len(lime_time))
    print("Greydanus average Tims:")
    print(sum(greydanus_time) / len(greydanus_time))
    print(sum(greydanus_time) / len(greydanus_time))
    print("Greydanus no softmax average Time:")
    print("RISE average Time:")
    print(sum(rise_time) / len(greydanus_time))
