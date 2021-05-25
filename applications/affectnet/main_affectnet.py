from applications.affectnet.vgg_face_batch_norm import get_model, get_preprocess
import cv2
from applications.atari.explanation import explainer, create_saliency_image
import applications.atari.rise as rise
import timeit
import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import mark_boundaries
import keras
import csv

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
    use_cc_images = False
    plot = use_cc_images

    affectnet_dir = "W:\AffectNet"

    model = get_model()
    target_size = (224, 224)
    model.load_weights('vgg_face_batch_norm_e_74_l_1.18.h5')

    # The no softmax model is needed for the noise sensitivity approach since it uses the outputs of the logit units to
    # compute the saliency map
    no_softmax_model = keras.models.load_model('vgg_face_no_softmax.h5')

    if use_cc_images:
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

    else:
        images = []
        directory = os.path.join(affectnet_dir, "Manually_Annotated_Images")
        file_list = os.path.join(affectnet_dir, "Manually_Annotated_file_lists", "validation.csv")
        with open(file_list, newline='') as csvfile:
            validation_images = csv.reader(csvfile, delimiter=',', quotechar='|')
            index = 0
            for row in validation_images:
                images.append(row[0])
                index += 1
                if index > 1001:  # take two more images to be sure that we have enough for the 1000 checkpoint
                    break

    output_directory = "explanations/"
    my_explainer = explainer(model=model)
    my_no_softmax_explainer = explainer(model=no_softmax_model)
    occlusion_time = []
    lime_time = []
    greydanus_time = []
    greydanus_time_no_softmax = []
    rise_time = []
    scores = []
    index = 0
    for image_name in images:
        img = preprocess_image(os.path.join(directory, image_name))
        org_img = cv2.imread(os.path.join(directory, image_name))
        name = image_name.split(".")[0]

        tmp_scores = []

        # Occlusion Sensitivity
        start = timeit.default_timer()
        saliency_map = my_explainer.generate_occlusion_explanation(input=img, patch_size=20, use_softmax= False)
        stop = timeit.default_timer()
        occlusion_time.append(stop - start)
        if use_cc_images:
            create_saliency_image(saliency_map=saliency_map, image=org_img,
                              output_path=output_directory + "occlusion_explanation_" + name, cmap="viridis")

        if use_cc_images:
            insertion = rise.CausalMetric(model=model, mode='ins', step=img.shape[0],
                                    substrate_fn=rise.custom_black)
        else:
            insertion = rise.CausalMetric(model=model, mode='ins', step=img.shape[0] * 3,
                                      substrate_fn=rise.custom_black)
        score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="occl", plot=plot)
        print("Occlusion Sensitivity: " + name + " AUC: " + str(rise.auc(score)))
        tmp_scores.append(score)

        # Noise Sensitivity
        if use_cc_images:
            start = timeit.default_timer()
            saliency_map = my_explainer.generate_greydanus_explanation(input=img, r=20)
            stop = timeit.default_timer()
            greydanus_time.append(stop - start)

        start = timeit.default_timer()
        saliency_map_no__softmax = my_no_softmax_explainer.generate_greydanus_explanation(input=img, r=20)
        stop = timeit.default_timer()
        greydanus_time_no_softmax.append(stop - start)

        if use_cc_images:
            create_saliency_image(saliency_map=saliency_map, image=org_img,
                                  output_path=output_directory + "greydanus_explanation_v2_" + name, cmap="viridis")
            create_saliency_image(saliency_map=saliency_map_no__softmax, image=org_img,
                                  output_path=output_directory + "greydanus_explanation_nosoftmax_" + name, cmap="viridis")

            score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="noise")
            print("Noise Sensitivity: " + name + " AUC: " + str(rise.auc(score)))

        score = insertion.single_run(img_tensor=img, explanation=saliency_map_no__softmax, name=name,
                                     approach="noise_no_softmax", plot=plot)
        print("Noise Sensitivity no softmax: " + name + " AUC: " + str(rise.auc(score)))
        tmp_scores.append(score)

        # LIME
        start = timeit.default_timer()
        explanation, mask, ranked_mask = my_explainer.generate_lime_explanation(rgb_image=True, input=img.astype("double"),
                                                                                hide_img=False)
        stop = timeit.default_timer()
        lime_time.append(stop - start)
        if use_cc_images:
            plt.axis('off')
            plt.imshow(mark_boundaries(explanation/2 + 0.5, mask))
            plt.savefig(output_directory + "lime_explanation_" + name + ".png")

        score = insertion.single_run(img_tensor=img, explanation=ranked_mask, name=name, approach="lime", plot=plot)
        print("LIME: " + name + " AUC: " + str(rise.auc(score)))
        tmp_scores.append(score)

        # RISE
        start = timeit.default_timer()
        saliency_map = my_explainer.generate_rise_prediction(input=img, probability=0.4, use_softmax=False)
        stop = timeit.default_timer()
        rise_time.append(stop - start)
        if use_cc_images:
            create_saliency_image(saliency_map=saliency_map, image=org_img,
                              output_path=output_directory + "rise_explanation_" + name)
        score = insertion.single_run(img_tensor=img, explanation=saliency_map, name=name, approach="rise", plot=plot)
        print("RISE: " + name + " AUC: " + str(rise.auc(score)))
        tmp_scores.append(score)

        scores.append(tmp_scores)

        if index % 10 == 0 and index != 0:
            print("Saving progress...")
            np.save(file="figures/backup_affectnet/pred_" + str(index), arr=scores)
            np.save(file="figures/backup_times/lime_" + str(index), arr=lime_time)
            np.save(file="figures/backup_times/occlusion_" + str(index), arr=occlusion_time)
            np.save(file="figures/backup_times/greydanus_" + str(index), arr=greydanus_time_no_softmax)
            np.save(file="figures/backup_times/rise_" + str(index), arr=rise_time)
            scores = []
        index += 1

    print("Occlusion average Time: ")
    print(sum(occlusion_time) / len(occlusion_time))
    print("LIME average Time:")
    print(sum(lime_time) / len(lime_time))
    # print("Greydanus average Tims:")
    # print(sum(greydanus_time) / len(greydanus_time))
    print("Greydanus no softmax average Time:")
    print(sum(greydanus_time_no_softmax) / len(greydanus_time_no_softmax))
    print("RISE average Time:")
    print(sum(rise_time) / len(rise_time))
