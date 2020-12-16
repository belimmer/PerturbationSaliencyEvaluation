from applications.affectnet.vgg_face_batch_norm import get_model, get_preprocess
import keras

if __name__ == '__main__':
    model = get_model()
    target_size = (224, 224)
    model.load_weights('vgg_face_batch_norm_e_74_l_1.18.h5')
    idx_of_layer_to_change = -1
    model.layers[idx_of_layer_to_change].activation = keras.activations.linear
    model.save('vgg_face_no_softmax.h5')