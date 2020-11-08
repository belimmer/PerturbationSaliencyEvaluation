import os
from keras.layers import Dense, GlobalMaxPooling2D, Dropout, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, Input
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input

n_output = 8#config.corpus.value # if not config.label_indices else len(config.label_indices)

# interface function that returns the model.
def get_model():
    """Creates a new model based on vggface"""

    main_input = Input(shape=(None, None, 3))
    x = ZeroPadding2D((1, 1))(main_input)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ZeroPadding2D((1,1))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(128, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(512, (3, 3), activation="relu")(x)
    x = BatchNormalization()(x)

    x = GlobalMaxPooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)
    x = Dense(n_output, activation='softmax')(x)

    # This is the model we will train
    return Model(inputs=main_input, outputs=x)


# returns the preprocessing function for the respective input
def get_preprocess():
    return custom_preprocess


def custom_preprocess(x):
    return preprocess_input(x, mode='tf')


