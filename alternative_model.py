from tensorflow.keras import layers
import tensorflow.keras as keras
import tensorflow as tf
import network_ss
import os
import numpy as np
import matplotlib.pyplot as plt


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (61,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = tf.keras.backend.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        # calc
        loss = y_true * tf.keras.backend.log(y_pred) * weights
        loss = -tf.keras.backend.sum(loss, -1)
        return loss

    return loss
def multiclass_weighted_dice_loss(class_weights):
    """
    Weighted Dice loss.
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.keras.backend.constant(class_weights)

    def loss(y_true, y_pred) :
        """
        Compute weighted Dice loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, tf.keras.backend.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * tf.keras.backend.sum(numerator, axis=list(axis_to_reduce))

        denominator = (y_true + y_pred) * class_weights  # Broadcasting
        denominator = tf.keras.backend.sum(denominator, axis=list(axis_to_reduce))

        return 1 - numerator / denominator

    return loss


dir_np_chargrid_1h = "./data/np_chargrid_unscaled_top_1h/" # "./data/np_chargrids_1h/"
dir_np_gt_1h = "./data/np_label_unscaled_top_1h" #"./data/np_gt_1h/"

num_classes = 5
img_size = (192,192)
train_test_split = 0.2
list_filenames = [f for f in os.listdir(dir_np_chargrid_1h) if os.path.isfile(os.path.join(dir_np_chargrid_1h, f))]
trainset, testset = network_ss.get_train_test_sets(list_filenames[:100])

print("len train:", len(trainset), "len test:", len(testset))

chargrid, seg_label = network_ss.extract_batch(trainset, len(trainset), 2, 2, 2, 2) # extract batch means get dataset :)

model = get_model(img_size, num_classes)
#model.summary()

# model.load_weights("./output/alternative/model_ep_5.ckpt")
#
# res = model.predict(chargrid)
#
# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
# ax1.imshow(chargrid[0,:,:,:].argmax(axis=2))      #in_cg[0,:,:,:].argmax(axis=2))    np.reshape(in_cg[0], [256,128])
# ax2.imshow(seg_label[0,:,:,:].argmax(axis=2))
# ax3.imshow(res[0,:,:,:].argmax(axis=2))
# plt.show()

history_loss = []
history_acc = []
history_val_loss = []
history_val_acc = []

loss_weights = np.array([1,1,5,5,1]) #[1.4,24,14,14,25]
# classes: 0 = background, 1 = total, 2 = address , 3 = company ,4 = date
custom_loss = multiclass_weighted_dice_loss(loss_weights)
# custom_loss = weighted_categorical_crossentropy(loss_weights)
model.compile(optimizer="rmsprop", loss=custom_loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])
for epoch in range(0, 10):
    history = model.fit(x=chargrid, y=seg_label, batch_size=16, verbose=1, validation_split=0.25)
    model.save_weights(f"./output/alternative/model_ep_{epoch}.ckpt")

    results = model.predict(chargrid)
    history_loss.append(history.history["loss"])
    history_acc.append(history.history["categorical_accuracy"])
    history_val_loss.append(history.history["val_loss"])
    history_val_acc.append(history.history["val_categorical_accuracy"])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(results[0, :, :, :].argmax(axis=2))
    ax2.imshow(results[1, :, :, :].argmax(axis=2))
    ax3.imshow(results[2, :, :, :].argmax(axis=2))
    plt.savefig(f"./output/alternative/predictions_ep_{epoch}.png")
    plt.close()

network_ss.plot_loss(history_loss, history_val_loss, "Global Loss", "./output/alternative/global_loss.pdf")
network_ss.plot_loss(history_acc, history_val_acc, "Global acc", "./output/alternative/global_acc.pdf")