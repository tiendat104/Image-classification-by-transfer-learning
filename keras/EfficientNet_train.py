

from keras.models import Sequential, load_model, Input, Model
from keras.layers import Conv2D, Dropout, BatchNormalization, MaxPooling2D,Dense, Activation, Flatten
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications import Xception, MobileNet, InceptionV3, ResNet50, DenseNet121, VGG16, VGG19
import efficientnet.keras as efn
import os

list_classes = os.listdir("../data/train")
image_shape = (150,150,3)
num_classes = 6

batch_size = 16
train_datagen = ImageDataGenerator( rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator( rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory( "../data/train", target_size=(150,150),
    batch_size= batch_size, class_mode = 'categorical')

val_generator = val_datagen.flow_from_directory(
    "../data/val",target_size=(150,150),
    batch_size= batch_size, class_mode = 'categorical')
test_generator = test_datagen.flow_from_directory(
    "../data/test", target_size=(150,150),
    batch_size= batch_size, class_mode = 'categorical')

def transfer_learning_EfficientNet(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    model = efn.EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    for layer in model.layers[:20]:
        layer.trainable = False
    for layer in model.layers:
        if "BatchNormalization" in layer.__class__.__name__:
            layer.trainable = True
    x = GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5, name="top_dropout")(x)
    outputs = Dense(num_classes, activation="softmax", name="pred")(x)
    model = Model(inputs, outputs, name="mobileNet")
    return model

def train(epochs = 200):
    model = transfer_learning_EfficientNet(input_shape=image_shape, num_classes=num_classes)
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    current_checkpoint_subdir = os.listdir('checkpoint')
    new_checkpoint_subdir = os.path.join("checkpoint", str(len(current_checkpoint_subdir) + 1))
    os.makedirs(new_checkpoint_subdir, exist_ok=False)

    current_log_subdir = os.listdir("logs")
    new_log_subdir = os.path.join("logs", str(len(current_log_subdir) + 1))
    os.makedirs(new_log_subdir, exist_ok=False)

    tensorboard = TensorBoard(log_dir=new_log_subdir)
    # early_stopper = EarlyStopping(monitor = 'val_loss', mode= 'min', patience=10)
    early_stopper = EarlyStopping(monitor='val_accuracy', mode='max', patience=10)
    checkpointer = ModelCheckpoint(filepath=os.path.join(new_checkpoint_subdir, "{epoch:03d}-{val_accuracy:.3f}.hdf5"),
                                   monitor='val_accuracy', mode='max', verbose=1,
                                   save_best_only=True)

    model.fit_generator(
        train_generator,
        epochs = epochs,
        validation_data= val_generator,
        callbacks= [tensorboard, checkpointer]
    )

def test():
    model = load_model("checkpoint/2/008-0.778.hdf5")
    result = model.evaluate_generator(test_generator)
    print("loss: ", result[0])
    print("accuracy: ", result[1])

if __name__ == "__main__":
    train(epochs=10)
















