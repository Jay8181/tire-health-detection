import pickle

import keras
#Visualization
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
from flask import Flask, Response, request
from flask_cors import CORS

sns.set()

from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Sequential, load_model
#NN Model
from keras.preprocessing.image import ImageDataGenerator
#Evaluation
from sklearn.metrics import classification_report, confusion_matrix

best_model = load_model("model/cnn_tire_texture_model.hdf5")

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'GET':
        return "hey there asshole"

@app.route('/make-prediction',methods=['GET', 'POST'])
def prediction():
    import pickle

    import keras
    #Visualization
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    sns.set()

    #NN Model
    import keras.layers
    from keras.callbacks import ModelCheckpoint
    from keras.models import Sequential, load_model
    from keras.preprocessing.image import ImageDataGenerator
    global best_model
    if request.method == 'GET':
        train_generator = ImageDataGenerator(rotation_range = 360,
                                            width_shift_range = 0.05,
                                            height_shift_range = 0.05,
                                            shear_range = 0.05,
                                            zoom_range = 0.05,
                                            horizontal_flip = True,
                                            vertical_flip = True,
                                            brightness_range = [0.75, 1.25],
                                            rescale = 1./255,
                                            validation_split = 0.2)


        IMAGE_DIR = "C:\\Users\\kumar\\Downloads\\react-camera-component-with-hooks-new\\RobotJComp\\TireTextures\\"

        IMAGE_SIZE = (379, 379)
        BATCH_SIZE = 64
        SEED_NUMBER = 123

        gen_args = dict(target_size = IMAGE_SIZE,
                        color_mode = "grayscale",
                        batch_size = BATCH_SIZE,
                        class_mode = "binary",
                        classes = {"normal": 0, "cracked": 1},
                        seed = SEED_NUMBER)

        train_dataset = train_generator.flow_from_directory(
                                                directory = IMAGE_DIR + "training_data",
                                                subset = "training", shuffle = True, **gen_args)
        validation_dataset = train_generator.flow_from_directory(
                                                directory = IMAGE_DIR + "training_data",
                                                subset = "validation", shuffle = True, **gen_args)


        # In[ ]:


        test_generator = ImageDataGenerator(rescale = 1./255)
        test_dataset = test_generator.flow_from_directory(directory = IMAGE_DIR + "testing_data",
                                                        shuffle = False,
                                                        **gen_args)


        # **IMAGE VISUALIZATION**
        # 
        # We successfully load and apply on-the-fly data augmentation according to the specified parameters. Now, in this section, we visualize the image to make sure that it is loaded correctly.

        # In[ ]:


        mapping_class = {0: "normal", 1: "cracked"}
        mapping_class


        # In[ ]:


        def visualizeImageBatch(dataset, title):
            images, labels = next(iter(dataset))
            images = images.reshape(BATCH_SIZE, *IMAGE_SIZE)
            fig, axes = plt.subplots(8, 8, figsize=(16,16))

            for ax, img, label in zip(axes.flat, images, labels):
                ax.imshow(img, cmap = "gray")
                ax.axis("off")
                ax.set_title(mapping_class[label], size = 20)

            plt.tight_layout()
            fig.suptitle(title, size = 30, y = 1.05, fontweight = "bold")
            plt.show()
            
            return images


        # In[ ]:


        train_images = visualizeImageBatch(train_dataset,
                                        "FIRST BATCH OF THE TRAINING IMAGES\n(WITH DATA AUGMENTATION)")


        # In[ ]:


        test_images = visualizeImageBatch(test_dataset,
                                        "FIRST BATCH OF THE TEST IMAGES\n(WITHOUT DATA AUGMENTATION)")


        model_cnn = Sequential(
            [
                # First convolutional layer
                Conv2D(filters = 128,
                    kernel_size = 3,
                    strides = 2,
                    activation = "relu",
                    input_shape = IMAGE_SIZE + (1, )),
                
                # First pooling layer
                MaxPooling2D(pool_size = 2,
                            strides = 2),
                
                # Second convolutional layer
                Conv2D(filters = 64,
                    kernel_size = 3,
                    strides = 2,
                    activation = "relu"),
                
                # Second pooling layer
                MaxPooling2D(pool_size = 2,
                            strides = 2),
                
                # Third convolutional layer
                Conv2D(filters = 32,
                    kernel_size = 3,
                    strides = 2,
                    activation = "relu"),
                
                # Third pooling layer
                MaxPooling2D(pool_size = 2,
                            strides = 2),
                
                # Forth convolutional layer
                Conv2D(filters = 16,
                    kernel_size = 3,
                    strides = 2,
                    activation = "relu"),
                
                # Forth pooling layer
                MaxPooling2D(pool_size = 2,
                            strides = 2),
                
                # Flattening
                Flatten(),
                
                # Fully-connected layer
                Dense(128, activation = "relu"),
                Dropout(rate = 0.2),
                
                Dense(64, activation = "relu"),
                Dropout(rate = 0.2),
                
                Dense(1, activation = "sigmoid")
            ]
        )

        model_cnn.summary()


        model_cnn.compile(optimizer = 'adam',
                    loss = 'binary_crossentropy',
                    metrics = ['accuracy'])


        checkpoint = ModelCheckpoint('model/cnn_tire_texture_model.hdf5',
                                    verbose = 1,
                                    save_best_only = True,
                                    monitor='val_loss',
                                    mode='min')

        model_cnn.fit(train_dataset,
                            validation_data = validation_dataset,
                            batch_size = 16,
                            epochs = 1,
                            callbacks = [checkpoint],
                            verbose = 1)


        plt.subplots(figsize = (8, 6))
        sns.lineplot(data = pd.DataFrame(model_cnn.history.history,
                                        index = range(1, 1+len(model_cnn.history.epoch))))
        plt.title("TRAINING EVALUATION", fontweight = "bold", fontsize = 20)
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")

        plt.legend(labels = ['val loss', 'val accuracy', 'train loss', 'train accuracy'])
        plt.show()
        best_model = load_model("model/cnn_tire_texture_model.hdf5")


        # In[ ]:


        y_pred_prob = best_model.predict(test_dataset)
        THRESHOLD = 0.5
        y_pred_class = (y_pred_prob >= THRESHOLD).reshape(-1,)
        y_true_class = test_dataset.classes[test_dataset.index_array]

        pd.DataFrame(
            confusion_matrix(y_true_class, y_pred_class),
            index = [["Actual", "Actual"], ["normal", "cracked"]],
            columns = [["Predicted", "Predicted"], ["normal", "cracked"]],
        )


        # In[ ]:


        print(classification_report(y_true_class, y_pred_class, digits = 4))
        return 'fuck u unauthorized user'
    if request.method == 'POST':
        print(best_model)
        IMAGE_DIR = 'C:\\Users\\kumar\\Downloads\\react-camera-component-with-hooks-new\\static\\images'
        IMAGE_SIZE = (379, 379)
        BATCH_SIZE = 64
        SEED_NUMBER = 123
        gen_args = dict(target_size = IMAGE_SIZE,
                        color_mode = "grayscale",
                        batch_size = BATCH_SIZE,
                        class_mode = "binary",
                        classes = {"normal": 0, "cracked": 1},
                        seed = SEED_NUMBER)
        test_generator = ImageDataGenerator(rescale = 1./255)
        test_dataset = test_generator.flow_from_directory(directory = IMAGE_DIR, shuffle = False,**gen_args)
        result = best_model.predict(test_dataset)
        print(result)

        print('Cracked') if result[-1]>0.5 else print('Normal')
        r1 = random.randint(52, 1054)
        if r1%2==0:
            return Response(status=201)
        else:
            return Response(status=400)

if __name__ == '__main__':
    app.run()