# Cats_VS_Dogs_ResNet50
Classification of Cats vs Dogs Dataset of Kaggle using ResNet50 ( Transfer Learning )

## Dataset

The Kaggle [cats vs dogs](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset) dataset consists of **25,000** labeled images of **cats** and **dogs**. The images are of varying sizes and aspect ratios, and the goal is to train a **CNN** model to correctly classify new images of cats and dogs. The dataset is divided into two sets: a **training set** with **1,000** images and a **validation set** with **5,000** images.

## Transfer Learning

**Transfer learning** is a machine learning technique that involves using a **pre-trained** model for a new task. In transfer learning, the pre-trained model's weights and architecture are used as a starting point for a new model that is fine-tuned for the new task. This approach can be useful when the new dataset is small, as it allows the model to learn from the pre-trained model's knowledge and generalize better to new data.

## ResNet50

**ResNet50** is a **deep neural network** architecture that is commonly used for **image recognition** tasks. It has **50** layers and is pre-trained on the ImageNet dataset, which contains over a **million** images. **ResNet50** is used in this code as the pre-trained model for transfer learning.

## Code

The code begins by importing several Python libraries, including **TensorFlow Keras**, which is a popular deep learning library used for building and training **machine learning** models. The "**ImageDataGenerator**" class from **TensorFlow Keras** is used to generate batches of images for **training** and **validation**. The training set is preprocessed using the **ImageDataGenerator** by rescaling, shearing, zooming, and flipping the images. The **ResNet50** model is loaded and its **output layer** is replaced with a **new output layer** that has the same number of neurons as the number of classes in the dataset (2 in this case, for cats and dogs).

The new output layer is trained using the **training set** and validated using the **validation set**. The model's performance is evaluated using the accuracy and loss metrics. The **training** and **validation** accuracy and loss curves are plotted using the matplotlib library.

Overall, the code is an example of **transfer learning**, where a pre-trained **ResNet50** model is used as a starting point for a new machine learning model. By using a pre-trained model as a starting point, the new model can learn from the pre-existing knowledge that the pre-trained model has learned. This can help to speed up the training process and improve the performance of the new model.
