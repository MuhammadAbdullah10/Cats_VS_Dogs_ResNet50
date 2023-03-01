#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_SIZE = [150, 150]

train_path = '/Users/muhammadabdullah/OneDrive - Higher Education Commission/Download for MAC/dogs-vs-cat-small/train'
valid_path = '/Users/muhammadabdullah/OneDrive - Higher Education Commission/Download for MAC/dogs-vs-cat-small/validation'
test_path = '/Users/muhammadabdullah/OneDrive - Higher Education Commission/Download for MAC/dogs-vs-cat-small/test'

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (150, 150),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(valid_path,
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')

test_data = test_datagen.flow_from_directory(test_path,
                                            target_size = (150, 150),
                                            batch_size = 32,
                                            class_mode = 'categorical')





############# for resnet 50 ###############



resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(150,150,3))


output = restnet.layers[-1].output
output = Flatten()(output)
restnet = Model(restnet.input, outputs=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()



model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=[150,150,3]))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(2, activation='softmax'))
opt = optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()



# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('AccVal_acc')


test_loss, test_acc = model.evaluate(test_data)


# In[ ]:




