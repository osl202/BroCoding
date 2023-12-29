import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers


batch_size = 32
img_height = 400#256
img_width = 400#256

train_data = tf.keras.utils.image_dataset_from_directory(
  "CCSN/CCSN_v2",
  labels='inferred',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_width, img_height),
  batch_size=batch_size)

val_data = tf.keras.utils.image_dataset_from_directory(
  "CCSN/CCSN_v2",
  labels='inferred',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_width, img_height),
  batch_size=batch_size)

class_names = train_data.class_names

# =============================================================================
# # =============================================================================
# # plt.figure(figsize=(10, 10))
# # for images, labels in val_data.take(1):
# #   for i in range(9):
# #     ax = plt.subplot(3, 3, i + 1)
# #     plt.imshow(images[i].numpy().astype("uint8"))
# #     plt.title(class_names[labels[i]])
# #     plt.axis("off")
# # =============================================================================
#  
# norm_layer = tf.keras.layers.Rescaling(1./255)
# norm_data = train_data.map(lambda x, y: (norm_layer(x), y))
# image_batch, labels_batch = next(iter(norm_data))
# first_image = image_batch[0]
# # Notice the pixel values are now in `[0,1]`.
# print(np.min(first_image), np.max(first_image))
# 
# AUTOTUNE = tf.data.AUTOTUNE
# 
# train_data = train_data.cache().prefetch(buffer_size=AUTOTUNE)
# val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)
# 
# num_classes = 11
# 
# model = Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# 
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(11))
# 
# model.compile(optimizer='adam', 
#              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              metrics=['accuracy'])
# history = model.fit(train_data, epochs=10, 
#                     validation_data=(val_data))
# 
# plt.figure(figsize=(8, 8))
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
# plt.legend(loc='lower right')
# 
# plt.savefig("output_report.png")
# 
# model.save("model.keras")
# =============================================================================
model = load_model('model.keras')
model.summary()
img = tf.keras.utils.load_img(
    'CCSN/CCSN_v2/As/1000012688.jpg', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
plt.figure(figsize=(10, 10))

img_tensor = tf.keras.preprocessing.image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
#Allows us to properly visualize our image by rescaling values in array
img_tensor /= 255.

ax = plt.subplot()
plt.imshow(img_tensor[0][::-1])
#plt.title(class_names[labels[0]])
plt.axis("off")



print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
#plt.imshow(test)