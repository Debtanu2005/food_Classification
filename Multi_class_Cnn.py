#!/usr/bin/env python
# coding: utf-8

# In[14]:


import zipfile
import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[16]:


zip_new= zipfile.ZipFile(r"C:\Users\aspir\Downloads\10_food_classes_all_data.zip")
zip_new.extractall()
zip_new.close()


# In[17]:


for dirpath , dirnames,filenames in os.walk("10_food_classes_all_data"):
    print(f"There are {len(dirnames)} classes and each classes have {len(filenames)} images in {dirpath}")


# In[18]:


train_dir="10_food_classes_all_data/train"
test_dir="10_food_classes_all_data/test"


# In[19]:


import pathlib
import numpy as np


# In[20]:


data_dir=pathlib.Path(train_dir)
class_names= np.array(sorted([item.name for item in data_dir.glob('*')]))


# In[21]:


print(class_names)


# In[53]:


target_image=os.path.join("10_food_classes_all_data/test/","ice_cream")
img=random.choice(os.listdir(target_image))
img_to= mpimg.imread(os.path.join(target_image,img))


# In[55]:


plt.imshow(img_to)
plt.axis("off")


# In[26]:


clas = os.path.join("10_food_classes_all_data/", "test")
classes = os.listdir(clas)


# In[5]:


import tensorflow as tf


# In[6]:


img_to.shape


# In[19]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[20]:


train_dir="10_food_classes_all_data/train/"
test_dir="10_food_classes_all_data/test/"

train_datagen=ImageDataGenerator(rescale=1./255,height_shift_range=0.3,width_shift_range=0.2,zoom_range=0.3,shear_range=0.2)
test_datagen=ImageDataGenerator(rescale=1./255,height_shift_range=0.3,width_shift_range=0.2,zoom_range=0.3,shear_range=0.2)

train_data_agumented= train_datagen.flow_from_directory(directory=train_dir,
                                             batch_size=18,
                                             target_size=(227,227),
                                             class_mode="categorical",
                                             seed=42)
valid_data_agumented= test_datagen.flow_from_directory(directory=test_dir,
                                             batch_size=18,
                                             target_size=(227,227),
                                             class_mode="categorical",
                                             seed=42)


train_datagen=ImageDataGenerator(rescale=1./255)
test_datagen=ImageDataGenerator(rescale=1./255)
train_data= train_datagen.flow_from_directory(directory=train_dir,
                                             batch_size=18,
                                             target_size=(227,227),
                                             class_mode="categorical",
                                             seed=42)
valid_data= test_datagen.flow_from_directory(directory=test_dir,
                                             batch_size=18,
                                             target_size=(227,227),
                                             class_mode="categorical",
                                             seed=42)


# In[21]:


images,labels= train_data.next()
images_agumented, labels_agumented= train_data_agumented.next()
import random
i= random.randint(0,18)


# In[22]:


print("Non Agumented")
plt.imshow(images[i])
plt.show()
print("Agumented")
plt.imshow(images_agumented[i])
plt.show()


# In[1]:


from tensorflow.keras  import Sequential
from tensorflow.keras  import layers


# In[24]:


from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(15, 9, strides=1, padding='valid', activation='relu', input_shape=(227, 227, 3)),
    layers.MaxPool2D(pool_size=(3, 3), strides=3),
    layers.Conv2D(68, 3, strides=1, activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    layers.Conv2D(108, 2, strides=1, padding='valid', activation='relu'),
    layers.MaxPool2D(pool_size=(3, 3), strides=2),
    layers.Flatten(),
    layers.Dense(216, activation='relu'),
    layers.Dense(192,activation='relu'),
    layers.Dense(10, activation='softmax')
])



# Print the model summary to verify the architecture
model.summary()




# In[81]:


model.compile( loss=tf.keras.losses.categorical_crossentropy,
             optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
             metrics=["accuracy"])
lr_scheduler= tf.keras.callbacks.EarlyStopping(  monitor='val_loss',
    min_delta=0.00001,
    patience=20,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
    start_from_epoch=0,)


# In[ ]:


history=model.fit(train_data_agumented,
         epochs=10,
         steps_per_epoch=len(train_data_agumented),
         validation_data=valid_data_agumented,
         validation_steps=len(valid_data_agumented)
                 callba)


# In[8]:


img=tf.keras.preprocessing.image.load_img("ham.jpeg")
image=tf.keras.preprocessing.image.img_to_array(img)
plt.imshow(image.astype(int))
image


# In[31]:


model.evaluate(train_data)


# In[10]:


def preprocess(img , target_shape):
    reshaped_image=tf.image.resize(img,target_shape)
    new_img=reshaped_image/255.
    
    return new_img


# In[11]:


buger=preprocess(image,(227,227))
buger.shape


# In[39]:


buger=tf.expand_dims(ice,0)


# In[13]:


plt.imshow(buger)


# In[58]:


def predict(image):
    pred=model.predict(image)
    print(f"prediction is:{class_names[tf.argmax(pred[0])]}")
    plt.imshow(tf.squeeze(image,0))


# In[59]:


pred=model.predict(buger)
pred[0]


# In[60]:


predict(buger)


# In[64]:


import pandas as pd
new_df= pd.DataFrame(history.history)


# In[65]:


new_df


# In[ ]:





# In[ ]:




