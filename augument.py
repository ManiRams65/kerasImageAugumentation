import weakref

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    rotation_range=180
    # horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory('./images/sample_part_of_material', target_size=(4000, 4000),
                                                    save_to_dir='./images/generate-image/')

print(train_generator)
i = 0
for batch in train_generator:
    i += 1
    if(i >= 50):
        break


# i = 0
# for batch in jf_datagen.flow_from_directory('images/train-image', target_size=(150, 150), save_to_dir='images/generate-image/'):
#     i += 1
#     if(i >= 10):
#         break
