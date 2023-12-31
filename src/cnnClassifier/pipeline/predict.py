import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from PIL import Image
import numpy as np
import tensorflow as tf

class PredictionPipeline :
    def __init__(self,filename):
        self.filename=filename

    def predict(self):
        # load model
        model=load_model(os.path.join("artifacts","training","model.h5"))


        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image=test_image/255.0
        test_image = np.expand_dims(test_image, axis = 0)
        print(model.predict(test_image))
        print("==============")
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)
        
        
        if result[0] == 0:
            prediction = 'Coccidiosis'
        
        elif result[0] == 1:
            prediction = 'Healthy'

        elif result[0] == 2:
            prediction = 'NCD'

        elif result[0] == 3:
            prediction = 'PCRCOCCI'

        elif result[0] == 4:
            prediction = 'PCR HEALTHY'

        elif result[0] == 5:
            prediction = 'PCR NCD'

        elif result[0] == 6:
            prediction = 'PCR SALMO'

        elif result[0] == 7:
            prediction = 'SALMO'

        return [{ "image" : prediction}]



