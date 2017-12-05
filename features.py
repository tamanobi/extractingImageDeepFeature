import io
import os
import sys
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
import numpy as np
from datetime import date
from PIL import Image
import tornado.escape
import tornado.ioloop
import tornado.web

model = VGG19(weights='imagenet')
#model = Model(inputs=model.input, outputs=model.get_layer('block4_pool').output)

class PostRegister(tornado.web.RequestHandler):
  def post(self):
    file_body = self.request.files['file'][0]['body']
    img = Image.open(io.BytesIO(file_body)).resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #block4_pool_features = model.predict(x)
    preds = model.predict(x)[0]
    print(preds)

application = tornado.web.Application([
  (r"/register", PostRegister),
])

if __name__ == "__main__":
  application.listen(8888)
  tornado.ioloop.IOLoop.instance().start()

