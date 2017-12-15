import io
import os
import sys
import json
from datetime import date
from PIL import Image
import requests

import tornado.escape
import tornado.ioloop
import tornado.web

import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense

tornado_port = 8888
database_name = 'danbooru'
gannoy_host = "localhost"
gannoy_port = 1323
gannoy_url = "http://{host}:{port}".format(host=gannoy_host, port=gannoy_port)

model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

def get_file_id(requested_file):
  file_id, ext = os.path.splitext(os.path.basename(requested_file['filename']))
  if not file_id.isdigit():
    raise tornado.web.HTTPError(status_code=400, log_message="Bad file name. It must be a number.")
  return int(file_id)

class FeatureExtractor:
  def __init__(self, file_body):
    self.feature = self.__extract(file_body)

  def __extract(self, file_body):
    img = Image.open(io.BytesIO(file_body)).resize((299, 299))
    x = image.img_to_array(img)
    if len(x.shape) > 2 and x.shape[2] == 4:
      x = x[:,:,:3]
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

  def to_dict(self):
    return {'features': self.feature.ravel().tolist()}

class Register(tornado.web.RequestHandler):
  def post(self):
    requested_file = self.request.files['file'][0]
    file_id = get_file_id(requested_file)

    feature = FeatureExtractor(requested_file['body'])
    r = requests.put('{gannoy_url}/databases/{database}/features/{key}'.format(gannoy_url=gannoy_url, database=database_name, key=file_id), json=feature.to_dict())
    self.write(str(r.status_code))

class Search(tornado.web.RequestHandler):
  def post(self):
    requested_file = self.request.files['file'][0]
    file_id = get_file_id(requested_file)

    feature = FeatureExtractor(requested_file['body'])
    r = requests.put('{gannoy_url}/databases/{database}/features/{key}'.format(gannoy_url=gannoy_url, database=database_name, key=file_id), json=feature.to_dict())

    if r.status_code == 200:
      search_result = requests.get('{gannoy_url}/search'.format(gannoy_url=gannoy_url), params={'database': database_name, 'key': file_id, 'limit': 10})
      self.write(search_result.text)
    else:
      self.write(str(r.status_code))

application = tornado.web.Application([
  (r"/register", Register),
  (r"/search", Search),
])

if __name__ == "__main__":
  application.listen(tornado_port)
  tornado.ioloop.IOLoop.instance().start()

