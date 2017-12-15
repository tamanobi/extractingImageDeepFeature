# Getting Start
```
$ pip install -r requirements.txt
$ python features.py
```

# Extracting Image Deep Feature
```
curl -X POST http://127.0.0.1:8888/extract -H "Content-Type: multipart/form-data" -F "file=@your_image.jpg"
```

You can get like bellow response json:
```
{"features": [0.4153945744037628, 0.3994259238243103, ...]}
```
