import keras
from keras.models import model_from_json

json_file = open('/app/model.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('/app/model.h5')
print("Loaded model from disk", loaded_model)

from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model.signature_def_utils import predict_signature_def
from tensorflow.python.saved_model import tag_constants

model_version = '1'
export_dir = 'export/Servo/' + model_version

builder = builder.SavedModelBuilder(export_dir)

signature = predict_signature_def(
    inputs={"inputs": loaded_model.input}, outputs={"score": loaded_model.output})

from keras import backend as K

with K.get_session() as sess:
    # Save the meta graph and variables
    builder.add_meta_graph_and_variables(sess=sess, tags=[tag_constants.SERVING], signature_def_map={"serving_default": signature})
    builder.save()
