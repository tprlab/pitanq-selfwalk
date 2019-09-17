import tensorflow as tf
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from keras import backend as K
from keras.models import load_model
from keras.models import model_from_json


def load_keras_model(json_file, model_file):
    jf = open(json_file, 'r')
    loaded_model_json = jf.read()
    jf.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_file)
    return loaded_model


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph



model = load_keras_model('./model.json', './model.h5')
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, ".", "ktf_model.pb", as_text=False)