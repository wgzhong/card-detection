import os
import json
import tvm
from tvm import relay

def save_model_to_json(mod, params, model_path='./'):
    model_dir = os.path.abspath(model_path)
    with open(os.path.join(model_dir, 'model.json'), 'w') as f_model_json:
        json.dump(tvm.save_json(mod), f_model_json)
        with open(os.path.join(model_dir, 'params.bin'), 'wb') as f_params:
            if params == None:
                params= {}
            f_params.write(relay.save_param_dict(params))

def load_model_from_json(model_path='./'):
    model_dir = os.path.abspath(model_path)
    try:
        with open(os.path.join(model_dir, 'model.json'), 'r') as f_model_json:
            print("*Info: Load saved JSON model from {}".format(os.path.join(model_dir, 'model.json')))
            mod = tvm.load_json(json.load(f_model_json))
            with open(os.path.join(model_dir, 'params.bin'), 'rb') as f_params:
                params = tvm.relay.load_param_dict(f_params.read())
        return mod, params
    except:
        return None, None