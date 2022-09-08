import os
import json
import tvm
from tvm import relay
import numpy as np

def update_fp16(graph):
    class DowncastMutator(relay.ExprMutator):
        def visit_call(self, call):
            new_op = self.visit(call.op)
            new_args = [self.visit(arg) for arg in call.args]
            if call.op == relay.op.get('cast'):
                return relay.cast(new_args[0], dtype="float16")
            elif call.op == relay.op.get('arange'):
                return relay.arange(new_args[0], new_args[1], new_args[2], dtype="float16")
            return relay.Call(new_op, new_args, call.attrs)               

        def visit_var(self, rvar):
            name_hint = rvar.name_hint
            dtype = rvar.type_annotation.dtype
            shape = rvar.type_annotation.shape
            assert(dtype=='float32')
            return relay.var(name_hint,shape=shape, dtype='float16')

        def visit_constant(self, const):
            value = const.data.asnumpy()
            dtype = const.data.dtype
            assert(dtype=='float32')
            return relay.const(value, dtype='float16')
    
                    
    def infer_type(node):
        mod = tvm.IRModule.from_expr(node)
        mod = relay.transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(node, relay.Function) else entry.body

    downcast_pass = DowncastMutator()
    func = downcast_pass.visit(graph)
    func = infer_type(func)
    return func

def save_weight(params, name, save="new"):
    tmp_name=name+"_weight"
    dir_path=os.getcwd()+"/"+tmp_name
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    new_params =  {}
    keys = []
    for key in sorted(params.keys()):
        # print(key)
        new_params[key] = params[key]
        keys.append(key)
    kernelpath = './'+tmp_name+'/weight_'
    for i in range(len(keys)):
        key = 'p' + str(i)
        if save == "old":
            key = keys[i]
        tmp = kernelpath + key + ".bin"
        with open(tmp, 'ab+') as f:
            params[key].asnumpy().tofile(f)
        f.close()

def get_calibration_dataset(mod, input_name):
    dataset = []
    input_shape = [int(x) for x in mod["main"].checked_type.arg_types[0].shape]
    for i in range(0):
        data = np.random.uniform(size=input_shape)
        dataset.append({input_name: data})
    return dataset
      
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

def run_auto_quantize_pass(mod, params, dataset):
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    # dataset = get_calibration_dataset(mod, "images")
    with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                skip_conv_layers=[],
                                weight_scale="max",
                                calibrate_chunk_by=num_cpu):
        mod = relay.quantize.quantize(mod, params, dataset)
    return mod

def export_so(mod, params, name, tar):
    if tar == "llvm":
        target = tvm.target.Target("llvm")
    else:
        target = tar
    with tvm.transform.PassContext(opt_level=2):
        compiled_lib = relay.build(mod, target, params=params)
    name = name+".so"
    if tar == "llvm":
        compiled_lib.export_library(name)
    else:
        compiled_lib.export_library(name, cc="/usr/bin/arm-linux-gnueabihf-g++")

def export_three_part(mod, params, name, tar):
    if tar == "llvm":
        target = tvm.target.Target("llvm")
    elif tar == "rasp3b":
        target = tvm.target.arm_cpu("rasp3b")
    else:
        print("not support")
        exit(0)
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build_module.build(mod, target, params=params)
    if not os.path.exists("./3p/"):
        os.makedirs("./3p/")
    libpath = "./3p/"+name+"_"+tar+".so"
    if tar == "llvm":
        lib.export_library(libpath)
    else:
        lib.export_library(libpath, cc="/usr/bin/arm-linux-gnueabihf-g++")
    graph_json_path = "./3p/"+name+"_"+tar+".json"
    with open(graph_json_path, 'w') as fo:
        fo.write(graph)
    param_path = "./3p/"+name+"_"+tar+".params"
    with open(param_path, 'wb') as fo:
        fo.write(relay.save_param_dict(params))
               