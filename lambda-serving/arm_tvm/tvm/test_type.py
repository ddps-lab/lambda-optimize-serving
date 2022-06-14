import os
import argparse
import numpy as np
import tvm
from tvm import relay, autotvm
import tvm.contrib.graph_executor as runtime
from tvm.contrib import graph_runtime
import mxnet as mx

import tvm
from tvm import relay


def get_network(name, batch_size, dtype="float32", layout="NCHW"):
    """Get the symbol definition and random weight of a network"""
    input_name = "data"
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        import mxnet

        n_layer = int(name.split("t")[1])
        block = mxnet.gluon.model_zoo.vision.get_resnet(1, n_layer, pretrained=True)
        
        sym = block(mx.sym.var('data'))
        if isinstance(sym, tuple):
            sym = mx.sym.Group([*sym])

        # os.path.join(conf.MRT_MODEL_ROOT, name)
        # sym_path = sym_path if sym_path else "./data/%s.json"%name
        # prm_path = prm_path if prm_path else "./data/%s.params"%name
        with open(f'{name}.json', "w") as fout:
            fout.write(sym.tojson())
        block.collect_params().save(f'{name}.params')


        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "mobilenet_v2":
        import mxnet

        multiplier = 1
        block = mxnet.gluon.model_zoo.vision.get_mobilenet_v2(
            multiplier, pretrained=True
        )
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype=dtype
        )
        if layout == "NHWC":
            mod = convert_to_nhwc(mod)
        else:
            assert layout == "NCHW"
    elif name == "bert":
        import gluonnlp

        seq_length = 128

        # Instantiate a BERT classifier using GluonNLP
        model_name = "bert_12_768_12"
        dataset = "book_corpus_wiki_en_uncased"
        model, _ = gluonnlp.model.get_model(
            name=model_name,
            dataset_name=dataset,
            pretrained=True,
            use_pooler=True,
            use_decoder=False,
            use_classifier=False,
        )

        # Convert the MXNet model into TVM Relay format
        shape_dict = {
            "data0": (batch_size, seq_length),
            "data1": (batch_size, seq_length),
            "data2": (batch_size,),
        }
        mod, params = relay.frontend.from_mxnet(model, shape_dict)
        input_shape = (shape_dict["data0"], shape_dict["data1"], shape_dict["data2"])

        mod = tvm.relay.transform.FastMath()(mod)
        mod = tvm.relay.transform.EliminateCommonSubexpr()(mod)
        BindPass = tvm.relay.transform.function_pass(
            lambda fn, new_mod, ctx: tvm.relay.build_module.bind_params_by_name(
                fn, params
            ),
            opt_level=1,
        )
        mod = BindPass(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
        mod = tvm.relay.transform.CombineParallelBatchMatmul()(mod)
        mod = tvm.relay.transform.FoldConstant()(mod)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_name, input_shape, output_shape


def make_network_key(network_name, batch_size):
    return "model:%s-batchsize:%s" % (network_name, batch_size)


def convert_to_nhwc(mod):
    """Convert to NHWC layout"""
    desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    return mod

def export_results(graph,lib,params,model,batch_size):

    name = f"./{model}_{batch_size}/model"
    try:
        target_path = f"./{model}_{batch_size}"
        os.mkdir(target_path)
    except:
        pass
    graph_fn, lib_fn, params_fn = [name+ext for ext in ('.json','.tar','.params')]
    lib.export_library(lib_fn)
    with open(graph_fn, 'w') as f:
        f.write(graph)
    with open(params_fn, 'wb') as f:
        f.write(relay.save_param_dict(params))


def benchmark(model, batch_size, dtype, target, repeat):
    layout = "NCHW"
    mod, params, input_name, input_shape, output_shape = get_network(
        model, batch_size, dtype, layout
    )
    ctx = tvm.cpu()

    if model in ["bert"]:
        # Build module
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        lib.export_library(f"./{model}_{batch_size}.tar")
        # Make module 
        module = runtime.GraphModule(lib["default"](ctx))
        
        ##########################################################################
        # old version 
        ##########################################################################
        # with tvm.transform.PassContext(opt_level=3):
        #     #lib = relay.build(mod, target=target, params=params)
        #     graph, lib, params = relay.build(mod, target=target, params=params)
        # ctx = tvm.cpu()
        
        # export_results(graph,lib,params,model,batch_size)

        # module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
        # # module.load_params(params)     


        # Feed input data
        seq_length = input_shape[0][1]
        data = np.random.uniform(size=input_shape[0])
        token_types = np.random.uniform(size=input_shape[1])
        valid_length = np.array([seq_length] * batch_size)
        module.set_input(data0=data, data1=token_types, data2=valid_length)

    else:
        # Build module
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        lib.export_library(f"./{model}_{batch_size}.tar")
        #Make module 
        module = runtime.GraphModule(lib["default"](ctx))
        
        with tvm.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build(mod, target=target, params=params)
        ctx = tvm.cpu()
        print(type(params))
#         print(params)
        print(type(graph))
#         print(graph)
        print(type(lib))
#         print(lib)

        export_results(graph,lib,params,model,batch_size)

        module = tvm.contrib.graph_runtime.create(graph, lib, ctx)
        module.load_params(params)     

        # Feed input data
        data = np.random.uniform(size=input_shape)
        module.set_input(input_name, data)

    # Evaluate
    ftimer = module.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=repeat)
    return np.array(ftimer().results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["resnet50", "mobilenet_v2", "bert", "all"],
        default="all",
        help="The name of the model",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=core-avx2",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.model == "all":
        models = ["resnet50", "mobilenet_v2", "bert"]
    else:
        models = [args.model]
    batch_sizes = [args.batch_size]
    dtypes = [args.dtype]

    target = tvm.target.Target(args.target)

    # Benchmark
    result_messages = []
    for model in models:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                network_key = make_network_key(model, batch_size)
                print("Benchmark %s ..." % network_key)

                
                prof_res = benchmark(
                    model, batch_size, dtype, target, args.repeat
                )

                prof_res *= 1000  # convert to millisecond
                message = "%-18s %-12s %-19s (%s)" % (
                    model,
                    batch_size,
                    "%.2f ms" % np.mean(prof_res),
                    "%.2f ms" % np.std(prof_res),
                )
                result_messages.append(message)

    # Print result
    print("-------------------------------------------------------------")
    print(
        "%-18s %-12s %-20s"
        % ("Model Name", "Batch size", "Mean Inference Time (std dev)")
    )
    print("-------------------------------------------------------------")
    for line in result_messages:
        print(line)
    print("-------------------------------------------------------------")
