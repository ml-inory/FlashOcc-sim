import argparse
import os, sys
import onnx
from onnx import helper, onnx_pb as onnx_proto, TensorProto, ValueInfoProto, version_converter


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create onnx model without bevpool')
    parser.add_argument('--input', help='input onnx file', default='bevdet_fp16_fuse_for_c_and_trt.onnx')
    parser.add_argument('--output', help='output onnx file', default='bevdet_ax.onnx')
    args = parser.parse_args()
    return args


def remove_origin_bev(graph):
    node_idx = -1
    del_node = None
    for i, node in enumerate(graph.node):
        if node.name == "bev_pool_v2_147":
            del_node = node
            node_idx = i
    
    graph.node.remove(del_node)

    rm_nodes = []
    input_nodes = ["ranks_depth", "ranks_feat", "ranks_bev", "interval_starts", "interval_lengths"]
    for node in graph.input:
        if node.name in input_nodes:
            rm_nodes.append(node)
    for node in rm_nodes:
        graph.input.remove(node)

    return node_idx


def create_ax_bev(graph):
    ax_bev = helper.make_node(op_type="AxBevPool", 
                              name="bev_pool_v2_147", 
                              inputs=["310", "316", "ranks_depth", "ranks_feat", "ranks_bev", "n_points"], 
                              outputs=["317"], 
                              domain="ai.onnx.contrib",
                              output_width=200,
                              output_height=200,
                              output_z=1)
    graph.node.append(ax_bev)

    ranks_depth = helper.make_tensor_value_info("ranks_depth", TensorProto.INT32, [185856])
    ranks_feat = helper.make_tensor_value_info("ranks_feat", TensorProto.INT32, [185856])
    ranks_bev = helper.make_tensor_value_info("ranks_bev", TensorProto.INT32, [185856])
    n_points = helper.make_tensor_value_info("n_points", TensorProto.INT32, [1])

    graph.input.extend([ranks_depth, ranks_feat, ranks_bev, n_points])

    
def main():
    args = parse_args()
    input_onnx = onnx.load(args.input)
    
    graph = input_onnx.graph
    remove_origin_bev(graph)
    create_ax_bev(graph)

    model = onnx.helper.make_model(
        graph,
        opset_imports=[
            onnx.helper.make_opsetid('', 11),
            onnx.helper.make_opsetid('ai.onnx.contrib', 1)
            ],
    )
    model = version_converter.convert_version(model, 11)
    # assert True == onnx.checker.check_model(input_onnx)
    onnx.save(model, args.output)
    print(f"Saved ax onnx to {args.output}")

if __name__ == '__main__':
    main()