# inspect_onnx.py
import onnx
from collections import Counter

model_path = 'models/face_landmarks.onnx'
model = onnx.load(model_path)

print('=== Inputs ===')
for inp in model.graph.input:
    name = inp.name
    tt = inp.type.tensor_type
    # get element type and dims (if present)
    elem_type = tt.elem_type if tt.HasField('elem_type') else 'unknown'
    dims = []
    if tt.HasField('shape'):
        for d in tt.shape.dim:
            if d.HasField('dim_value'):
                dims.append(d.dim_value)
            elif d.HasField('dim_param'):
                dims.append(d.dim_param)
            else:
                dims.append('?')
    print(f'Input name: {name}  elem_type: {elem_type}  shape: {dims}')

print('\n=== Outputs ===')
for out in model.graph.output:
    name = out.name
    tt = out.type.tensor_type
    dims = []
    if tt.HasField('shape'):
        for d in tt.shape.dim:
            if d.HasField('dim_value'):
                dims.append(d.dim_value)
            elif d.HasField('dim_param'):
                dims.append(d.dim_param)
            else:
                dims.append('?')
    print(f'Output name: {name}  shape: {dims}')

print('\n=== Ops used (counts) ===')
ops = [node.op_type for node in model.graph.node]
print(Counter(ops))

print('\n=== First 60 nodes (op_type, name) ===')
for node in model.graph.node[:60]:
    print(node.op_type, node.name)

print('\n=== Initializers (constants) ===')
for init in model.graph.initializer:
    print(init.name, tuple(init.dims))

# Optional: try shape inference and print inferred value_info
try:
    inferred = onnx.shape_inference.infer_shapes(model)
    print('\n=== Inferred value_info (first 40) ===')
    for vi in inferred.graph.value_info[:40]:
        print(vi.name)
except Exception as e:
    print('\nShape inference failed:', e)
