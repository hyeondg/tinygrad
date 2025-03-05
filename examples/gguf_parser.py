import logging
import os
import sys
from typing import Dict
logger = logging.getLogger("reader")
from gguf.gguf_reader import GGUFReader



def stat_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)
    print(reader)

    # List all key-value pairs in a columnized format
    print("Key-Value Pairs:") # noqa: NP100
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        print(f"{key:{max_key_length}} : {value}") # noqa: NP100
    print("----") # noqa: NP100



    # List all tensors
    print("Tensors:") # noqa: NP100
    tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
    print("-" * 80) # noqa: NP100
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)) # noqa: NP100

def parse_gguf_metadata(gguf) -> Dict[str, int]:
    import re
    reader = GGUFReader(gguf)
    ret = {}
    pattern = {
        r".*head_count$": 'n_heads',
        r".*head_count_kv$": 'n_kv_heads',
        r".*block_count$": 'n_layers',
        r".*embedding_length$": 'dim',
        r".*context_length$": 'max_context',
        r".*rope\.freq_base$": 'rope_theta',
        r".*layer_norm_rms_epsilon$": 'norm_eps',
    }
    for key, field in reader.fields.items():
        for k, v in pattern.items():
            if re.match(k, key):
                ret[v] = field.parts[field.data[0]][0].item()
                break
    for tensor in reader.tensors:
        if re.match(r"^token_embd.weight$", tensor.name):
            assert tensor.shape[0] == ret['dim']
            ret['vocab_size'] = tensor.shape[1].item()
        elif re.match(r".*ffn_down.weight$", tensor.name):
            ret['hidden_dim'] = tensor.shape[0].item()
            assert tensor.shape[1] == ret['dim']
        continue
    return ret



if __name__ == '__main__':
  path = '/home/hyeondg/models/qwen1_5-7b-chat-q4_0.gguf'
  stat_gguf_file(path)

  print(parse_gguf_metadata(path))

