import logging
import os
import sys
from typing import Dict
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import gguf
from gguf.gguf_reader import GGUFReader

logger = logging.getLogger("reader")


class Tokenizer:
  pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

  def __init__(self, model_path: str):
    mergeable_ranks = load_tiktoken_bpe(model_path)
    self.num_base_tokens = len(mergeable_ranks)
    special_tokens = [
      "<|begin_of_text|>",
      "<|end_of_text|>",
      "<|reserved_special_token_0|>",
      "<|reserved_special_token_1|>",
      "<|reserved_special_token_2|>",
      "<|reserved_special_token_3|>",
      "<|start_header_id|>",
      "<|end_header_id|>",
      "<|reserved_special_token_4|>",
      "<|eot_id|>",
    ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
    self.special_tokens = {
      token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)
    }

    self.model = tiktoken.Encoding(
      name=model_path,
      pat_str=self.pat_str,
      mergeable_ranks=mergeable_ranks,
      special_tokens=self.special_tokens,
    )


  @property
  def bos_id(self):
    return self.special_tokens["<|begin_of_text|>"]

  @property
  def stop_tokens(self):
    return {
      self.special_tokens["<|end_of_text|>"],
      self.special_tokens["<|eot_id|>"],
    }

  def decode(self, toks):
    return self.model.decode([t for t in toks if t < self.num_base_tokens])

  def encode(self, text, allow_special=False):
    return self.model.encode(
      text,
      allowed_special="all" if allow_special else set(),
      disallowed_special=set(),
    )


def stat_gguf_file(gguf_file_path):
  """
  Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

  Parameters:
  - gguf_file_path: Path to the GGUF file.
  """

  reader = GGUFReader(gguf_file_path)
  print(reader)

  # List all key-value pairs in a columnized format
  print("Key-Value Pairs:")  # noqa: NP100
  max_key_length = max(len(key) for key in reader.fields.keys())
  for key, field in reader.fields.items():
    value = field.parts[field.data[0]]
    print(f"{key:{max_key_length}} : {value}")  # noqa: NP100
  print("----")  # noqa: NP100

  # List all tensors
  print("Tensors:")  # noqa: NP100
  tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
  print(
    tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")
  )  # noqa: NP100
  print("-" * 80)  # noqa: NP100
  for tensor in reader.tensors:
    shape_str = "x".join(map(str, tensor.shape))
    size_str = str(tensor.n_elements)
    quantization_str = tensor.tensor_type.name
    print(
      tensor_info_format.format(
        tensor.name, shape_str, size_str, quantization_str
      )
    )  # noqa: NP100


def parse_gguf_metadata(gguf) -> Dict[str, int]:
  import re

  reader = GGUFReader(gguf)
  ret = {}
  pattern = {
    r".*head_count$": "n_heads",
    r".*head_count_kv$": "n_kv_heads",
    r".*block_count$": "n_layers",
    r".*embedding_length$": "dim",
    r".*context_length$": "max_context",
    r".*rope\.freq_base$": "rope_theta",
    r".*layer_norm_rms_epsilon$": "norm_eps",
  }
  for key, field in reader.fields.items():
    for k, v in pattern.items():
      if re.match(k, key):
        ret[v] = field.parts[field.data[0]][0].item()
        break
  for tensor in reader.tensors:
    if re.match(r"^token_embd.weight$", tensor.name):
      assert tensor.shape[0] == ret["dim"]
      ret["vocab_size"] = tensor.shape[1].item()
    elif re.match(r".*ffn_down.weight$", tensor.name):
      ret["hidden_dim"] = tensor.shape[0].item()
      assert tensor.shape[1] == ret["dim"]
    continue
  return ret


def save_tokenizer_model(gguf, dst: str | None = None):
  """
  Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.
  Saves the tokenizer.model file to

  Parameters:
  - gguf_file_path: Path to the GGUF file.
  - dst: Path to saved `tokenizer.model` file
  """
  import base64
  reader = GGUFReader(gguf)
  encode = lambda t: base64.b64encode(t)

  print("Key-Value Pairs:")  # noqa: NP100
  with open(dst + "/tokenizer.model", "w") as f:
    max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
      if "tokenizer.ggml.tokens" in key:
        for idx, i in enumerate(field.data):
          value: "arraylike" = field.parts[i]
          bt: bytearray = bytes(list(value))
          if "PAD" in bt.decode():
            continue

          from html import unescape
          #print(f"{bytes(list(value))} {encode(bytes(list(value))).decode('utf-8')} {idx}")
          t = unescape(f"&{list(value)};")
          print(f"{idx} {value} {t}")
          #f.write(f"{encode(bt).decode()} {idx}\n")


  """

  for tensor in reader.tensors:
    if tensor.name != "token_embd.weight":
      continue
    else:
      print(tensor, tensor.shape)
      return
      with open(dst, 'r') as f:
        pass
  """


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="Run LLaMA in tinygrad", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model", type=str, default=None, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")
  parser.add_argument("--dst", type=str, default=None, help="Folder with the original weights to load, or single .index.json, .safetensors or .bin file")
  args = parser.parse_args()

  path = args.model
  #path = "/home/hyeondg/models/openelm-3b-instruct.Q4_0.gguf"
  # stat_gguf_file(path)
  string = "INC90LjQutGC0L4="
  import base64

  decoder = base64.b64decode(string)
  print(decoder.decode())

  save_tokenizer_model(path, args.dst)
  print('⽗'.encode())
