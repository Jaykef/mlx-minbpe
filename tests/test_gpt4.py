import tiktoken
text = "<|endoftext|>hello world"
enc = tiktoken.get_encoding("cl100k_base")
print(enc.encode(text, allowed_special="all"))
# [100257, 15339, 1917]

# OURS
from mlx_minbpe import MLXGPT4Tokenizer
tokenizer = MLXGPT4Tokenizer()
print(tokenizer.encode(text, allowed_special="all"))