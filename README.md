# mlx-minbpe

mlx port of Karpath's (minbpe)[https://github.com/karpathy/minbpe]

```
pip install requirements.txt
```
## Quick Start
```python
from mlx_minbpe import MLXBasicTokenizer
tokenizer = MLXBasicTokenizer()
text = "aaabdaaabac"
tokenizer.train(text, 256 + 3) # 256 are the byte tokens, then do 3 merges
print(tokenizer.encode(text))
# [258, 100, 258, 97, 99]
print(tokenizer.decode([258, 100, 258, 97, 99]))
# aaabdaaabac
tokenizer.save("toy")
# writes two files: toy.model (for loading) and toy.vocab (for viewing)
```
## Usage
  
- For MLXBasicTokenizer: Minimal (byte-level) Byte Pair Encoding tokenizer.
  Algorithmically follows along the GPT tokenizer:
  https://github.com/openai/gpt-2/blob/master/src/encoder.py
  
  Does not handle the regular expression splitting pattern
  Does not handle any special tokens
      
  ```python
  from mlx_minbpe import MLXBasicTokenizer
  tokenizer = MLXBasicTokenizer()
  tokenizer.train(very_long_training_string, vocab_size=4096)
  tokenizer.encode("hello world") # string -> tokens
  tokenizer.decode([1000, 2000, 3000]) # tokens -> string
  tokenizer.save("mymodel") # writes mymodel.model and mymodel.vocab
  tokenizer.load("mymodel.model") # loads the model back, the vocab is just for vis
  ```

- For MLXRegexTokenizer: Unlike BasicTokenizer, it handles an optional regex splitting pattern and optional special tokens.
      
  ```python
  from mlx_minbpe import MLXRegexTokenizer
  tokenizer = MLXRegexTokenizer()
  tokenizer.train(very_long_training_string, vocab_size=32768)
  tokenizer.register_special_tokens({"<|endoftext|>": 32768})
  tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
  ```
