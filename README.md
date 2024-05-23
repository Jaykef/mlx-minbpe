# mlx-minbpe

mlx port of Karpath's !(minbpe)[https://github.com/karpathy/minbpe]

```
pip install requirements.txt
```

## Demo

https://github.com/Jaykef/mlx-minbpe/assets/11355002/52f54c27-1040-40b1-b9dd-5b022c38f7a4

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

## Train
```
python train.py
```

```python
# train.py
import os
import time
from mlx_minbpe import MLXBasicTokenizer, MLXRegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/taylorswift.txt", "r", encoding="utf-8").read()

# create a directory for models, so we don't pollute the current directory
os.makedirs("models", exist_ok=True)

t0 = time.time()
for TokenizerClass, name in zip([MLXBasicTokenizer, MLXRegexTokenizer], ["mxl_basic", "mlx_regex"]):

    # construct the Tokenizer object and kick off verbose training
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    # writes two files in the models directory: name.model, and name.vocab
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")
```
