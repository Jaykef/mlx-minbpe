# mlx-minbpe

mlx port of Karpath's (minbpe)[https://github.com/karpathy/minbpe]

## Usage
```python
from mlx_minbpe import MLXBasicTokenizer
tokenizer = MLXBasicTokenizer()
tokenizer.train(very_long_training_string, vocab_size=4096)
tokenizer.encode("hello world") # string -> tokens
tokenizer.decode([1000, 2000, 3000]) # tokens -> string
tokenizer.save("mymodel") # writes mymodel.model and mymodel.vocab
tokenizer.load("mymodel.model") # loads the model back, the vocab is just for vis
```

