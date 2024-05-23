from mlx_minbpe import MLXRegexTokenizer

tokenizer = MLXRegexTokenizer()
tokenizer.register_special_tokens({"<|endoftext|>": 32768})
encoded = tokenizer.encode("<|endoftext|>hello world", allowed_special="all")
print(encoded)

import os
import time
text = open("taylorswift.txt", "r", encoding="utf-8").read()
t0 = time.time()
for TokenizerClass, name in zip([MLXRegexTokenizer], ["mlx_regex"]):
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")