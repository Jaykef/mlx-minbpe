from mlx_minbpe import MLXBasicTokenizer

import os
import time
text = open("taylorswift.txt", "r", encoding="utf-8").read()
t0 = time.time()
for TokenizerClass, name in zip([MLXBasicTokenizer], ["mlx_basic"]):
    tokenizer = TokenizerClass()
    tokenizer.train(text, 512, verbose=True)
    
    prefix = os.path.join("models", name)
    tokenizer.save(prefix)
t1 = time.time()
print(f"Training took {t1 - t0:.2f} seconds")

