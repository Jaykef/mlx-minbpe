import mlx.core as mx
from mlx_base import Tokenizer, get_stats, merge

class MLXBasicTokenizer(Tokenizer):

    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # input text preprocessing
        text_bytes = text.encode("utf-8") # raw bytes
        ids = mx.array(list(text_bytes)) # MLX array of integers in range 0..255

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # int -> bytes
        for i in range(num_merges):
            # count up the number of times every consecutive pair appears
            stats = get_stats(ids.tolist())
            # find the pair with the highest count
            pair = mx.argmax(mx.array(list(stats.values())), keepdims=True).item()
            pair = list(stats.keys())[pair]
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = mx.array(merge(ids.tolist(), pair, idx))
            merges[pair] = idx
            # add(a: scalar | array, b: scalar | array, stream: None | Stream | Device = None)
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # save the merge
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges 
        self.vocab = vocab  

    def encode(self, text):
        # given a string text, return the token ids
        text_bytes = text.encode("utf-8") # raw bytes
        ids = mx.array(list(text_bytes)) # MLX array of integers in range 0..255
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids.tolist())
            pair_indices = mx.array(list(stats.keys()))
            pair_merge_indices = mx.array([self.merges.get(tuple(pair), float("inf")) for pair in pair_indices])
            pair_idx = mx.argmin(pair_merge_indices)
            pair = tuple(pair_indices[pair_idx].tolist())
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = mx.array(merge(ids.tolist(), pair, idx))
            
        return ids.tolist()

    def decode(self, ids):
        # given a list of token ids, return the string text
        tokens = [self.vocab[idx] for idx in ids]
        text_bytes = b''.join(tokens)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

