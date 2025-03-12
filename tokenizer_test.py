from tokenizers import Tokenizer

tokenizer = Tokenizer.from_pretrained("bert-base-cased")
encoded = tokenizer.encode("I can feel the magic, can you?")
print(encoded.ids)
print(encoded.tokens)
