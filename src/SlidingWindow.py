import re


from SimpleTokenizerV1 import SimpleTokenizerV1


with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text=f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words=sorted(set(preprocessed)) 
    vocab = {token: integer for integer,token in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab)
    enc_text=tokenizer.encode(raw_text)
    print(len(enc_text))
     