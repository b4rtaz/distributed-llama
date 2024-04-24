import os
import struct
import sys
from typing import List
from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}")
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def export(self):
        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):

            # decode the token and light postprocessing
            t = self.sp_model.id_to_piece(i)
            s = self.sp_model.get_score(i)
            if i == self.bos_id:
                t = '\n<s>\n'
            elif i == self.eos_id:
                t = '\n</s>\n'
            t = t.replace('▁', ' ') # sentencepiece uses this character as whitespace
            b = t.encode('utf-8') # bytes of this token, utf-8 encoded

            tokens.append(b)
            scores.append(s)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)
 
        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        outputPath = 'dllama-' + os.path.basename(self.model_path).replace('.model', '.t')
        with open(outputPath, 'wb') as f:
            f.write(struct.pack('IIIiii',
                0x567123,
                self.n_words,
                max_token_length,
                self.bos_id,
                self.eos_id,
                self.pad_id))

            for bytes, score in zip(tokens, scores):
                print(f"{bytes.decode('utf-8')} {score}")
                f.write(struct.pack("fI", score, len(bytes)))
                f.write(bytes)
        print(f'Created {outputPath}')

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print('Invalid usage')
        exit(1)

    t = Tokenizer(sys.argv[1])
    t.export()
