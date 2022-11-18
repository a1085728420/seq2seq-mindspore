"""Create Dataset."""
import os
import argparse

from src.dataset.bi_data_loader import BiLingualDataLoader, TextDataLoader
from src.dataset.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description='Generate dataset file.')
parser.add_argument("--src_folder", type=str, required=False,
                    help="Raw corpus folder.")

parser.add_argument("--output_folder", type=str, required=False,
                    help="Dataset output path.")

if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    dicts = []
    train_src_file = "train.tok.clean.bpe.32000.en"
    train_tgt_file = "train.tok.clean.bpe.32000.fr"
    test_src_file = "newstest2014.en"
    test_tgt_file = "newstest2014.fr"

    vocab = args.src_folder + "/vocab.bpe.32000"
    bpe_codes = args.src_folder + "/bpe.32000"
    pad_vocab = 8
    tokenizer = Tokenizer(vocab, bpe_codes, src_en='en', tgt_fr='fr', vocab_pad=pad_vocab)

    train = BiLingualDataLoader(
        src_filepath=os.path.join(args.src_folder, train_src_file),
        tgt_filepath=os.path.join(args.src_folder, train_tgt_file),
        tokenizer=tokenizer,
        source_max_sen_len=51,
        target_max_sen_len=50,
        schema_address=args.output_folder + "/" + train_src_file + ".json"
    )
    print(f" | It's writing, please wait a moment.")
    train.write_to_mindrecord(
        path=os.path.join(
            args.output_folder,
            os.path.basename(train_src_file) + ".mindrecord"
        ),
        train_mode=True
    )
    test = TextDataLoader(
        src_filepath=os.path.join(args.src_folder, test_src_file),
        tokenizer=tokenizer,
        source_max_sen_len=None,
        schema_address=args.output_folder + "/" + test_src_file + ".json"
    )
    print(f" | It's writing, please wait a moment.")
    test.write_to_mindrecord(
        path=os.path.join(
            args.output_folder,
            os.path.basename(test_src_file) + ".mindrecord"
        ),
        train_mode=False
    )
    print(f" | Vocabulary size: {tokenizer.vocab_size}.")
    