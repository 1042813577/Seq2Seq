import tensorflow as tf
import argparse
import training
from testing import test_and_save


def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--max_enc_len", default=200, help="编码器输入的最大长度", type=int)
    parser.add_argument("--max_dec_len", default=50, help="解码器输入的最大长度", type=int)
    parser.add_argument("--max_dec_steps", default=100,
                        help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30,
                        help="Minimum number of words of the predicted abstract", type=int)
    # parser.add_argument("--beam_size", default=3,
    #                     help="beam size for beam search decoding (must be equal to batch size in decode mode)",
    #                     type=int)
    parser.add_argument("--batch_size", default=16, help="batch size", type=int)
    parser.add_argument("--word2vec_path", default="data/w2v_embed.txt", type=str)
    parser.add_argument("--vocab_size", default=1059, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    # path
    # /ckpt/checkpoint/checkpoint
    parser.add_argument("--seq2seq_model_dir", default='ckpt/seq2seq', help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default='data/train_set.seg_x.txt',
                        help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default='data/train_set.seg_y.txt',
                        help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default='data/test_set.seg_x.txt',
                        help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default='data/vocab.txt', help="Vocab path")
    parser.add_argument("--word2vec_output", default='word2vec.txt', help="Vocab path")
    parser.add_argument("--test_save_dir", default='data/', help="test_save_dir")

    # others
    parser.add_argument("--steps_per_epoch", default=200, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=10, help="Number of examples to test", type=int)
    parser.add_argument("--epochs", default=5, help="train epochs", type=int)



    # mode
    parser.add_argument("--mode", default='train', help="training, eval or test options")
    parser.add_argument("--model", default='SequenceToSequence', help="which model to be slected")
    parser.add_argument("--pointer_gen", default=True, help="use pointer gen")
    parser.add_argument("--is_coverage", default=True, help="is_coverage")
    parser.add_argument("--greedy_decode", default=False, help="use greedy_decoder")
    parser.add_argument("--transformer", default=False, help="use transformer")

    args = parser.parse_args()
    params = vars(args)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        # 设置gpu内存自增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    if params["mode"] == "train":
        training.train(params)

    elif params["mode"] == "test":
        test_and_save(params)


if __name__ == '__main__':
    main()