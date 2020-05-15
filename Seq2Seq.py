import tensorflow as tf
from Component import Encoder
from Component import Decoder
from Component import Attention


def load_word2vec(path):
    with open(path, 'r', encoding='utf-8') as f:
        word2vec = {}
        for line in f:
            temp = line.strip().split('\t')
            word = temp[0]
            vec = temp[1:]
            word2vec[word] = vec
    return word2vec


class SequenceToSequence(tf.keras.Model):
    def __init__(self, params):
        super(SequenceToSequence, self).__init__()
        self.embedding_matrix = load_word2vec(params['word2vec_path'])
        self.params = params
        print(params["batch_size"])
        self.encoder = Encoder.Encoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       units_num=params["enc_units"],
                                       batch_size=params["batch_size"])

        self.attention = Attention.BahdanauAttention(units=params["attn_units"])

        self.decoder = Decoder.Decoder(vocab_size=params["vocab_size"],
                                       embedding_dim=params["embed_size"],
                                       embedding_matrix=self.embedding_matrix,
                                       dec_units=params["dec_units"],
                                       batch_size=params["batch_size"])

    def call(self, dec_input, dec_hidden, enc_output, dec_target):
        predictions = []
        attentions = []

        context_vector, _ = self.attention(dec_hidden, enc_output)

        for t in range(1, dec_target.shape[1]):
            pred, dec_hidden = self.Decoder(dec_input,
                                            dec_hidden,
                                            enc_output,
                                            context_vector)

            context_vector, attn = self.attention(dec_hidden, enc_output)
            # using teacher forcing
            dec_input = tf.expand_dims(dec_target[:, t], 1)

            predictions.append(pred)
            attentions.append(attn)
            # tf.concat与tf.stack这两个函数作用类似，
            # 都是在某个维度上对矩阵(向量）进行拼接，
            # 不同点在于前者拼接后的矩阵维度不变，后者则会增加一个维度。
        return tf.stack(predictions, 1), dec_hidden
