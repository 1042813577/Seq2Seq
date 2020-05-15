import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, units_num, batch_size, bi=True):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = embedding_matrix
        self.units_num = units_num
        self.batch_size = batch_size
        self.bi = bi

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.units_num, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        if self.bi:
            self.units_num = units_num // 2
            self.bi_gru = tf.keras.layers.Bidirectional(self.gru)

        def call(self, enc_input):
            # (batch_size, enc_len, embedding_dim)
            enc_input_embedded = self.embedding(enc_input)

            initial_state = self.gru.get_initial_state(enc_input_embedded)

            if self.bi:
                # 是否使用双向GRU
                output, forward_state, backward_state = self.bi_gru(enc_input_embedded, initial_state=initial_state * 2)
                enc_hidden = tf.keras.layers.concatenate([forward_state, backward_state], axis=-1)

            else:
                # 单向GRU
                output, enc_hidden = self.gru(enc_input_embedded, initial_state=initial_state)

            return output, enc_hidden