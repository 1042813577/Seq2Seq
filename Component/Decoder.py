import tensorflow as tf


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_sz = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix],
                                                   trainable=False)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, dec_input, prev_dec_hidden, enc_output, context_vector):
        # 使用上次的隐藏层（第一次使用编码器隐藏层）、编码器输出计算注意力权重
        # enc_output shape == (batch_size, max_length, hidden_size)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        dec_input = self.embedding(dec_input)

        # 将上一循环的预测结果跟注意力权重值结合在一起作为本次的GRU网络输入
        # dec_input (batch_size, 1, embedding_dim + hidden_size)
        dec_input = tf.concat([tf.expand_dims(context_vector, 1), dec_input], axis=-1)
        # passing the concatenated vector to the GRU
        dec_output, dec_hidden = self.gru(dec_input)

        # dec_output shape == (batch_size * 1, hidden_size)
        dec_output = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        # pred shape == (batch_size, vocab)
        pred = self.fc(dec_output)
        return pred, dec_hidden
