import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

        #  model.keras.layers.Dense

    def call(self, dec_hidden, enc_output):
        # dec_hidden shape == (batch_size, hidden size)
        # enc_output (batch_size, enc_len, enc_units)

        # hidden_with_time_axis shape == (batch_size, 1, dec_units)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, enc_len, attn_units)
        # 计算注意力权重值
        # score shape == (batch_size, enc_len, 1)
        score = self.V(tf.nn.tanh(
            self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention_weights (batch_size, enc_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # # 使用注意力权重*编码器输出作为返回值，将来会作为解码器的输入
        # enc_output (batch_size, enc_len, enc_units)
        # attention_weights (batch_size, enc_len, 1)
        context_vector = attention_weights * enc_output

        # context_vector shape after sum == (batch_size, enc_units)
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights