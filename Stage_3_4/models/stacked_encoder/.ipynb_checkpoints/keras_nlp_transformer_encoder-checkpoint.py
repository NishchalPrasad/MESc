                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self._built = False
        self.supports_masking = True

    def _build(self, input_shape):
        # Create layers based on input shape.
        self._built = True
        feature_size = input_shape[-1]
        self._attention_head_size = int(feature_size // self.num_heads)
        self._multi_head_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self._attention_head_size,
            value_dim=self._attention_head_size,
            dropout=self.dropout,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )

        self._attention_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )
        self._feedforward_layernorm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
        )

        self._attention_dropout = keras.layers.Dropout(rate=self.dropout)

        self._intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dense = keras.layers.Dense(
            feature_size,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
        )
        self._output_dropout = keras.layers.Dropout(rate=self.dropout)

    def _add_and_norm(self, input1, input2, norm_layer):
        return norm_layer(input1 + input2)

    def _feed_forward(self, input):
        x = self._intermediate_dense(input)
        x = self._output_dense(x)
        return self._output_dropout(x)

    def call(self, inputs, padding_mask=None, attention_mask=None):
        """Forward pass of the TransformerEncoder.
        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, feature_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
                False means the certain certain is masked out.
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].
        Returns:
            A Tensor of the same shape as the `inputs`.
        """

        if not self._built:
            self._build(inputs.shape)

        mask = merge_padding_and_attention_mask(
            inputs,
            padding_mask,
            attention_mask,
        )

        # Self attention.
        attended = self._multi_head_attention_layer(
            inputs, inputs, inputs, attention_mask=mask
        )
        attended = self._attention_dropout(attended)
        attended = self._add_and_norm(
            inputs,
            attended,
            self._attention_layernorm,
        )
        # Feedforward.
        feed_forward_output = self._feed_forward(attended)
        return self._add_and_norm(
            attended, feed_forward_output, self._feedforward_layernorm
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
            }
        )
        return config