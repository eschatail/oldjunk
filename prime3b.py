import keras.backend as K
from keras.layers import Layer, Dense, Input, Conv2DTranspose, Activation, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
import torch
import torch.nn as nn
import tensorflow as tf
from keras.models import Sequential
import random
import numpy as np
from keras.optimizers import Adam



class FoxNet(tf,keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FoxNet, self).__init__(**kwargs)
        
        # Add layers
        self.htm_layer = HTMLayer(n_inputs=256, n_columns=2048, n_cells_per_column=32)
        self.transformer = RecursiveTransformerArray(units=64, num_layers=2, max_length=100)
        self.anlsm_layer = ANLsmLayer(n_reservoir=1000, spectral_radius=0.9, sparsity=0.5, alpha=1.0)
        self.ds_nasrl_layer = DS_NASRL(num_conv_layers=3, num_dense_layers=2, in_shape=(32, 32, 1), out_shape=(10,))
        self.tdssdslayer = TernaryDenseSparseDeepSpikingLayer(num_inputs=1000, num_outputs=1000, num_layers=42)
        self.adv_layer = AdversarialLayer(num_neurons=128)
        self.gat_layer = GATLayer(num_heads=8, d_model=512, dff=2048, max_seq_len=100, spiking_threshold=0.42)
        self.som_layer = SOM(num_neurons=10000, learning_rate=0.42)
        
        # Define trainable variables
        self.kernel = self.add_weight(shape=(3, 3, None, 64), initializer='glorot_uniform', dtype='int8')
        self.kernel.assign(K.abs(self.kernel))
    
        self.bias = self.add_weight(shape=(64,), initializer='zeros', dtype='int8')
        self.bias.assign(K.abs(self.bias))
        
        self.batch_norm = BatchNormalization(trainable=True)
        self.bnn = tfp.layers.DenseFlipout(640000)
        self.dropout = tf.keras.layers.Dropout(0.2) 
        self.rnn = tf.keras.layers.LSTM(64, return_sequences=True)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape[:-1]) + [6400, 6400, 64])

    def build(self, input_shape):
        self.padding = ((0, 0), (0, 0))
        super(FoxNet, self).build(input_shape)

    def call(self, inputs):
        inputs = K.cast(inputs, 'int8')
        
        for i in range(2):
            if self.padding[i][0] > 0 or self.padding[i][1] > 0:
                inputs = K.spatial_2d_padding(inputs, self.padding[i])
        
        htm_output = self.htm_layer(inputs)
        transformed_output = self.transformer(htm_output)
        anlsm_output = self.anlsm_layer(transformed_output)
        ds_nasrl_output = self.ds_nasrl_layer(anlsm_output)
        spiking_output = self.tdssdslayer(ds_nasrl_output)
        adv_output = self.adv_layer(spiking_output)
        gat_output = self.gat_layer(adv_output)
        som_output = self.som_layer(gat_output)
        
        synaptic_input = K.conv2d(som_output, self.kernel, strides=(1,1), padding='valid', data_format=None)
        
        if self.use_bias:
            synaptic_input = K.bias_add(synaptic_input, self.bias, data_format=None)

        spikes = K.round(synaptic_input)

        if self.activation is not None:
            spikes = self.activation(spikes)

        if self.normalization:
            spikes = self.batch_norm(spikes)

        dropout_output = self.dropout(spikes, training=True)
        bnn_output = self.bnn(dropout_output)
        
        output = tfp.distributions.MultivariateNormalDiag(loc=bnn_output, scale_diag=0.2).sample()
        recursive_output = self.rnn(output)
        output = K.cast(recursive_output, 'int8')

    def connect_networks(input_networks, output_networks):
    # Connect input networks to output networks
        for i in range(len(input_networks)):
            input_networks[i].output = output_networks[i]
            output_networks[i].input_network = input_networks[i]

    def set_input_size(network, input_size):
        network.input_size = input_size
    
    def unify_networks(input_networks, output_networks):
    # Connect input and output networks
        connect_networks(input_networks, output_networks)
    
    # Set input sizes for input networks
    for input_network in input_networks:
        set_input_size(input_network, random.randint(42000, 64000))

    # Predict for input and output networks
    for input_network, output_network in zip(input_networks, output_networks):
        input_network.predict(X)
        output_network.predict(X)
    
    # Create new instances of each network
    input_networks = [TernaryDense(), TernaryDenseSparseDeepSpikingLayer(), ConvolutionalNetwork(),
                      FeedForwardNetwork(), RecurrentNetwork(), LSTMNetwork(), GRUNetwork(),
                      Autoencoder(), RBM(), GAN(), Generator(), Discriminator(), SOM(), AdversarialLayer(),
                      GATLayer(), DS_NASRL(), ANLsmLayer(), FoxNet(), MySelf()]

    output_networks = [TernaryDense(), TernaryDenseSparseDeepSpikingLayer(), ConvolutionalNetwork(),
                       FeedForwardNetwork(), RecurrentNetwork(), LSTMNetwork(), GRUNetwork(),
                       Autoencoder(), RBM(), GAN(), Generator(), Discriminator(), SOM(), AdversarialLayer(),
                       GATLayer(), DS_NASRL(), ANLsmLayer(), FoxNet(), MySelf()]

    # Connect input and output networks
    connect_networks(input_networks, output_networks)
    
    # Set input sizes for input networks
    for input_network in input_networks:
        set_input_size(input_network, random.randint(42000, 64000))

    # Predict for input and output networks
    for input_network, output_network in zip(input_networks, output_networks):
        input_network.predict(X)
        output_network.predict(X)

        return input_networks, output_networks

    def unify_recursive(input_networks, output_networks, input_networks_2=None, output_networks_2=None, input_networks_3=None,
                        output_networks_3=None, input_networks_4=None, output_networks_4=None, input_networks_5=None, output_networks_5=None,
                        input_networks_6=None, output_networks_6=None, input_networks_7=None, output_networks_7=None, input_networks_8=None,
                        output_networks_8=None, input_networks_9=None, output_networks_9=None, input_networks_10=None, output_networks_10=None,
                        input_networks_11=None, output_networks_11=None, input_networks_12=None, output_networks_12=None, input_networks_13=None,
                        output_networks_13=None, input_networks_14=None, output_networks_14=None, input_networks_15=None, output_networks_15=None,
                        input_networks_16=None, output_networks_16=None, input_networks_17=None, output_networks_17=None, input_networks_18=None,
                        output_networks_18=None, input_networks_19=None, output_networks_19=None, input_networks_20=None, output_networks_20=None):

        all_input_networks = [input_networks, input_networks_2, input_networks_3, input_networks_4, input_networks_5, input_networks_6,
                                  input_networks_7, input_networks_8, input_networks_9, input_networks_10, input_networks_11, input_networks_12,
                                  input_networks_13, input_networks_14, input_networks_15, input_networks_16, input_networks_17, input_networks_18,
                                  input_networks_19, input_networks_20]
    all_output_networks = [output_networks, output_networks_2, output_networks_3, output_networks_4, output_networks_5, output_networks_6,
                                   output_networks_7, output_networks_8, output_networks_9, output_networks_10, output_networks_11, output_networks_12,
                                   output_networks_13, output_networks_14, output_networks_15, output_networks_16, output_networks_17, output_networks_18,
                                   output_networks_19, output_networks_20]
    # Connect input and output networks
    connect_networks(input_networks, output_networks)
    input_networks_2 = [input_network_2_1, input_network_2_2]
    output_networks_2 = [output_network_2_1, output_network_2_2]

    input_networks_3 = [input_network_3_1]
    output_networks_3 = [output_network_3_1]
    
    # Set input sizes for input networks
    for input_network in input_networks:
        set_input_size(input_network, random.randint(42000, 64000))

    # Predict for input and output networks
    for input_network, output_network in zip(input_networks, output_networks):
        input_network.predict(X)
        output_network.predict(X)

    # Call each output network with its own input of X
    for i, output_network in enumerate(output_networks):
        current_input = X
        next_input = None

        while next_input is not current_input:
            # Update previous action and best failed count for each iteration
            output_network.best_failed_count = output_network.best_failed_count + 1 if next_input is None else 0
            output_network.previous_action = current_input if i == 0 else output_networks[i-1](current_input)

            next_input = output_network(current_input)
            current_input = next_input

    # Return the updated input and output networks
    return input_networks, output_networks
        
    for i in range(2):
            if self.padding[i][0] > 0 or self.padding[i][1] > 0:
                output = output[:, self.padding[i][0]:-self.padding[i][1], self.padding[i][2]:-self.padding[i][3], :]
        
            if self.return_sequences:
                return output
            else:
                return K.expand_dims(K.sum(output, axis=1), axis=-2)


    def get_last_output(self, recursion_count):
        if recursion_count > 0:
            return self.output

    # Recursive unifier with greedy epsilon emulation
    self.best_failed_count = 0 if recursion_count == 0 else self.best_failed_count
    self.previous_action = None if recursion_count == 0 else self.previous_action

    if self.previous_action is not None:
        if self.best_failed_count >= 2 and self.best_failed_count > 0 or self.padding[i][1] > 0:
            self.output = self.output[:, self.padding[i][0]:-self.padding[i][1], self.padding[i][2]:-self.padding[i][3], :]
    
    if self.return_sequences:
        return self.output
    else:
        return K.expand_dims(K.sum(self.output, axis=1), axis=-2)


        if self.previous_action is not None and self.best_failed_count >= 2:
            new_action = self.generate_new_action(output[0], self.previous_action[0])
            output = self.modify_action(output, new_action)
            self.previous_action = new_action
        
    # If best_failed_count is < 2, increment the counter with probability p
        else:
            p = np.random.uniform(0.0, 1.0)
        if p > 0.5:
            self.best_failed_count += 1

    return output
    
    def get_last_output(self, recursion_count):
        if recursion_count > 0:
            return self.output

    # Recursive unifier with random failure emulation
    self.best_failed_count = 0 if recursion_count == 0 else self.best_failed_count
    self.previous_action = None if recursion_count == 0 else self.previous_action

    if self.previous_action is not None and self.best_failed_count >= 2:
        new_action = self.generate_new_action(self.output[0], self.previous_action[0])
        self.output = self.modify_action(self.output, new_action)
        self.previous_action = new_action

    else:
        p = np.random.uniform(0.0, 1.0)
        if p > 0.5:
            self.best_failed_count += 1

    if self.return_sequences:
        return self.output
    else:
        return K.expand_dims(K.sum(self.output, axis=1), axis=-2)
    

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'timestep': self.timestep,
            'strides': self.strides,
            'activation': self.activation,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'return_sequences': self.return_sequences,
            'normalization': self.normalization,
            'train_normalization': self.train_normalization,
            'max_length': self.max_length,
            'n_inputs': self.n_inputs,
            'n_columns': self.n_columns,
            'n_cells_per_column': self.n_cells_per_column,
            'initial_permanence_threshold': self.initial_permanence_threshold,
            'connected_permanence_threshold': self.connected_permanence_threshold,
            'min_overlap': self.min_overlap,
            'lr': self.lr,
            'n_reservoir': self.n_reservoir,
            'spectral_radius': self.spectral_radius,
            'sparsity': self.sparsity,
            'alpha': self.alpha,
            'num_conv_layers': self.num_conv_layers,
            'num_dense_layers': self.num_dense_layers,
            'in_shape': self.in_shape,
            'out_shape': self.out_shape,
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'num_spiking_layers': self.num_spiking_layers,
            'magnitude_weighting': self.magnitude_weighting,
            'cosine_activation': self.cosine_activation,
            'num_neurons': self.num_neurons,
            'spiking_threshold': self.spiking_threshold,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'num_som_neurons': self.num_som_neurons,
            'som_sigma': self.som_sigma,
            'som_learning_rate': self.som_learning_rate
        }
        base_config = super(FoxNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvolutionalNetwork(tf.keras.layers.Layer):
    def __init__(self, num_classes=420):
        super().__init__()

        self.conv1 = keras.layers.Conv2D(filters=3200, kernel_size=(300, 300), activation='relu')
        self.maxpool1 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.conv2 = keras.layers.Conv2D(filters=640, kernel_size=(300, 300), activation='relu')
        self.maxpool2 = keras.layers.MaxPool2D(pool_size=(2, 2))
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(units=num_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        return self.dense(x)

class RecurrentNetwork(tf.keras.layers.Layer):
    def __init__(self, num_classes=42):
        super().__init__()

        self.gru = keras.layers.GRU(units=255, return_sequences=True)
        self.dropout = keras.layers.Dropout(0.1)
        self.fc = keras.layers.TimeDistributed(keras.layers.Dense(units=num_classes))

    def call(self, x):
        out = self.gru(x)
        out = self.dropout(out)
        return self.fc(out)

class LSTMNetwork(tf.keras.layers.Layer):
    def __init__(self, num_classes=6400):
        super().__init__()

        self.lstm1 = keras.layers.LSTM(units=2550, return_sequences=True)
        self.dropout = keras.layers.Dropout(0.2)
        self.lstm2 = keras.layers.LSTM(units=1280)
        self.flatten = keras.layers.Flatten()
        self.fc = keras.layers.Dense(units=num_classes)
    
    def call(self, x):
        out = self.lstm1(x)
        out = self.dropout(out)
        out = self.lstm2(out)
        out = self.flatten(out)
        return self.fc(out)

class GRUNetwork(tf.keras.layers.Layer):
    def __init__(self, num_classes=42):
        super().__init__()

        self.gru1 = keras.layers.GRU(units = 255, return_sequences=True)
        self.dropout = keras.layers.Dropout(0.1)
        self.gru2 = keras.layers.GRU(units = 428)
        self.fc = keras.layers.Dense(units=num_classes)

    def call(self, x):
        out = self.gru1(x)
        out = self.dropout(out)
        out = self.gru2(out)
        return self.fc(out)

class Autoencoder(tf.keras.layers.Layer):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.encoder = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=latent_dim, activation='relu')
        ])
        self.decoder = keras.Sequential([
            keras.layers.Dense(units=128, activation='relu'),
            keras.layers.Dense(units=256, activation='relu'),
            keras.layers.Dense(units=784, activation='sigmoid'),
            keras.layers.Reshape((28, 28, 1))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class RBM(tf.keras.layers.Layer):
    def __init__(self, num_hidden=128):
        super().__init__()

        self.hidden_layer = keras.layers.Dense(units=num_hidden, activation='sigmoid')
        self.visible_layer = keras.layers.Dense(units=784, activation='sigmoid')
        self.k = 10

    def call(self, x):
        visible = x
        for i in range(self.k):
            hidden_probs = self.hidden_layer(visible)
            hidden = tf.cast(tf.random.poisson(hidden_probs) >= 0.5, tf.float32)
            visible_probs = self.visible_layer(hidden)
            visible = tf.cast(tf.random.poisson(visible_probs) >= 0.5, tf.float32)
        return visible

class Generator(tf.keras.models.Model):
    def __init__(self):
        super().__init__()

        self.layer1 = keras.layers.Dense(units=1200, activation='relu')
        self.layer2 = keras.layers.Dense(units=784, activation='sigmoid')

    def call(self, x):
        x = self.layer1(x)
        return self.layer2(x)

class Discriminator(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.layer1 = keras.layers.Dense(units=128, activation='relu')
        self.dropout = keras.layers.Dropout(0.2)
        self.layer2 = keras.layers.Dense(units=1)

    def call(self, x):
        x = self.layer1(x)
        x = self.dropout(x)
        return self.layer2(x)

class GAN(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def call(self, x):
        return self.generator(x)

    def generator_loss(self, fake_preds):
        return keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake_preds), fake_preds)
    
    def discriminator_loss(self, real_preds, fake_preds):
        real_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(real_preds), real_preds)
        fake_loss = keras.losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(fake_preds), fake_preds)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal(shape=(batch_size, 1000))

        with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
            fake_images = self.generator(noise)
            real_preds = self.discriminator(real_images)
            fake_preds = self.discriminator(fake_images)
            d_loss = self.discriminator_loss(real_preds, fake_preds)
            g_loss = self.generator_loss(fake_preds)

        d_gradients = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        g_gradients = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return {'d_loss': d_loss.numpy().mean(), 'g_loss': g_loss.numpy().mean()}

class SOM(tf.keras.layers.Layer):
    def __init__(self, num_neurons=10000, sigma=None, learning_rate=0.42):
        super().__init__()

        self.num_neurons = num_neurons
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = self.add_weight(shape=(num_neurons, 784),
                                        initializer='random_normal',
                                        trainable=True)
    
    def call(self, inputs):
        inputs = tf.reshape(inputs, (-1, 784, 1))
        distances = tf.reduce_sum((inputs - self.weights)**2, axis=-1)
        bmu_indices = tf.argmin(distances, axis=1)
        bmu = tf.gather(self.weights, bmu_indices)
        return bmu

    def fit(self, x_train, epochs=100):
        for epoch in range(epochs):
            for x in x_train:
                x = tf.cast(tf.reshape(x, (-1, 784, 1)), tf.float32)
                distances = tf.reduce_sum((x - self.weights)**2, axis=-1)
                bmu_index = tf.argmin(distances, axis=0)
                bmu = tf.gather(self.weights, bmu_index)
                sigma = self.sigma or tf.math.ceil(tf.constant(200 / (epoch + 1)))
                lr = self.learning_rate / (epoch + 1)
                influence = tf.math.exp(-distances / (2 * sigma ** 2))
                self.weights.assign_add(lr * influence * (x - self.weights))

class GATLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model, dff, max_seq_len, spiking_threshold=0.42):
        super(GATLayer, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.max_seq_len = max_seq_len
        self.spiking_threshold = spiking_threshold

        self.Wq = layers.Dense(units=d_model)
        self.Wk = layers.Dense(units=d_model)
        self.Wv = layers.Dense(units=d_model)
        self.fully_connected1 = layers.Dense(units=dff, activation='relu')
        self.fully_connected2 = layers.Dense(units=dff, activation='sigmoid')
        self.dropout = layers.Dropout(0.1)
        self.spiking_layer = layers.Dense(units=1, activation=tf.nn.relu)
        
    def call(self, inputs):
        q = self.Wq(inputs)  # Query
        k = self.Wk(inputs)  # Key
        v = self.Wv(inputs)  # Value
        
        # Here you can perform additional computations specific to your deep spiking neural network
        x = self.fully_connected1(q)
        x = self.fully_connected2(x)
        x = self.dropout(x)

        # Spiking activation
        spiking_output = self.spiking_layer(x)
        spiking_output = tf.where(spiking_output < self.spiking_threshold, 0.0, 1.0)

        return spiking_output

        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

class AdversarialLayer(Layer):
    def __init__(self, num_neurons, **kwargs):
        super(AdversarialLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons

    def build(self, input_shape):
        self.weights = self.add_weight(shape=(input_shape[1], self.num_neurons), initializer='random_normal', trainable=True)
        super(AdversarialLayer, self).build(input_shape)
    def call(self, inputs):
        transformed_inputs = K.dot(inputs, self.weights)
        return transformed_inputs
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_neurons)

class TernaryDenseSparseDeepSpikingLayer(tf.keras.layers.Layer):
    def __init__(self, num_inputs=1000, num_outputs=1000, num_layers=42, magnitude_weighting=True, cosine_activation=True, **kwargs):
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.magnitude_weighting = magnitude_weighting
        self.cosine_activation = cosine_activation
        super(TernaryDenseSparseDeepSpikingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        
        self.layer_weights = []
        self.layer_biases = []
        self.layer_weights.append(self.add_weight(name='input_weights', shape=(input_shape[-1], self.num_outputs), initializer='random_normal', trainable=True))
        self.layer_biases.append(self.add_weight(name='input_biases', shape=(self.num_outputs,), initializer='zeros', trainable=True))


        for i in range(1, self.num_layers):
            
            self.layer_weights.append(self.add_weight(name='hidden' + str(i) + '_weights', shape=(self.num_outputs, self.num_outputs), initializer='random_normal', trainable=True))

            self.layer_biases.append(self.add_weight(name='hidden' + str(i) + '_biases', shape=(self.num_outputs,), initializer='zeros', trainable=True))


            self.layer_weights.append(self.add_weight(name='output_weights', shape=(self.num_outputs, input_shape[-1]), initializer='random_normal', trainable=True))
            self.layer_biases.append(self.add_weight(name='output_biases', shape=(input_shape[-1],), initializer='zeros', trainable=True))

        super(TernaryDenseSparseDeepSpikingLayer, self).build(input_shape)

class DS_NASRL():
    def __init__(self, num_conv_layers, num_dense_layers, in_shape, out_shape):
        self.num_conv_layers = num_conv_layers
        self.num_dense_layers = num_dense_layers
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.model = self._build_model()

    def _spatial_squeeze(self, x):
        return K.squeeze(K.squeeze(x, 1), 1)


    def _build_generator(self):
        model = Sequential()
        model.add(Dense(256, input_shape=self.in_dim))
        model.add(Activation('relu'))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dense(np.prod(self.in_dim)))
        model.add(Activation('tanh'))
        model.add(Reshape(self.in_dim))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model
    
    def _build_discriminator(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.in_dim))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model
    
    def _build_adversary(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.in_dim))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer)
        return model
    
    

    def _build_model(self):
        model = Sequential()

        #Add Convolutional layers to the model
        for i in range(self.num_conv_layers):
            if i == 0:
                model.add(Conv2D(32, (3, 3), padding='same', input_shape=self.in_shape))
            else:
                model.add(Conv2D(32 * pow(2,(i-1)), (3, 3), padding='same'))

            model.add(Dropout(0.5))
            model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(Flatten())

        #Add Dense layers to the model
        for i in range(self.num_dense_layers):
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))

        model.add(Dense(np.prod(self.out_shape), activation='sigmoid'))
        model.add(Reshape(self.out_shape))

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))

        return model

class ANLsmLayer(Layer):
    
    def __init__(self, n_reservoir, spectral_radius, sparsity, alpha, **kwargs):
        """
        Parameters:
        - n_reservoir: number of neurons in the reservoir
        - spectral_radius: maximum eigenvalue of the reservoir matrix
        - sparsity: percentage of the weights that are zero in the reservoir matrix
        - alpha: scaling factor for the analog input signal
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.alpha = alpha
        super(ANLsmLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        _, self.input_size = input_shape
        
        # Create the reservoir matrix with spectral_radius and sparsity
        W_reservoir = tf.random.normal(shape=[self.input_size+self.n_reservoir, self.n_reservoir], mean=0, stddev=1.0)
        W_reservoir *= tf.cast(tf.random.uniform(shape=[self.input_size+self.n_reservoir, self.n_reservoir]) < self.sparsity, tf.float32)
        max_eig = tf.reduce_max(tf.abs(tf.linalg.eigvals(W_reservoir)))
        self.W_reservoir = W_reservoir * (self.spectral_radius / max_eig)
        
        # Initialize reservoir state to zero
        self.reservoir_state = tf.zeros([self.n_reservoir])
        
        # Initialize the feedforward weights W_out
        self.W_out = self.add_weight(name='W_out', shape=[self.n_reservoir, 1],
                                     initializer='random_normal', trainable=True)
        super(ANLsmLayer, self).build(input_shape)
        
    def call(self, input):
        # Apply the analog input signal to the reservoir
        analog_input = input * self.alpha
        concatenated_input = tf.concat([analog_input, self.reservoir_state], axis=0)
        self.reservoir_state = tf.tanh(tf.matmul(concatenated_input[tf.newaxis,:], self.W_reservoir))
        
        # Predict output using the trained output weights
        output = tf.matmul(self.reservoir_state, self.W_out)
        return output
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

class HTMLayer(Layer):
    def __init__(self, n_inputs, n_columns, n_cells_per_column, initial_permanence_threshold=0.5, connected_permanence_threshold=0.8, min_overlap=10, lr=0.314, **kwargs):
        """  Parameters: - n_inputs: number of input bits -
        n_columns: number of columns in the HTM layer -
        n_cells_per_column: number of cells (synapses) per column -
        initial_permanence_threshold: initial threshold for permanences (0 to 1) -
        connected_permanence_threshold: threshold for permanences to be considered connected (0 to 1) -
        min_overlap: minimum number of active input bits required to activate a column -
        lr: learning rate for update rules """
        self.n_inputs = n_inputs
        self.n_columns = n_columns
        self.n_cells_per_column = n_cells_per_column
        self.initial_permanence_threshold = initial_permanence_threshold
        self.connected_permanence_threshold = connected_permanence_threshold
        self.min_overlap = min_overlap
        self.lr = lr
        super(HTMLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.input_size = input_shape
        self.columns = np.random.rand(self.n_columns, self.n_cells_per_column, self.n_inputs)
        self.columns *= self.initial_permanence_threshold
        W_out
        self.W_out = self.add_weight(name='W_out', shape=[self.n_columns, 1], initializer='random_normal', trainable=True)
        super(HTMLayer, self).build(input_shape)

    def call(self, input):
     overlaps = np.sum(input[:, np.newaxis, :] * self.columns, axis=-1)
     # Select the active columns (those that surpass min_overlap)
     active_columns = np.where(overlaps > self.min_overlap, 1, 0) # Compute the predicted output by summing the active columns' weights
     predicted_output = np.sum(active_columns[:, :, np.newaxis] * self.W_out, axis=0) # Update the permanences of the connected synapses
     connected_synapses = self.columns > self.connected_permanence_threshold
     input_reshaped = input[np.newaxis, :, np.newaxis]
     overlaps_reshaped = overlaps[:, :, np.newaxis]
     dw = self.lr * (input_reshaped * connected_synapses - overlaps_reshaped * connected_synapses)
     self.columns += dw
     return predicted_output

    def compute_output_shape(self, input_shape):

        return (input_shape[0], 1)

class RecursiveTransformerArray(Layer):
    def __init__(self, units, num_layers, max_length=100, **kwargs):
        # units: Number of units in each layer's hidden state
        # num_layers: Number of layers in the recursive transformer array
        # max_length: Maximum length of the input sequence (or time steps)
        
        self.units = units
        self.num_layers = num_layers
        self.max_length = max_length
        
        self.transformer_layers = []
        self.rnn_layer = tf.keras.layers.LSTM(units, return_sequences=True)
        
        # Create the transformer layers for each layer of the recursive transformer array
        for i in range(self.num_layers):
            self.transformer_layers.append(
                tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64, value_dim=64, dropout=0))
            
        super(RecursiveTransformerArray, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Pad the input if necessary
        if inputs.shape[1] < self.max_length:
            pads = self.max_length - inputs.shape[1]
            inputs = tf.pad(inputs, [[0, 0], [0, pads], [0, 0]])
        
        # Process the input recursively using the transformer layers
        for i in range(self.num_layers):
            # Transformer layer forward pass
            input_transformed = self.transformer_layers[i](inputs, inputs)
            inputs = tf.keras.layers.LayerNormalization()(input_transformed + inputs)
            
            # Apply RNN to the transformer output
            inputs = self.rnn_layer(inputs)
        
        # Return the output of the final RNN layer
        return inputs

class Iterator(Layer):
    def __init__(self, filters, kernel_size, timestep, strides=(1, 1), activation=None, padding='valid',
                 data_format=None, dilation_rate=(1, 1), use_bias=True, kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, bias_constraint=None, return_sequences=False, normalization=False,
                 train_normalization=True, num_layers=2, max_length=100, n_inputs=256, n_columns=2048, n_cells_per_column=32,
                 initial_permanence_threshold=0.5, connected_permanence_threshold=0.8, min_overlap=10, lr=0.314,
                 n_reservoir=1000, spectral_radius=0.9, sparsity=0.5, alpha=1.0, num_conv_layers=3, num_dense_layers=2,
                 in_shape=(32, 32, 1), out_shape=(10,), num_inputs=1000, num_outputs=1000, num_spiking_layers=42,
                 magnitude_weighting=True, cosine_activation=True, num_neurons=128, spiking_threshold=0.42, num_heads=8,
                 d_model=512, dff=2048, num_som_neurons=10000, som_sigma=None, som_learning_rate=0.42, **kwargs):
        
        super(Iterator, self).__init__(**kwargs)
        
        # Add an HTM layer
        self.htm_layer = HTMLayer(n_inputs=n_inputs, n_columns=n_columns, n_cells_per_column=n_cells_per_column,
                                  initial_permanence_threshold=initial_permanence_threshold,
                                  connected_permanence_threshold=connected_permanence_threshold,
                                  min_overlap=min_overlap, lr=lr)
        
        # Add a transformer layer similar to GPT-3
        self.transformer = RecursiveTransformerArray(units=64, num_layers=num_layers, max_length=max_length)
        self.transformer = tf.keras.experimental.SequenceTransduction(tf.keras.layers.MultiHeadAttention(
                                            num_heads=8, key_dim=64, value_dim=64, dropout=0))
        
        # Add an ANLSM layer
        self.anlsm_layer = ANLsmLayer(n_reservoir=n_reservoir, spectral_radius=spectral_radius, sparsity=sparsity, alpha=alpha)
        
        # Add a Deep Supervised Neural Architecture Search (DS-NAS) Reinforcement Learning (RL) layer
        self.ds_nasrl_layer = DS_NASRL(num_conv_layers=num_conv_layers, num_dense_layers=num_dense_layers,
                                       in_shape=in_shape, out_shape=out_shape)
        
        # Add a ternary dense sparse deep spiking layer
        self.tdssdslayer = TernaryDenseSparseDeepSpikingLayer(num_inputs=num_inputs, num_outputs=num_outputs,
                                                             num_layers=num_spiking_layers, magnitude_weighting=magnitude_weighting,
                                                             cosine_activation=cosine_activation)
        
        # Add an adversarial layer
        self.adv_layer = AdversarialLayer(num_neurons=num_neurons)
        
        # Add a graph attention network (GAT) layer
        self.gat_layer = GATLayer(num_heads=num_heads, d_model=d_model, dff=dff, max_seq_len=max_length,
                                  spiking_threshold=spiking_threshold)
        
        # Add a self-organizing map (SOM) layer
        self.som_layer = SOM(num_neurons=num_som_neurons, sigma=som_sigma, learning_rate=som_learning_rate)
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.timestep = timestep
        self.strides = strides
        self.activation = activation
        self.padding = padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.return_sequences = return_sequences
        self.normalization = normalization
        self.train_normalization = train_normalization
        
        # Define a Bayesian Dense layer with 64,000 neurons and int8 weights
        self.kernel = self.add_weight(shape=(self.kernel_size[0], self.kernel_size[1], None, self.filters),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint, dtype='int8')
        # Initialize the weights to be positive
        self.kernel.assign(K.abs(self.kernel))
        
        # Define int8 biases
        self.bias = self.add_weight(shape=(self.filters,),
                                    initializer=self.bias_initializer,
                                    name='bias',
                                    regularizer=self.bias_regularizer,
                                    constraint=self.bias_constraint, dtype='int8')
        # Initialize the biases to be positive
        self.bias.assign(K.abs(self.bias))
        
        if self.normalization:
            self.batch_norm = BatchNormalization(trainable=self.train_normalization)
        
        # Define a Bayesian Dense layer with 64,000 neurons
        self.bnn = tfp.layers.DenseFlipout(640000)
        
        # Add dropout regularization to reduce overfitting
        self.dropout = tf.keras.layers.Dropout(0.2) 
        
        # Add a Recursive layer
        self.rnn = tf.keras.layers.LSTM(64, return_sequences=True)
    
    def compute_output_shape(self, input_shape):
        if self.return_sequences:
            return tuple(list(input_shape[:-1]) + [self.filters])
        else:
            return tuple(list(input_shape[:-1]) + [6400,6400, self.filters])

    def build(self, input_shape):
        if len(input_shape) == 4:
            self.padding = ((0, 0) for o in self.padding) # Convert padding to tuple of tuples 
            self.input_dim = input_shape[-1]
        elif len(input_shape) == 3:
            self.padding = ((0, 0) for o in self.padding) # Convert padding to tuple of tuples 
            self.input_dim = input_shape[-1]
        elif len(input_shape) == 2:
            self.padding = ((0, 0), (0, 0) or o in self.padding) # Convert padding to tuple of tuples 
            self.input_dim = input_shape[1]
        else:
            self.padding = ((0, 0), (0, 0)) # No padding required 
            self.input_dim = input_shape[1]
        
        super(Iterator, self).build(input_shape)

    def call(self, inputs):
        # Convert inputs to int8 data type
        inputs = K.cast(inputs, 'int8')
        
        # Pad inputs
        padding = self.padding
        for i in range(len(padding)):
            if padding[i][0] > 0 or padding[i][1] > 0:
                inputs = K.spatial_2d_padding(inputs, padding[i])
        
        # Pass the input through the HTM layer
        htm_output = self.htm_layer(inputs)
        
        # Pass the output through the recursive transformer array
        transformed_output = self.transformer(htm_output)
        
        # Pass the output through the ANLSM layer
        anlsm_output = self.anlsm_layer(transformed_output)
        
        # Pass the output through the DS-NASRL layer
        ds_nasrl_output = self.ds_nasrl_layer(anlsm_output)
        
        # Pass the output through the ternary dense sparse deep spiking layer
        spiking_output = self.tdssdslayer(ds_nasrl_output)
        
        # Pass the output through the adversarial layer
        adv_output = self.adv_layer(spiking_output)
        
        # Pass the output through the graph attention network (GAT) layer
        gat_output = self.gat_layer(adv_output)
        
        # Pass the output through the self-organizing map (SOM) layer
        som_output = self.som_layer(gat_output)
        
        # Synaptic input is the convolutional output plus the bias term
        synaptic_input = K.conv2d(som_output, self.kernel,
                                  strides=self.strides,
                                  padding='valid',
                                  data_format=self.data_format,
                                  dilation_rate=self.dilation_rate)

        if self.use_bias:
            synaptic_input = K.bias_add(synaptic_input, self.bias,
                                        data_format=self.data_format)

        # Convert synaptic input to spikes
        spikes = K.round(synaptic_input * self.timestep)

        # Apply activation function
        if self.activation is not None:
            spikes = self.activation(spikes)

        # Apply normalization
        if self.normalization:
            spikes = self.batch_norm(spikes)

        # Pass the input through the dropout layer
        dropout_output = self.dropout(spikes, training=True)
    
        # Pass the dropout output through the BNN layer
        bnn_output = self.bnn(dropout_output)
    
        # Convert the BNN output to a deterministic prediction by sampling from its posterior distribution
        output = tfp.distributions.MultivariateNormalDiag(loc=bnn_output, scale_diag=0.2).sample()
        
        # Add a Recursive layer
        recursive_output = self.rnn(output)
        
        # Convert output to int8 data type
        output = K.cast(recursive_output, 'int8')
        
        # Unpad the output if necessary
        for i in range(len(padding)):
            if padding[i][0] > 0 or padding[i][1] > 0:
                output = output[:, padding[i][0]:-padding[i][1], padding[i][2]:-padding[i][3], :]
        
        if self.return_sequences:
            return output
        else:
            return K.expand_dims(K.sum(output, axis=1), axis=-2)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'timestep': self.timestep,
            'strides': self.strides,
            'activation': self.activation,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'return_sequences': self.return_sequences,
            'normalization': self.normalization,
            'train_normalization': self.train_normalization,
            'num_layers': self.num_layers,
            'max_length': self.max_length,
            'n_inputs': self.n_inputs,
            'n_columns': self.n_columns,
            'n_cells_per_column': self.n_cells_per_column,
            'initial_permanence_threshold': self.initial_permanence_threshold,
            'connected_permanence_threshold': self.connected_permanence_threshold,
            'min_overlap': self.min_overlap,
            'lr': self.lr,
            'n_reservoir': self.n_reservoir,
            'spectral_radius': self.spectral_radius,
            'sparsity': self.sparsity,
            'alpha': self.alpha,
            'num_conv_layers': self.num_conv_layers,
            'num_dense_layers': self.num_dense_layers,
            'in_shape': self.in_shape,
            'out_shape': self.out_shape,
            'num_inputs': self.num_inputs,
            'num_outputs': self.num_outputs,
            'num_spiking_layers': self.num_spiking_layers,
            'magnitude_weighting': self.magnitude_weighting,
            'cosine_activation': self.cosine_activation,
            'num_neurons': self.num_neurons,
            'spiking_threshold': self.spiking_threshold,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'dff': self.dff,
            'num_som_neurons': self.num_som_neurons,
            'som_sigma': self.som_sigma,
            'som_learning_rate': self.som_learning_rate
        }
        base_config = super(Iterator, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))













