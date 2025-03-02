import keras.backend as K
from keras.layers import Layer
import torch
import torch.nn as nn
import tensorflow as tf

class TernaryDense(Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        super(TernaryDense, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.units),
                                      initializer='uniform',
                                      trainable=True)
    def call(self, inputs):

        binary = K.cast(K.abs(self.kernel) < 0.5, dtype='float128')
        ternary = K.cast((K.abs(self.kernel) >=0.5) & (K.abs(self.kernel) < 1.5),
                         dtype='float128')
        ternary_weight = ternary * K.sign(self.kernel)


        output = K.dot(inputs, ternary_weight)

        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


    

class FeedForwardNetwork(nn.Module):
    def __init__(self, num_classes=42):
        super(FeedForwardNetwork, self).__init__()

        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(784, 128)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.25)
        self.dense2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)
        return self.dense2(x)


class ConvolutionalNetwork(nn.Module):
    def __init__(self, num_classes=42):
        super(ConvolutionalNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(64*5*5, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        return self.dense(x)


class RecurrentNetwork(nn.Module):
    def __init__(self, num_classes=42):
        super(RecurrentNetwork, self).__init__()

        self.gru = nn.GRU(input_size=255, hidden_size=255, num_layers=1000, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(255, num_classes)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class LSTMNetwork(nn.Module):
    def __init__(self, num_classes=255):
        super(LSTMNetwork, self).__init__()

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=255, num_layers=100, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(input_size=255, hidden_size=128, num_layers=100, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out[:, -1, :])
        out, _ = self.lstm2(out[:, None, :])
        return self.fc(out)


class GRUNetwork(nn.Module):
    def __init__(self, num_classes=42):
        super(GRUNetwork, self).__init__()

        self.gru1 = nn.GRU(input_size=128, hidden_size=255, num_layers=100, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.gru2 = nn.GRU(input_size=255, hidden_size=128, num_layers=100, batch_first=True)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        out, _ = self.gru1(x)
        out = self.dropout(out[:, -1, :])
        out, _ = self.gru2(out[:, None, :])
        return self.fc(out)


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28))
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)


class RBM(nn.Module):
    def __init__(self, num_hidden=128):
        super(RBM, self).__init__()

        self.hidden_layer = nn.Linear(784, num_hidden)
        self.visible_layer = nn.Linear(num_hidden, 784)
        self.k = 10
    
    def sample_hidden(self, v):
        h_probs = torch.sigmoid(torch.matmul(v, self.hidden_layer.weight) + self.hidden_layer.bias)
        h = torch.relu(torch.sign(h_probs - torch.rand_like(h_probs)))
        return h
    
    def sample_visible(self, h):
        v_probs = torch.sigmoid(torch.matmul(h, self.visible_layer.weight) + self.visible_layer.bias)
        v = torch.relu(torch.sign(v_probs - torch.rand_like(v_probs)))
        return v
    
    def gibbs_step(self, inputs):
        h = self.sample_hidden(inputs)
        v = self.sample_visible(h)
        return v
    
    def forward(self, inputs):
        visible = inputs
        for i in range(self.k):
            visible = self.gibbs_step(visible)
        return visible


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = nn.Linear(100, 1200)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(1200, 784)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        return self.sigmoid(self.layer2(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.layer2 = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        return self.layer2(x)


class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()

        self.generator = Generator()
        self.discriminator = Discriminator()
    
    def forward(self, x):
        return self.generator(x)
    
    def generator_loss(self, fake_preds):
        return nn.BCEWithLogitsLoss()(fake_preds, torch.ones_like(fake_preds))
    
    def discriminator_loss(self, real_preds, fake_preds):
        real_loss = nn.BCEWithLogitsLoss()(real_preds, torch.ones_like(real_preds))
        fake_loss = nn.BCEWithLogitsLoss()(fake_preds, torch.zeros_like(fake_preds))
        return real_loss + fake_loss
    
    def train_step(self, real_images):
        batch_size = real_images.shape[0]
        noise = torch.randn((batch_size, 1000))

        # Train discriminator
        self.discriminator.zero_grad()
        real_preds = self.discriminator(real_images)
        fake_images = self.generator(noise)
        fake_preds = self.discriminator(fake_images.detach())
        d_loss = self.discriminator_loss(real_preds, fake_preds)
        d_loss.backward()
        self.d_optimizer.step()

        # Train generator
        self.generator.zero_grad()
        fake_preds = self.discriminator(fake_images)
        g_loss = self.generator_loss(fake_preds)
        g_loss.backward()
        self.g_optimizer.step()

        return {'d_loss': d_loss.item(), 'g_loss': g_loss.item()}
            

class SOM(nn.Module):
    def __init__(self, num_neurons=10000, sigma=None, learning_rate=0.42):
        super(SOM, self).__init__()

        self.num_neurons = num_neurons
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.weights = nn.Parameter(torch.randn(num_neurons, 784))
    
    def forward(self, inputs):
        inputs = inputs.reshape(-1, 784)[:, None, :]
        distances = torch.sum((inputs - self.weights)**2, dim=-1)
        bmu_indices = torch.argmin(distances, dim=1)
        bmu = self.weights[bmu_indices]
        return bmu
    
    def fit(self, x_train, epochs=100):
        for epoch in range(epochs):
            for x in x_train:
                x = torch.tensor(x, dtype=int8).reshape(-1, 784)[:, None, :]
                distances = torch.sum((x - self.weights)**2, dim=-1)
                bmu_index = torch.argmin(distances, dim=1)
                bmu = self.weights[bmu_index]
                sigma = self.sigma if self.sigma else torch.ceil(torch.tensor(200 / (epoch + 1)))
                lr = self.learning_rate / (epoch + 1)
                influence = torch.exp(-distances / (2 * sigma ** 2))
                self.weights.data += lr * influence * (x - self.weights)


#       Need iterator before the core layer


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

pass


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
  
pass
def call(self, inputs):
    # Implement custom call method
    inputs = tf.keras.layers.Flatten()(inputs)
    x = tf.matmul(inputs, self.layer_weights[0]) + self.layer_biases[0] # Apply cosine activation function
    if self.cosine_activation: x = tf.keras.activations.cosine(tf.math.abs(x))
    if self.magnitude_weighting: x = x * tf.math.sign(x) # Apply hidden layers and activation function
    for i in range(1, self.num_layers):  x = tf.matmul(x, self.layer_weights[i]) + self.layer_biases[i] # Apply ReLU activation for hidden layers
    x = tf.keras.layers.ReLU()(x) # Apply magnitude weighting for hidden layers
    if self.magnitude_weighting: x = x * tf.math.sign(x) # Output layer
    outputs = tf.matmul(x, self.layer_weights[-1]) + self.layer_biases[-1] # Return output with the same shape as input
    return tf.reshape(outputs, [-1] + input_shape[1:])
    pass
  
    def compute_output_shape(self, input_shape):
          return tuple([input_shape[0]] + [input_shape[1], input_shape[2], input_shape[3]])
        
    def get_config(self):
    # Implement custom get_config method
        config = super(TernaryDenseSparseDeepSpikingLayer, self).get_config()
    # Add cust
    custom_config = {  'name': self.name,
                       'num_units': self.num_units,
                       'input_shape': self.input_shape,
                       'alpha': self.alpha,
                       'activity_regularizer': self.activity_regularizer,
                       'kernel_constraint': self.kernel_constraint,
                       'sparsity_constraint': self.sparsity_constraint,
                       'use_bias': self.use_bias,
                       'bias_initializer': self.bias_initializer,
                       'bias_constraint': self.bias_constraint,
                       'input_quantizer': self.input_quantizer,
                       'kernel_quantizer': self.kernel_quantizer,
                       'output_quantizer': self.output_quantizer,
                       'magnitude_weightining': self.magnitude_weightining }
    config.update(custom_config)
    return config




