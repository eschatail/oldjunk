import keras.backend as K
from keras.layers import Layer, Dense, Input, Conv2DTranspose, Activation, Dropout, Flatten, Reshape, Conv2D, \
    MaxPooling2D, LSTM, InputLayer
import tensorflow as tf
from keras.models import Sequential
import random
import numpy as np
from keras.optimizers import Adamax, Lion
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Input, Conv2DTranspose, \
    Activation, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, LSTM, BatchNormalization, Lambda
import tensorflow_probability as tfp


class FoldingFourier(tf.keras.layers.Layer):

    def __init__(self, num_segments=7, **kwargs):
        super(FoldingFourier, self).__init__(**kwargs)
        self.num_segments = num_segments

        # Precompute fixed point values
        self.fixed_points = tf.constant([0, tf.constant(np.pi/2), tf.constant(np.pi)])

    def build(self, input_shape):
        pass  # No trainable weights to be created

    def call(self, inputs):
        # Calculate values for the first eighth of the sine
        x = K.arange(0, tf.constant(np.pi/2), tf.constant(np.pi/(2*self.num_segments)))
        sine_values = K.sin(x)

        # Use folding and mirroring to populate the remaining segments
        folded_values = tf.reverse(sine_values, [0])  # Reflect the values
        mirrored_values = tf.concat([folded_values[:-1], sine_values], axis=0)  # Mirror the values

        # Combine the computed values with the precomputed fixed point values
        all_values = tf.concat([self.fixed_points, mirrored_values], axis=0)

        # Interpolate based on the input values
        rescaled_values = tf.gather(all_values, tf.cast(inputs * tf.constant(self.num_segments/np.pi), tf.float32))

        return rescaled_values

    def compute_output_shape(self, input_shape):
        return input_shape  # The output shape is the same as the input shape



class BabyQstar(tf.keras.Model):
    def __init__(self):
        super(BabyQstar, self).__init__()
        self.policy_nn = GPTModel()  # Policy NN: GPT-based model for implementing thought traces
        self.value_nn = GPTModel()  # Value NN: GPT-based model for scoring intermediate reasoning steps

    def call(self, inputs):
        # Pass inputs through policy NN to generate solution thought traces
        thought_traces = self.policy_nn(inputs)

        # Pass thought traces through value NN to score the likelihood of correctness for each reasoning step
        scores = self.value_nn(thought_traces)

        return thought_traces, scores


class GPTModel(tf.keras.Model):
    def __init__(self):
        super(GPTModel, self).__init__()
        # Define layers and architecture for the GPT-based model
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.transformer_blocks = [
            layers.TransformerBlock() for _ in range(num_transformer_blocks)
        ]
        self.dense = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        outputs = self.dense(x)
        return outputs


class QModel(tf.keras.Model):
    def __init__(self, num_actions):
        super(QModel, self).__init__()
        self.policy_nn = GPTQCapsules()  # Policy NN: GPT-based model for implementing thought traces
        self.value_nn = GPTQCapsules()  # Value NN: GPT-based model for scoring intermediate reasoning steps
        self.qlearning = QLearning(num_actions)
        self.num_actions = num_actions

    def call(self, inputs, training=None):
        # Pass inputs through policy NN to generate solution thought traces
        thought_traces = self.policy_nn(inputs)

        # Pass thought traces through value NN to score the likelihood of correctness for each reasoning step
        scores = self.value_nn(thought_traces)

        # Pass thought traces to QLearning to compute Q-values
        q_values = self.qlearning(thought_traces)

        return thought_traces, scores, q_values


class GPTQCapsules(tf.keras.Model):
    def __init__(self):
        super(GPTQCapsules, self).__init__()
        # Define layers and architecture for the GPT-based model
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.transformer_blocks = [tf.keras.layers.TransformerBlock() for _ in
                                   range(num_transformer_blockss=8, capsule_dim=16, routings=3)]
        self.routing_capsules = tf.keras.layers.CapsuleLayer(num_capsules=num_classes, capsule_dim=16, routings=3)

    def call(self, inputs):
        x = self.embedding(inputs)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.primary_capsules(x)
        outputs = self.routing_capsules(x)
        return outputs


class QLearning(keras.Model):
    def __init__(self, num_actions):
        super(QLearning, self).__init__()
        self.qmodel = QModel(num_actions)
        self.num_actions = num_actions

    def call(self, inputs, training=None):
        thought_traces, scores = self.qmodel(inputs)
        q_values = scores[:, -1, :]  # Take the scores for the final step as Q-values
        return q_values

    def train_step(self, states, actions, rewards, next_states):
        with tf.GradientTape() as tape:
            # Compute Q-values for current states
            q_values = self(states, training=True)

            # Use epsilon-greedy exploration strategy to select actions for next_states
            next_q_values = self(next_states, training=True)
            explore = tf.random.uniform(actions.shape[:1]) < epsilon
            random_actions = tf.random.uniform(actions.shape, maxval=self.num_actions, dtype=tf.float32)
            next_actions = tf.where(explore, random_actions, tf.argmax(next_q_values, axis=1))

            # Compute target Q-values using bellman equation
            target_q_values = rewards + discount_factor * tf.reduce_max(next_q_values, axis=1)
            mask = tf.one_hot(actions, self.num_actions)
            q_values_masked = tf.reduce_sum(q_values * mask, axis=1)
            loss = tf.reduce_mean(tf.square(target_q_values - q_values_masked))

        # Update model weights
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}


# Create an instance of the QLearning class
#ql_model = QLearning(num_actions)


def maml_train(model, tasks, optimizer, inner_steps=1, alpha=0.01):
    for task in tasks:
        task_x, task_y = task.train
        with tf.GradientTape() as outer_tape:
            inner_loss = 0.0
            for i in range(inner_steps):
                with tf.GradientTape(persistent=True) as inner_tape:
                    y_pred = model(task_x)
                    loss = tf.reduce_mean(tf.keras.losses.CosineSimilarity())
                grads = inner_tape.gradient(loss, model.trainable_variables)
                model.set_weights([w - alpha * g for w, g in zip(model.get_weights(), grads)])
                inner_loss += loss
            inner_loss /= inner_steps
            y_pred = model(task_x)
            outer_loss = tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy()) + inner_loss
        grads = outer_tape.gradient(outer_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def maml_fine_tune(model, task, optimizer, inner_steps=1, alpha=0.01):
    task_x, task_y = task.train
    with tf.GradientTape() as outer_tape:
        inner_loss = 0.0
        for i in range(inner_steps):
            with tf.GradientTape(persistent=True) as inner_tape:
                y_pred = model(task_x)
                loss = tf.reduce_mean(tf.keras.losses.CosineSimilarity())
            grads = inner_tape.gradient(loss, model.trainable_variables)
            model.set_weights([w - alpha * g for w, g in zip(model.get_weights(), grads)])
            inner_loss += loss
        inner_loss /= inner_steps
        y_pred = model(task_x)
        outer_loss = tf.reduce_mean(tf.keras.losses.SparseCategoricalCrossentropy()) + inner_loss
    grads = outer_tape.gradient(outer_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def zero_shot_learn(model, unseen_x, unseen_y, seen_x, seen_y, alpha=0.01):
    seen_y_one_hot = tf.one_hot(seen_y, depth=output_shape[-1])
    with tf.GradientTape() as tape:
        seen_y_pred = model(seen_x)
        seen_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(seen_y_one_hot, seen_y_pred))
    seen_grads = tape.gradient(seen_loss, model.trainable_variables)
    seen_optimizer = tf.keras.optimizers.Lion()
    seen_optimizer.apply_gradients(zip(seen_grads, model.trainable_variables))

    with tf.GradientTape() as tape:
        seen_y_pred = model(seen_x)
        seen_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(seen_y_one_hot, seen_y_pred))
        unseen_y_pred = model(unseen_x)
        unseen_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(unseen_y, unseen_y_pred))
        total_loss = seen_loss + unseen_loss
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer = tf.keras.optimizers.Lion()
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


global_memory_size = tf.constant((7, 7, 7), dtype=tf.float32)
local_memory_size = tf.constant((7, 7, 7), dtype=tf.float32)
num_internal_memory_matrices = 7
internal_memory_sizes = [(7, 7, 7), (7, 7, 7), (7, 7, 7), (7, 7, 7)]

# Define initial values for the global and local memory cells
global_memory_init = tf.zeros_like(global_memory_size, dtype=tf.float32)
local_memory_init = tf.zeros_like(local_memory_size, dtype=tf.float32)
internal_memory_init = [tf.zeros_like(size, dtype=tf.float32) for size in internal_memory_sizes]


def memory_cell(inputs, global_memory=global_memory_init, local_memory=local_memory_init, internal_memory=7):
    # Combine input with previous local memory
    x = inputs
    x = tf.squeeze(x, axis=-2)
    x = tf.concat([x, local_memory, global_memory], axis=-1)

    # Apply global and local memory updates
    global_memory_update = tf.matmul(x, tf.random.normal(global_memory.shape, dtype=tf.float32))
    global_memory_total = global_memory + global_memory_update

    local_memory_update = tf.matmul(x, tf.random.normal(local_memory.shape, dtype=tf.float32))
    local_memory_total = local_memory + local_memory_update + global_memory_update

    # Write to internal memory matrices
    if internal_memory is not None:
        write_masks = [tf.cast(tf.math.greater(local_memory_update, 0), dtype=tf.float32) for _ in
                       range(num_internal_memory_matrices)]
        output = [local_memory * write_masks[i] + x * (1 - write_masks[i]) for i in
                  range(num_internal_memory_matrices)]
        for i in range(num_internal_memory_matrices):
            internal_memory[i] = tf.concat([global_memory[i], output[i]], axis=0)

    if global_memory is not None:
        write_masks = [tf.cast(tf.math.greater(global_memory_update, 0), dtype=tf.float32) for _ in
                       range(num_internal_memory_matrices)]
        output = [local_memory * write_masks[i] + x * (1 - write_masks[i]) for i in
                  range(num_internal_memory_matrices)]
        for i in range(num_internal_memory_matrices):
            global_memory[i] = tf.concat([global_memory[i], output[i]], axis=0)

    # Return output and updated local memory
    return output, local_memory_total, global_memory_total


layer_configs = [
    {'type': 'memory_cell_layer',
        'params': {'memory_cell_fn': memory_cell, 'internal_memory': internal_memory_init}}
    ]






class FoxNet(tf.keras.Model):
    def __init__(self, mutation_rate=0.42, activation=tf.keras.activations.swish,
                 normalization=False, use_bias=True, kernel_init='glorot_uniform',
                 bias_init='zeros', batch_norm_momentum=0.99, dropout_rate=0.002,
                 bnn_units=10, bnn_activation=tf.keras.activations.swish,
                 bnn_scale=0.2, **kwargs):
        super(FoxNet, self).__init__(**kwargs)


        self.best_failed_count = None
        self.mutation_rate = mutation_rate
        self.activation = tf.keras.activations.get(activation)
        self.normalization = normalization
        self.use_bias = use_bias
        self.kernel_init = tf.keras.initializers.get(kernel_init)
        self.bias_init = tf.keras.initializers.get(bias_init)
        self.batch_norm_momentum = batch_norm_momentum
        self.dropout_rate = dropout_rate
        self.bnn_units = bnn_units
        self.bnn_activation = tf.keras.activations.get(bnn_activation)
        self.bnn_scale = bnn_scale

        self.dropout = Dropout(self.dropout_rate, dtype=tf.float32)
        self.bnn = tfp.layers.DenseFlipout(self.bnn_units,
                                           activation=self.bnn_activation, dtype=tf.float32)

        self.output_distribution = tfp.distributions.MultivariateNormalDiag(loc=0., scale_diag=[self.bnn_scale])

        self.initial_layer = GPTModel()
        # Adjust the input shape and units as needed
        self.final_layer = GPTModel()
        # Adjust the input shape and units as needed
        self.babyq = BabyQstar()
        self.gptm = GPTModel()


    def call(self, inputs):

        Lambda(lambda x: x)(inputs)

#        input_shape = inputs  # Adjust the input shape

#        input_layer = Input(shape=input_shape)

        llm_input = tf.expand_dims(inputs, axis=-1)

        llm_input = self.babyq(llm_input)

        llm_input = self.initial_layer(llm_input)  # Apply the initial layer to input


        llm_input = Lambda(lambda x: x)(llm_input)

        #initial_input = tf.squeeze(initial_input, axis=1)

        #initial_input = memory_cell(initial_input, global_memory=global_memory_init, local_memory=local_memory_init,
                             #       internal_memory=7)

#        v = tf.strings.to_number(input_layer, out_type=tf.float16)
#        initial_input = tf.expand_dims(v, axis=3)

        output = self.output_distribution.sample()
        output = Lambda(lambda x: x)(output)
        output = tf.expand_dims(output, axis=-1)
        output = self.bnn(output)
        output = Lambda(lambda x: x)(output)

        spikes = tf.round(output)
        spikes = Lambda(lambda x: x)(spikes)

        if self.activation is not None:
            spikes = self.activation(spikes)

        dropout_output = self.dropout(spikes, training=True)
        dropout_output = Lambda(lambda x: x)(dropout_output)
        output = tf.multiply(output, dropout_output)

        Lambda(lambda x: x)(output)

        output = self.final_layer(llm_input)  # Apply the final layer to input

        #output = tf.multiply(output, v)

        output = Lambda(lambda x: x)(output)

        return output



    def maml_mutate(self, model, tasks, optimizer, mutation_rate, alpha=0.01):
        for task in tasks:
            task_x, task_y = task.train
            with tf.GradientTape() as outer_tape:
                inner_loss = 0.0
                for i in range(inner_steps):
                    with tf.GradientTape(persistent=True) as inner_tape:
                        y_pred = model(task_x)
                        loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(task_y, y_pred))
                    grads = inner_tape.gradient(loss, model.trainable_variables)
                    model.set_weights([w - alpha * g for w, g in zip(model.get_weights(), grads)])
                    inner_loss += loss
                inner_loss /= inner_steps
                y_pred = model(task_x)
                outer_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(task_y, y_pred)) + inner_loss
            grads = outer_tape.gradient(outer_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Choose a mutation type randomly
        mutation_type = np.random.choice(['add', 'delete', 'replace', 'mutate'])

        if mutation_type == 'add':
            # Choose a random layer type and randomly generate its parameters
            new_layer_type = np.random.choice(
                ['fox_htm_layer', 'fox_transformer', 'fox_anlsm_layer', 'fox_ds_nasrl_layer', 'fox_tdssdslayer',
                 'fox_adv_layer', 'fox_gat_layer', 'fox_som_layer'])
            new_layer_params = {}
            if new_layer_type == 'fox_htm_layer':
                new_layer_params['n_inputs'] = np.random.randint(0, 0)
                new_layer_params['n_columns'] = np.random.randint(4, 7)
                new_layer_params['n_cells_per_column'] = np.random.randint(4, 7)
            elif new_layer_type == 'fox_transformer':
                new_layer_params['units'] = np.random.choice([3, 5, 7])
                new_layer_params['num_layers'] = np.random.choice([3, 5, 7])
                new_layer_params['max_length'] = np.random.choice([3, 5, 7])
            elif new_layer_type == 'fox_anlsm_layer':
                new_layer_params['n_reservoir'] = np.random.randint(4, 7)
                new_layer_params['spectral_radius'] = np.random.uniform(0.5, 2.0)
                new_layer_params['sparsity'] = np.random.uniform(0.05, 0.5)
                new_layer_params['alpha'] = np.random.uniform(0.5, 2.0)
            elif new_layer_type == 'fox_ds_nasrl_layer':
                new_layer_params['num_layers'] = np.random.choice([3, 5, 7])
            elif new_layer_type == 'fox_tdssdslayer':
                new_layer_params['num_layers'] = np.random.choice([3, 5, 7])
            elif new_layer_type in ['fox_adv_layer', 'fox_gat_layer']:
                new_layer_params['num_neurons'] = np.random.choice([3, 5, 7])
            elif new_layer_type == 'fox_som_layer':
                new_layer_params['num_neurons'] = np.random.choice([3, 5, 7])
                new_layer_params['learning_rate'] = np.random.uniform(0.01, 0.1)

            # Determine the genealogy set for the new layer instance
            ancestor_indices = set()
            for i, layer in enumerate(model.layers):
                if isinstance(layer, (fox_NASRL, eval(new_layer_type))):
                    ancestor_indices.update(genealogy[i])
            if len(ancestor_indices) == 0:
                ancestor_indices = {index for index in range(model.n_layers)}

            # Add the new layer instance and update the genealogy dictionary
            new_layer_instance = eval(new_layer_type)(**new_layer_params)
            model.layers.append(new_layer_instance)
            model.layer_configs.append({'type': new_layer_type, 'params': new_layer_params})
            genealogy[model.n_layers] = ancestor_indices | {model.n_layers}
            model.n_layers += 1

        elif mutation_type == 'delete':
            # Choose a random layer instance to be deleted
            potential_indices = [i for i in range(model.n_layers) if len(
                genealogy[i]) > 1 and i not in nas_layer_indices and i not in original_layer_indices]
            if len(potential_indices) > 0:
                index = np.random.choice(potential_indices)
            else:
                index = np.random.choice([i for i in range(model.n_layers) if
                                          i not in nas_layer_indices and i not in original_layer_indices])

            # Determine permitted deletion indices
            genealogy_set = genealogy[index]
            permitted_indices = set()
            for i, layer_genealogy in genealogy.items():
                if i in nas_layer_indices:
                    if layer_genealogy.issubset(genealogy_set):
                        permitted_indices.add(i)
                elif i in original_layer_indices:
                    if layer_genealogy.issubset(genealogy_set) and i not in nas_layer_indices:
                        permitted_indices.add(i)
                else:
                    if layer_genealogy.issubset(
                            genealogy_set) and i not in nas_layer_indices and i not in original_layer_indices:
                        permitted_indices.add(i)

            # Remove the chosen layer instance and update the genealogy dictionary
            model.layers.pop(index)
            model.layer_configs.pop(index)
            genealogy.pop(index)
            model.n_layers -= 1

            for i, layer_genealogy in genealogy.items():
                if len(layer_genealogy.intersection(genealogy_set)) > 0 and i not in permitted_indices:
                    # Layer instance is no longer connected to the FoxNet ancestor tree
                    genealogy[i] = {i}

        elif mutation_type == 'replace':
            # Choose a layer instance randomy
            index = np.random.choice(range(model.n_layers))
            old_layer_type = model.layer_configs[index]['type']

            if old_layer_type in ['fox_tdssdslayer', 'fox_adv_layer', 'fox_gat_layer', 'fox_nas_layer']:
                # Non-sequentially-layered instances:
                # Replace the instance with another one of a different random type that
                # preserves the input-output shape
                new_layer_type = np.random.choice(
                    ['fox_htm_layer', 'fox_transformer', 'fox_anlsm_layer', 'fox_ds_nasrl_layer'])
                while True:
                    new_layer_instance = eval(new_layer_type)(**model.layer_configs[index]['params'],
                                                              in_shape=model.layers[index].in_shape,
                                                              out_shape=model.layers[index].out_shape)
                    if new_layer_instance.compute_output_shape(model.layers[index].in_shape) == model.layers[
                        index].out_shape:
                        model.layers[index] = new_layer_instance
                        model.layer_configs[index]['type'] = new_layer_type
                        break
            elif index in original_layer_indices:
                # Original FoxNet instances cannot be replaced
                pass
            else:
                # Sequentially-layered instances:
                # Replace the instance with another instance of the same type
                new_layer_instance = eval(old_layer_type)(**model.layer_configs[index]['params'])
                model.layers[index] = new_layer_instance

        else:  # mutation_type == 'mutate'
            # Choose a layer instance randomy
            index = np.random.choice(range(model.n_layers))

            # Mutate the parameters of the layer instance
            layer_config = model.layer_configs[index]
            layer_type = layer_config['type']
            layer_params = layer_config['params']

            if layer_type == 'fox_htm_layer':
                param_names = ['n_inputs', 'n_columns', 'n_cells_per_column']
            elif layer_type == 'fox_transformer':
                param_names = ['units', 'num_layers', 'max_length']
            elif layer_type == 'fox_anlsm_layer':
                param_names = ['n_reservoir', 'spectral_radius', 'sparsity', 'alpha']
            elif layer_type == 'fox_ds_nasrl_layer':
                param_names = ['num_layers']
            elif layer_type == 'fox_tdssdslayer':
                param_names = ['num_layers']
            elif layer_type in ['fox_adv_layer', 'fox_gat_layer']:
                param_names = ['num_neurons']
            elif layer_type == 'fox_som_layer':
                param_names = ['num_neurons', 'learning_rate']
            else:
                raise ValueError('Unknown layer type %s' % layer_type)

            for param_name in param_names:
                if np.random.rand() < mutation_rate:
                    old_value = layer_params[param_name]
                    new_value = np.random.uniform(old_value * 0.5, old_value * 1.5)
                    layer_params[param_name] = new_value
            model.layer_configs[index]['params'] = layer_params
            model.layers[index].__init__(**layer_params)

            if np.random.rand() < mutation_rate:
                model.activation = np.random.choice(
                    ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign'])
            if np.random.rand() < mutation_rate:
                model.normalization = not model.normalization
            if np.random.rand() < mutation_rate:
                model.use_bias = not model.use_bias
            if np.random.rand() < mutation_rate:
                model.kernel_init = np.random.choice(['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'])
            if np.random.rand() < mutation_rate:
                model.bias_init = np.random.choice(
                    ['zeros', 'ones', 'glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'])
            if np.random.rand() < mutation_rate:
                model.batch_norm_momentum = np.random.uniform(0.9, 0.999)
            if np.random.rand() < mutation_rate:
                model.dropout_rate = np.random.uniform(0.1, 0.5)
            if np.random.rand() < mutation_rate:
                model.rnn_units = np.random.randint(4, 7)
            if np.random.rand() < mutation_rate:
                model.rnn_return_sequences = not model.rnn_return_sequences
            if np.random.rand() < mutation_rate:
                model.bnn_units = np.random.randint(4, 7)
            if np.random.rand() < mutation_rate:
                model.bnn_activation = np.random.choice(
                    ['relu', 'sigmoid', 'tanh', 'softmax', 'softplus', 'softsign'])
            if np.random.rand() < mutation_rate:
                model.bnn_scale = np.random.uniform(0.1, 0.5)
            if np.random.rand() < mutation_rate:
                model.kernel_regularizer = np.random.choice(['l1', 'l2', 'l1_l2'])
            if np.random.rand() < mutation_rate:
                model.bias_regularizer = np.random.choice(['l1', 'l2', 'l1_l2'])
            if np.random.rand() < mutation_rate:
                model.activity_regularizer = np.random.choice([None, 'l1', 'l2', 'l1_l2'])
            if np.random.rand() < mutation_rate:
                model.kernel_constraint = np.random.choice([None, 'unit_norm', 'max_norm'])
            if np.random.rand() < mutation_rate:
                model.bias_constraint = np.random.choice([None, 'unit_norm', 'max_norm'])


def build_model():
    model = FoxNet()
    model.compile(optimizer=tf.keras.optimizers.Lion and tf.keras.optimizers.Adadelta(learning_rate=1.0),
                  loss=None,
                  metrics=[tf.keras.metrics.AUC()])
    return model


if __name__ == "__main__":
    model = build_model()
