import numpy as np
import theano
import theano.tensor as T
from common.time_executor import get_execution_time

# by specifying [10] as the hidden_layer_neuron implies using 1 hidden layer with 10 neurons
# respectively by specifying [100, 100] -> 2 hidden layers each layer 100 neurons


class SoftmaxNeuralNetwork:

    def __init__(self, train_x, train_y, num_features=6, list_of_neuron_on_hidden_layer=list([10]), decay=1e-6,
                 verbose=True):

        self.train_x = train_x
        self.train_y = train_y
        self.verbose = verbose

        self.train_cost = []
        self.train_prediction = []
        self.train_exec_time = []

        self.test_cost = []
        self.test_prediction = []

        weights = []
        biases = []

        # first layer which connect to the input layer
        weights.append(
            self.init_weight(len(train_x[0]), list_of_neuron_on_hidden_layer[0]) )
        biases.append(
            self.init_bias(list_of_neuron_on_hidden_layer[0]))

        previous_layer = list_of_neuron_on_hidden_layer[0]

        for layer in range(1, len(list_of_neuron_on_hidden_layer)):
            weights.append(
                self.init_weight(previous_layer, list_of_neuron_on_hidden_layer[layer]))

            biases.append(
                self.init_bias(list_of_neuron_on_hidden_layer[layer]))

        # for output layer
        weights.append(
            self.init_weight(previous_layer, num_features)
        )

        biases.append(
            self.init_bias(num_features)
        )

        # construct neural network

        layers = []

        x_input = T.matrix('X')
        y_output = T.matrix('Y')

        prev_input = x_input

        for i in range(len(weights)-1):
            calculation = T.nnet.sigmoid(T.dot(prev_input, weights[i]) + biases[i])
            layers.append(calculation)
            prev_input = calculation

        # last output layer, use softmax function
        calculation = T.nnet.softmax(T.dot(prev_input, weights[len(weights)-1]) +
                                     biases[len(biases) - 1])
        layers.append(calculation)

        y_prediction = T.argmax(calculation, axis=1)

        sum_sqr_weights = T.sqr(weights[0])

        for i in range(1, len(weights)):
            sum_sqr_weights += T.sum(T.sqr(weights[i]))

        cost = T.mean(T.nnet.categorical_crossentropy(calculation, y_output)) + decay*T.sum(sum_sqr_weights)
        params = list(weights+biases)
        updates = self.sgd(cost=cost, params=params)

        self.computation = theano.function(
            inputs=[x_input, y_output],
            updates=updates,
            outputs=cost
        )

        self.prediction = theano.function(
            inputs=[x_input],
            outputs=y_prediction
        )

        return

    def init_bias(self, n):
        return theano.shared(np.zeros(n), theano.config.floatX)

    def init_weight(self, n_in, n_out, is_logistic_function=False):

        weight = np.random.uniform(
            size=(n_in, n_out),
            low=-np.sqrt(6. / (n_in + n_out)),
            high=np.sqrt(6. / (n_in + n_out)),
        )

        if is_logistic_function:
            weight = weight*4

        return theano.shared(weight, theano.config.floatX)

    def sgd(self, cost, params, lr=0.01):

        # return list of gradients
        grads = T.grad(cost=cost, wrt=params)

        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    def start_train(self, test_x, test_y, epochs=1000, batch_size=100):

        for i in range(epochs):

            prediction_batch = []

            cost, exec_time = get_execution_time(self.start_one_iter_func, batch_size, prediction_batch)

            # predictions of train data
            prediction = np.mean(prediction_batch)

            self.train_cost.append(cost)
            self.train_prediction.append(prediction)
            self.train_exec_time.append(exec_time)

            if i % 5*batch_size == 0 or i == epochs-1:

                if self.verbose:
                    print ('epoch: %d, train cost: %s, train predictions: %s \n' % (i, cost, prediction))
                self.start_test(test_x=test_x, test_y=test_y)
                if self.verbose:
                    print('------------------------------------\n')

    def start_one_iter_func(self, batch_size, prediction_batch):

        cost = 0

        for cnt in range(0, len(self.train_x), batch_size):

            end = cnt + batch_size

            if end > len(self.train_x):
                end = len(self.train_x)

            train_x_batch = self.train_x[cnt:end]
            train_y_batch = self.train_y[cnt:end]

            cost += self.computation(train_x_batch, train_y_batch)
            prediction = self.prediction(self.train_x)
            predict_in_percentage = np.mean(np.argmax(self.train_y, axis=1) == prediction)
            prediction_batch.append(predict_in_percentage)

        return cost

    def start_test(self, test_x, test_y):

        cost = self.computation(test_x, test_y)
        prediction = self.prediction(test_x)

        self.test_cost.append(cost)
        self.test_prediction.append(prediction)

        if self.verbose:
            print ('test cost: %s, test predictions: %s \n' % (cost, np.mean(np.argmax(test_y, axis=1) == prediction)))

    def get_train_result(self):

        return self.train_cost, self.train_prediction, np.sum(self.train_exec_time)

    def get_test_result(self):

        return self.test_cost, self.test_prediction
