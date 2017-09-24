from preprocess.data_preprocess import DataCollector
from training.SoftmaxNN import SoftmaxNeuralNetwork

if __name__ == "__main__":

    data_collector = DataCollector()
    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    number_data = train_x.shape[0]

    softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y, list_of_neuron_on_hidden_layer=[10])
    softmax_nn.start_train(batch_size=number_data)
