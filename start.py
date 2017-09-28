from preprocess.data_preprocess import DataCollector
from training.softmax_nn import SoftmaxNeuralNetwork
from postpreprocess.data_visualization import DataVisualization

if __name__ == "__main__":

    data_collector = DataCollector()
    train_x, train_y = data_collector.get_train_data()
    test_x, test_y = data_collector.get_test_data()

    # number_data = train_x.shape[0]
    number_data = 32
    number_epoch = 1000

    softmax_nn = SoftmaxNeuralNetwork(train_x=train_x.as_matrix(), train_y=train_y, list_of_neuron_on_hidden_layer=[10])
    softmax_nn.start_train(batch_size=number_data, test_x=test_x, test_y=test_y, epochs=number_epoch)

    cost_train, prediction_train, exec_time = softmax_nn.get_train_result()
    prediction_test = softmax_nn.get_test_result()

    print ("Execution Time: %s \n" % exec_time)

    # example of the visualize
    data_visualization = DataVisualization()
    data_visualization.show_plot(
        list_x_point=range(number_epoch), list_y_point=cost_train,
        x_label="epochs", y_label="costs", title="Cross Entropy", figure_name="cross_entropy_cost.png"
    )
