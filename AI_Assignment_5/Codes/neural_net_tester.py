# Unit tester for neural_net.py
#
import sys

from neural_net import train, test,\
     make_neural_net_basic,\
     make_neural_net_two_layer,\
     make_neural_net_challenging,\
     make_neural_net_large

from neural_net_data import simple_data_sets,\
     harder_data_sets,\
     challenging_data_sets,\
     two_moons_data_set,\
     all_data_sets

def main(neural_net_func, data_sets, rate=1.0, max_iterations=10000):
    verbose = True
    for name, training_data, test_data in data_sets:
        print("-"*40)
        print("Training on %s data" %(name))
        nn = neural_net_func()
        train(nn, training_data, rate=rate, max_iterations=max_iterations,
              verbose=verbose)
        print("Trained weights:")
        for w in nn.weights:
            print("Weight '%s': %f"%(w.get_name(),w.get_value()))
        print("Testing on %s test-data" %(name))
        result = test(nn, test_data, verbose=verbose)
        print("Accuracy: %f"%(result))

if __name__=="__main__":
    test_names = ["simple"]
    if len(sys.argv) > 1:
        test_names = sys.argv[1:]

    for test_name in test_names:
        if test_name == "simple":
            # these test simple logical configurations
            main(make_neural_net_basic,
                 simple_data_sets)

        elif test_name == "two_layer":
            # these test cases are slightly harder
            main(make_neural_net_two_layer,
                 simple_data_sets + harder_data_sets)

        elif test_name == "challenging":
            # these tests require a more complex architecture.
            main(make_neural_net_challenging, challenging_data_sets)

        elif test_name == "two_moons":
            # this dataset illustrates the overfitting problem
            main(make_neural_net_large, two_moons_data_set, max_iterations=1000)

        else:
            print("unrecognized test name %s" %(test_name))
