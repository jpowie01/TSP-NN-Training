from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.customxml import NetworkWriter

import time
from graph import Graph, convert_output

# Configuration
N = 5
GRAPHS = 250
RANDOM_TESTS = 100
HIDDENLAYERS = 5
LEARNINGRATE = 0.015
MOMENTUM = 0.85
MINUTES = 3 * 60
ITERATIONS = 999999999
MAX_ERROR = 0.000001
WEIGHTDECAY = 0.0


# Data
trainingData = SupervisedDataSet(N*N, N*N)
trainingDataInput = []
trainingDataOutput = []

# Prepare training data
print('Preparing data...')
allGraphs = []
for i in range(GRAPHS):
    print(str(i) + ' out of ' + str(GRAPHS))
    graph = Graph(N)
    graph.compute_tsp()
    allGraphs.append(graph)
    input = ()
    for j in range(N):
        for k in range(N):
            input += (graph.each_to_each[j][k], )
    output = ()
    for j in range(N):
        for k in range(N):
            if graph.order_tsp[j] == k:
                output += (1, )
            else:
                output += (0, )
    trainingData.addSample(input, output)
    trainingDataInput.append(input)
    trainingDataOutput.append(output)
print('Done')

# Print header
print('====================================')
print('====================================')
print('  GRAPHS:', GRAPHS)
print('  HIDDENLAYERS:', HIDDENLAYERS)
print('  LEARNINGRATE:', LEARNINGRATE)
print('  MOMENTUM:', MOMENTUM)
print('====================================')
print('====================================')

# Prepare recurrent network
net = RecurrentNetwork()

# Add layers
net.addInputModule(LinearLayer(N*N, name='in'))
for layer in range(1, HIDDENLAYERS+1):
    net.addModule(SigmoidLayer(N*N, name='hidden' + str(layer)))
net.addOutputModule(TanhLayer(N*N, name='out'))

# Add connections between layers
net.addConnection(FullConnection(net['in'], net['hidden1']))
for layer in range(1, HIDDENLAYERS):
    net.addConnection(FullConnection(net['hidden'+str(layer)], net['hidden'+str(layer+1)]))
net.addConnection(FullConnection(net['hidden' + str(HIDDENLAYERS)], net['out']))
net.addRecurrentConnection(FullConnection(net['hidden' + str(HIDDENLAYERS)], net['hidden1']))
net.sortModules()

# Trainer
trainer = BackpropTrainer(net, dataset=trainingData, learningrate=LEARNINGRATE, momentum=MOMENTUM, weightdecay=WEIGHTDECAY)

# Preparations
print('Training...')
start = time.time()
one_minute = time.time()
iteration = 1

# Train network many times
for epoch in range(0, ITERATIONS):
    # Train
    error = trainer.train()
    print(str(epoch) + '\t' + str(time.time() - start) + '\t' + str(error))

    # Print results once per 3 minutes
    if time.time() - one_minute >= 3 * 60:
        # Save network
        NetworkWriter.writeToFile(net, 'results/neural_network_'+str(iteration)+'.xml')
        iteration += 1
        one_minute = time.time()

        # Print result for training set
        print('')
        print('===================================')
        print('Training set:')
        print('===================================')
        for i in range(GRAPHS):
            input = trainingDataInput[i]
            output = trainingDataOutput[i]
            net_output = net.activate(input)
            order = convert_output(net_output, N)

            # Print result
            print('Test no.' + str(i) + ':')
            for j in range(N):
                print('%1.4f\t%1.4f' % (allGraphs[i].graph[j][0], allGraphs[i].graph[j][1]))
            for y in range(N):
                for x in range(N):
                    print('%1.4f' % output[y * N + x], end=' ')
                print('   ', end='')
                for x in range(N):
                    print('%1.4f' % net_output[y * N + x], end=' ')
                print('')

        # Print result for random generated set
        print('')
        print('===================================')
        print('Random set:')
        print('===================================')
        for i in range(RANDOM_TESTS):
            graph = Graph(N)
            graph.compute_tsp()
            input = ()
            for j in range(N):
                for k in range(N):
                    input += (graph.each_to_each[j][k], )
            output = ()
            for j in range(N):
                for k in range(N):
                    if graph.order_tsp[j] == k:
                        output += (1, )
                    else:
                        output += (0, )
            net_output = net.activate(input)
            order = convert_output(net_output, N)

            # Print result
            print('Test no.' + str(i) + ':')
            for i in range(N):
                print('%1.4f\t%1.4f' % (graph.graph[i][0], graph.graph[i][1]))
            for y in range(N):
                for x in range(N):
                    print('%1.4f' % output[y * N + x], end=' ')
                print('   ', end='')
                for x in range(N):
                    print('%1.4f' % net_output[y * N + x], end=' ')
                print('')

    # End learning
    if error < MAX_ERROR or time.time() - start > MINUTES * 60:
        break
