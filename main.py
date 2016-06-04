from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import TanhLayer, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain.tools.customxml import NetworkWriter

import time
from graph import Graph

# Configuration
N = 6
GRAPHS = 10000
HIDDENLAYERS = 6
LEARNINGRATE = 0.2
MOMENTUM = 0.3
MINUTES = 3 * 60
ITERATIONS = 999999999
MAX_ERROR = 0.000001
WEIGHTDECAY = 0.0


# Data
trainingData = SupervisedDataSet(N*N, N*N)
trainingDataInput = []
trainingDataOutput = []

# Prepare training data
print('Generating data...')
allGraphs = []
for i in range(GRAPHS):
    # Prepare graph
    graph = Graph(N)
    graph.compute_tsp()
    allGraphs.append(graph)

    # Create input
    sample_input = ()
    for j in range(N):
        for k in range(N):
            sample_input += (graph.each_to_each[j][k], )

    # Create output
    sample_output = ()
    for j in range(N):
        for k in range(N):
            if graph.order_tsp[j] == k:
                sample_output += (1, )
            else:
                sample_output += (0, )

    # Save sample
    trainingData.addSample(sample_input, sample_output)
    trainingDataInput.append(sample_input)
    trainingDataOutput.append(sample_output)

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
trainer = BackpropTrainer(net,
                          dataset=trainingData,
                          learningrate=LEARNINGRATE,
                          momentum=MOMENTUM,
                          weightdecay=WEIGHTDECAY)

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

    # Save network once per 30 seconds
    if time.time() - one_minute >= 30:
        # Save network
        NetworkWriter.writeToFile(net, 'networks/neural_network_'+str(iteration).zfill(4)+'.xml')
        print('Network saved to networks/neural_network_'+str(iteration).zfill(4)+'.xml')
        iteration += 1
        one_minute = time.time()

    # End learning
    if error < MAX_ERROR or time.time() - start > MINUTES * 60:
        break
