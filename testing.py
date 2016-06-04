from graph import Graph, convert_output
from pybrain.tools.customxml import NetworkReader
import glob

N = 6
RANDOM_TESTS = 2000

# Prepare data
print('Generating data...')
graphs = []
inputs = []
for i in range(RANDOM_TESTS):
    # Create graph
    graph = Graph(N)
    graph.compute_tsp()

    # Prepare input
    sample_input = ()
    for j in range(N):
        for k in range(N):
            sample_input += (graph.each_to_each[j][k],)

    # Save data
    graphs.append(graph)
    inputs.append(sample_input)

# Prepare list of all networks
print('Reading files...')
files = glob.glob("./networks/*.xml")
files.sort()

# Test each network
best_network_name = ''
best_network_error = 99999
for file_name in files:
    # Create network
    net = NetworkReader.readFrom(file_name)

    # Test network with graphs
    random_error_list = []
    for i in range(RANDOM_TESTS):
        # Activate network and get output
        net_output = net.activate(inputs[i])
        order = convert_output(net_output, N)

        # Original
        original_len = 0
        for j in range(N - 1):
            original_len += graphs[i].each_to_each[graphs[i].order_tsp_list[j]][graphs[i].order_tsp_list[j + 1]]
        original_len += graphs[i].each_to_each[graphs[i].order_tsp_list[N - 1]][graphs[i].order_tsp_list[0]]

        # Compute
        network_len = 0
        for j in range(N - 1):
            network_len += graphs[i].each_to_each[order[j]][order[j + 1]]
        network_len += graphs[i].each_to_each[order[N - 1]][order[0]]

        # Add diff to table
        random_error_list.append(abs(original_len - network_len) / original_len)

    # Print errors
    avg_random_error = sum(random_error_list) / float(len(random_error_list))
    if avg_random_error < best_network_error:
        best_network_error = avg_random_error
        best_network_name = file_name
    print(file_name, '=> Random error:', avg_random_error * 100, '%')

# Print best network
print('=================================')
print('=================================')
print('Best network:', best_network_name)
print('Error:', best_network_error * 100, '%')
print('=================================')
print('=================================')
