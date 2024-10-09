import matplotlib.pyplot as plt

# Data for minibatch sizes and corresponding overall embedding ratios
minibatch_sizes = [2048, 1024, 512, 256, 128, 64]
embedding_ratios = [34.80, 43.30, 50.87, 49.49, 51.73, 45.76]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(minibatch_sizes, embedding_ratios, marker='o', color='b', label='Overall Embedding Ratio')

# Adding labels and title
plt.xlabel('Minibatch Size')
plt.ylabel('Overall Embedding Ratio (%)')
plt.title('Overall Embedding Ratio vs Minibatch Size (using d7525)')
plt.grid(True)

# Show the plot
plt.show()
