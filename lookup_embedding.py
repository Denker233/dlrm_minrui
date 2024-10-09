import matplotlib.pyplot as plt

# Data for number of indices per lookup and corresponding overall embedding ratios
indices_per_lookup = [100, 300, 500]
embedding_ratios = [34.80, 28.68, 25.45]

# Plotting the graph
plt.figure(figsize=(10, 6))
plt.plot(indices_per_lookup, embedding_ratios, marker='o', color='b', label='Overall Embedding Ratio')

# Adding labels and title
plt.xlabel('Number of Indices per Lookup')
plt.ylabel('Overall Embedding Ratio (%)')
plt.title('Overall Embedding Ratio vs Number of Indices per Lookup for d7525')
plt.grid(True)

# Show the plot
plt.show()
