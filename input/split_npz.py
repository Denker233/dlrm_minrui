import numpy as np

# Load the original .npz file without modifying it
# data = np.load("kaggleAdDisplayChallenge_processed.npz")
data = np.load("./input/original_preprocessed_data.npz")
# np.savez("original_preprocessed_data.npz_2", **data)

# Calculate 1/7 of the total number of keys (arrays) in the .npz file
subset_data = {}
fraction = 7  # We are taking 1/7 of the data

# Loop through each key and take the first 1/7 of entries
for key in data.keys():
    item = data[key]
    print(item)
    # Check if item is a dictionary-like structure
    if isinstance(item, np.ndarray) and item.dtype == 'O':  # 'O' dtype indicates objects
        nested_data = item.item()  # Convert to a dictionary if stored as object
        subset_nested_data = {}

        # Loop through each nested key and reduce the entries
        for nested_key, array in nested_data.items():
            # Ensure that the nested item is an array
            if isinstance(array, np.ndarray):
                # Calculate the subset size for the array
                subset_size = len(array) // fraction
                subset_nested_data[nested_key] = array[:subset_size]  # Select the first 1/7 of entries

        # Store the subset of nested data under the top-level key
        subset_data[key] = subset_nested_data
    else:
        # If it's a direct array, reduce it directly
        if isinstance(item, np.ndarray):
            subset_size = len(item) // fraction
            subset_data[key] = item[:subset_size]  # Select the first 1/7 of entries


# Create a dictionary to store only the selected data (1/7 of the content)
# subset_data = {key: data[key] for key in subset_keys}

# Save the subset to a new .npz file
np.savez("./input/kaggleAdDisplayChallenge_processed.npz", **subset_data)
