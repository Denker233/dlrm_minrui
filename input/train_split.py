
import os

def get_file_size_in_gb(file_path):
    
    if os.path.exists(file_path):
        size_in_bytes = os.path.getsize(file_path)
        return size_in_bytes / (1024 ** 3)  # Convert bytes to GB
    return None

def split_file(input_filename, output_filename, split_factor):
    """
    Splits a file into a portion based on the split factor.
    :param split_factor: The denominator for splitting the file (e.g., 7 means 1/7 of the file).
    """
    with open(input_filename, 'r') as input_file, open(output_filename, 'w') as output_file:
        # Count the total number of lines
        total_lines = sum(1 for _ in input_file)

        # Rewind the input file to the beginning
        input_file.seek(0)

        # Calculate the number of lines to write based on the split factor
        lines_to_write = total_lines // split_factor

        # Write the calculated number of lines to the new file
        for line_num, line in enumerate(input_file, start=1):
            if line_num <= lines_to_write:
                output_file.write(line)
            else:
                break


# Specify the split factor and file names
try:
    split_factor = int(input("Enter the split factor (e.g., 7 for 1/7th): "))
    if split_factor <= 0:
        raise ValueError("Split factor must be greater than 0.")
except ValueError as e:
    print(f"Invalid input: {e}")
split_file('train_original.txt', 'train.txt', split_factor)
split_file('test_original.txt', 'test.txt', split_factor)

for file in ['train.txt', 'test.txt']:
    size_in_gb = get_file_size_in_gb(file)
    if size_in_gb is not None:
        print(f"File: {file}, Size: {size_in_gb:.2f} GB")
    else:
        print(f"File: {file} does not exist.")
