import numpy as np
import json
import gzip
import os
import jsonlines

def write_jsonl(filename, data, append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = 'a'
    else:
        mode = 'w'
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with jsonlines.open(filename, mode) as fp:
            for x in data:
                fp.write(x)
def stream_jsonl(filename):
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r", encoding='utf-8') as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)

def read_and_norm():
    input_file = "./data/all_results.jsonl"
    data = []
    data_label = []
    for id, sample in enumerate(stream_jsonl(input_file)):
        data.append([sample["st_de"], sample["difference"], sample["peak"]])
        data_label.append([sample["label"]])
    
    data_array = np.array(data)
    data_label_array = np.array(data_label)
    
    min_values = np.min(data_array, axis=0)
    max_values = np.max(data_array, axis=0)
    normalized_array = (data_array - min_values) / (max_values - min_values)
    normalized_array = np.round(normalized_array, decimals=2)
    
    return normalized_array, data_label_array, min_values, max_values
def main():
    normalized_array, data_label_array, min_values, max_values = read_and_norm()
    array_length = len(normalized_array)
    test_set_size = int(0.2 * array_length)

    test_indices = np.random.choice(array_length, test_set_size, replace=False)
    test_set = normalized_array[test_indices]
    test_labels = data_label_array[test_indices]
    test_labels = test_labels.flatten()

    train_set = np.delete(normalized_array, test_indices, axis=0)
    train_labels = np.delete(data_label_array, test_indices)
    
    return train_set, train_labels, test_set, test_labels, min_values, max_values

if __name__ == "__main__":
    main()
    # print(normalized_array.tolist())
