import csv
import random



def load_datasets_from_csv(path):
    datasets = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        columns = list(zip(*reader))  # Transpose rows to columns
        for col in columns:
            try:
                datasets.append([int(val) for val in col])
            except ValueError:
                continue  # skip non-integer columns (like headers)
    return datasets

def generate_starting_list(n, min_val, max_val):
    return [random.randint(min_val, max_val) for _ in range(n)]

def export_results_to_csv(results, filename="results.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Steps", "Time (s)"])
        for algo, stats in results.items():
            writer.writerow([algo, stats.get("Steps", ""), stats.get("Time (s)", "")])