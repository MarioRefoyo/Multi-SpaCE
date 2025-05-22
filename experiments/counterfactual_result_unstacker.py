import os
import pickle


INDEX_START = 0
DATASETS = ["Cricket"]
MODEL = "inceptiontime_noscaling"
METHOD = "comte"
RESULTS_PATH = "./experiments/results"
PICKLE_FILE_NAME = "backup_counterfactuals_0-19"

if __name__ == "__main__":
    for dataset in DATASETS:
        dataset_results_path = f"{RESULTS_PATH}/{dataset}/{MODEL}/{METHOD}"

        # Open pickle file
        with open(f"{dataset_results_path}/{PICKLE_FILE_NAME}.pickle", "rb") as f:
            file = pickle.load(f)
            print("Data loaded")

        # Iterate over all counterfactuals and store them
        for i, counterfactual in enumerate(file):
            with open(f'{dataset_results_path}/{METHOD}_{INDEX_START+i:04d}-{INDEX_START+i:04d}.pickle', 'wb') as f:
                pickle.dump([counterfactual], f, pickle.HIGHEST_PROTOCOL)
