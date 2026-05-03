import pickle
import os
import re

# DATASET = 'UWaveGestureLibrary'
DATASET = 'ECG200'
# PREFIX_FILES_NAME = "fitness_evolutions_v6_mut2"
PREFIX_FILES_NAME = "subspace"


PARTIAL_RESULT_RE = re.compile(r"^(?P<prefix>.+)_(?P<start>\d+)-(?P<end>\d+)\.pickle$")


def list_partial_result_files(dataset, model_to_explain_name, prefix_files_name, path="."):
    folder = f"{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}"
    partial_files = []
    for filename in os.listdir(folder):
        match = PARTIAL_RESULT_RE.match(filename)
        if match is None:
            continue
        if match.group("prefix") != prefix_files_name:
            continue
        partial_files.append(
            {
                "filename": filename,
                "start": int(match.group("start")),
                "end": int(match.group("end")),
            }
        )
    partial_files.sort(key=lambda item: item["start"])
    return partial_files


def infer_fragmentation_samples(dataset, model_to_explain_name, prefix_files_name, path="."):
    partial_files = list_partial_result_files(dataset, model_to_explain_name, prefix_files_name, path=path)
    if not partial_files:
        raise FileNotFoundError(
            f"No partial result files found for {dataset}/{model_to_explain_name}/{prefix_files_name}."
        )

    first_file = partial_files[0]
    last_file = partial_files[-1]
    fragmentation = first_file["end"] - first_file["start"] + 1
    total = last_file["end"] + 1
    return fragmentation, total


def concatenate_and_store_partial_results(dataset, model_to_explain_name, prefix_files_name, suffixes_list, path='.'):
    all_files_list = []
    for suffix in suffixes_list:
        with open(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/{prefix_files_name}_{suffix}.pickle', 'rb') as f:
            part_file = pickle.load(f)

        all_files_list = all_files_list + part_file

    # Store concatenated file
    with open(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/counterfactuals.pickle', 'wb') as f:
        pickle.dump(all_files_list, f, pickle.HIGHEST_PROTOCOL)


def remove_partial_files(dataset, model_to_explain_name, prefix_files_name, path='.'):
    partial_files = list_partial_result_files(dataset, model_to_explain_name, prefix_files_name, path=path)
    for partial_file in partial_files:
        os.remove(f'{path}/{dataset}/{model_to_explain_name}/{prefix_files_name}/{partial_file["filename"]}')


def concatenate_result_files(dataset, model_to_explain_name, prefix_file_name):
    partial_files = list_partial_result_files(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        path="./experiments/results",
    )
    if not partial_files:
        raise FileNotFoundError(
            f"No partial result files found for {dataset}/{model_to_explain_name}/{prefix_file_name}."
        )

    suffixes_list = [f"{item['start']:04d}-{item['end']:04d}" for item in partial_files]
    concatenate_and_store_partial_results(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        suffixes_list,
        path='./experiments/results'
    )
    # Remove the temporal files
    remove_partial_files(
        dataset,
        model_to_explain_name,
        prefix_file_name,
        path='./experiments/results'
    )


if __name__ == "__main__":
    concatenate_result_files(DATASET, PREFIX_FILES_NAME)
