import os

from datasets import load_dataset


def create_datasets(datasets, folder):
    for dataset in datasets:
        print(f"Creating {dataset} dataset...")
        inpath = os.path.join(folder, f"{dataset}.jsonl")
        outpath = os.path.join(folder, dataset)
        create_dataset(inpath, outpath)


def create_dataset(inpath, outpath):
    dataset = load_dataset("json", data_files=inpath)
    dataset.save_to_disk(outpath)


if __name__ == "__main__":
    folder = "/shared/3/projects/hiatus/tagged_data/mlm_finetuning/"
    datasets = ["train", "dev", "test"]

    create_datasets(datasets, folder)
