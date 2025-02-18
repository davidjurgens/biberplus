from datasets import load_dataset, concatenate_datasets, DatasetDict


def combine_and_save_datasets(split_files, out_path):
    datasets = []
    for file_path in split_files:
        dataset = load_dataset("json", data_files=file_path)
        datasets.append(dataset['train'])

    combined = concatenate_datasets(datasets)
    combined = combined.shuffle()
    new_dataset = DatasetDict()
    new_dataset['train'] = combined

    new_dataset.save_to_disk(out_path)


def load_split_files(input_directories, split):
    return [f"{input_directory}/{split}.jsonl" for input_directory in input_directories]


if __name__ == '__main__':
    input_directories = [
        '/shared/3/projects/hiatus/tagged_data/amazon/',
        '/shared/3/projects/hiatus/tagged_data/reddit/',
        '/shared/3/projects/hiatus/tagged_data/book3corpus/',
        '/shared/3/projects/hiatus/tagged_data/wiki/',
        '/shared/3/projects/hiatus/tagged_data/wiki_discussions/',
        '/shared/3/projects/hiatus/tagged_data/realnews/',
        '/shared/3/projects/hiatus/tagged_data/gmane/'
    ]

    for split in ['train', 'dev', 'test']:
        print(f"Working on {split}")
        split_files = load_split_files(input_directories, split)
        out_path = '/shared/3/projects/hiatus/tagged_data/sampled_dataset/'
        combine_and_save_datasets(split_files, f"{out_path}{split}")
