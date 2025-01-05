import json
import random
import os

def downsample_by_rate(input_file, output_file, sample_rate=0.05):
    print(f"\nDownsampling {input_file} to {output_file} at {sample_rate*100}% sample rate")
    step = int(1/sample_rate)  # e.g., 20 for 5% sampling
    
    lines_read = 0
    lines_written = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        # Process file line by line
        for i, line in enumerate(fin):
            lines_read += 1
            try:
                json.loads(line.strip())
                
                if i > 0 and i % 1000000 == 0:  # Log every million lines
                    print(f"Processed {i:,} lines from {input_file}")
                if i % step == 0:  # Take every nth line
                    fout.write(line)
                    lines_written += 1
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON on line {i+1}")
                continue
    
    print(f"Completed: Read {lines_read:,} lines, wrote {lines_written:,} lines")
    return lines_read, lines_written

def sample_max_lines(datasets, output_dir, max_samples=1000000):
    base_paths = {
        'amazon': '/shared/3/projects/hiatus/tagged_data/amazon/',
        'reddit': '/shared/3/projects/hiatus/tagged_data/reddit/',
        'book3corpus': '/shared/3/projects/hiatus/tagged_data/book3corpus/',
        'wiki': '/shared/3/projects/hiatus/tagged_data/wiki/',
        'wiki_discussions': '/shared/3/projects/hiatus/tagged_data/wiki_discussions/',
        'realnews': '/shared/3/projects/hiatus/tagged_data/realnews/',
        'gmane': '/shared/3/projects/hiatus/tagged_data/gmane/'
    }
    
    # Create output files at the start
    train_file_name = os.path.join(output_dir, f'train_{max_samples}.jsonl')
    dev_file_name = os.path.join(output_dir, f'dev_{max_samples}.jsonl')
    test_file_name = os.path.join(output_dir, f'test_{max_samples}.jsonl')
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear/create the output files
    open(train_file_name, 'w').close()
    open(dev_file_name, 'w').close()
    open(test_file_name, 'w').close()

    total_written = 0
    
    for dataset in datasets:
        path = base_paths.get(dataset)
        if not path:
            print(f"Dataset {dataset} not found in base paths.")
            continue

        input_file = f'{path}corpus.jsonl'
        print(f"\nSampling up to {max_samples} lines from {input_file}")
        
        lines_read = 0
        lines_written = 0
        
        samples_for_dataset = []
        with open(input_file, 'r') as fin:
            for i, line in enumerate(fin):
                lines_read += 1
                try:
                    json.loads(line.strip())
                    samples_for_dataset.append(line)
                    lines_written += 1
                    if lines_written >= max_samples:
                        print(f"Reached maximum of {max_samples} samples for {dataset}")
                        break
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON on line {i+1}")
                    continue
        
        # Shuffle and split for this dataset
        random.shuffle(samples_for_dataset)
        dataset_size = len(samples_for_dataset)
        train_end = int(dataset_size * 0.8)
        dev_end = train_end + int(dataset_size * 0.1)

        # Append to the output files
        with open(train_file_name, 'a') as train_file:
            train_file.writelines(samples_for_dataset[:train_end])
        with open(dev_file_name, 'a') as dev_file:
            dev_file.writelines(samples_for_dataset[train_end:dev_end])
        with open(test_file_name, 'a') as test_file:
            test_file.writelines(samples_for_dataset[dev_end:])
            
        total_written += len(samples_for_dataset)
        print(f"Completed: Read {lines_read:,} lines, wrote {lines_written:,} lines for {dataset}")

    # Count final lines in each file
    train_count = sum(1 for _ in open(train_file_name, 'r'))
    dev_count = sum(1 for _ in open(dev_file_name, 'r'))
    test_count = sum(1 for _ in open(test_file_name, 'r'))

    print(f"Total samples: {total_written}, Train: {train_count}, Dev: {dev_count}, Test: {test_count}")
    return total_written, train_count, dev_count, test_count

def main():
    print("Starting downsampling process...")

    output_dir = '/shared/3/projects/hiatus/tagged_data/'
    datasets = ['amazon', 'reddit', 'book3corpus', 'wiki', 'wiki_discussions', 'realnews', 'gmane']
    total_samples, train_count, dev_count, test_count = sample_max_lines(datasets, output_dir, max_samples=1000000)
    print(f"Total samples: {total_samples}, Train: {train_count}, Dev: {dev_count}, Test: {test_count}")

    print("\nDownsampling complete for all files!")

if __name__ == "__main__":
    main()
