import json

def downsample_and_save(input_file, output_file, sample_rate=0.05):
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

# Create downsampled versions
base_path = '/shared/3/projects/hiatus/tagged_data'
print("Starting downsampling process...")

for sample_rate in [0.01]:  # 5% and 1% sampling
    percentage = int(sample_rate * 100)
    for dataset in ['train', 'dev', 'test']:
        input_file = f'{base_path}/binary_{dataset}.jsonl'
        output_file = f'{base_path}/binary_{dataset}_{percentage}pct.jsonl'
        total_read, total_written = downsample_and_save(input_file, output_file, sample_rate=sample_rate)

print("\nDownsampling complete for all files!")
