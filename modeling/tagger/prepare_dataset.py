import os
import json
import pandas as pd

from sklearn.model_selection import train_test_split

def get_dataset_paths():
    return {
        'amazon': '/shared/3/projects/hiatus/tagged_data/amazon/',
        'reddit': '/shared/3/projects/hiatus/tagged_data/reddit/',
        'book3corpus': '/shared/3/projects/hiatus/tagged_data/book3corpus/',
        'wiki': '/shared/3/projects/hiatus/tagged_data/wiki/',
        'wiki_discussions': '/shared/3/projects/hiatus/tagged_data/wiki_discussions/',
        'realnews': '/shared/3/projects/hiatus/tagged_data/realnews/',
        'gmane': '/shared/3/projects/hiatus/tagged_data/gmane/'
    }

def get_biber_features():
    return [
        "BIN_QUAN_mean", "BIN_QUAN_std",
        "BIN_QUPR_mean", "BIN_QUPR_std",
        "BIN_AMP_mean", "BIN_AMP_std",
        "BIN_PASS_mean", "BIN_PASS_std",
        "BIN_XX0_mean", "BIN_XX0_std",
        "BIN_JJ_mean", "BIN_JJ_std",
        "BIN_BEMA_mean", "BIN_BEMA_std",
        "BIN_CAUS_mean", "BIN_CAUS_std",
        "BIN_CONC_mean", "BIN_CONC_std",
        "BIN_COND_mean", "BIN_COND_std",
        "BIN_CONJ_mean", "BIN_CONJ_std",
        "BIN_CONT_mean", "BIN_CONT_std",
        "BIN_DPAR_mean", "BIN_DPAR_std",
        "BIN_DWNT_mean", "BIN_DWNT_std",
        "BIN_EX_mean", "BIN_EX_std",
        "BIN_FPP1_mean", "BIN_FPP1_std",
        "BIN_GER_mean", "BIN_GER_std",
        "BIN_RB_mean", "BIN_RB_std",
        "BIN_PIN_mean", "BIN_PIN_std",
        "BIN_INPR_mean", "BIN_INPR_std",
        "BIN_TO_mean", "BIN_TO_std",
        "BIN_NEMD_mean", "BIN_NEMD_std",
        "BIN_OSUB_mean", "BIN_OSUB_std",
        "BIN_PASTP_mean", "BIN_PASTP_std",
        "BIN_VBD_mean", "BIN_VBD_std",
        "BIN_PHC_mean", "BIN_PHC_std",
        "BIN_PIRE_mean", "BIN_PIRE_std",
        "BIN_PLACE_mean", "BIN_PLACE_std",
        "BIN_POMD_mean", "BIN_POMD_std",
        "BIN_PRMD_mean", "BIN_PRMD_std",
        "BIN_WZPRES_mean", "BIN_WZPRES_std",
        "BIN_VPRT_mean", "BIN_VPRT_std",
        "BIN_PRIV_mean", "BIN_PRIV_std",
        "BIN_PIT_mean", "BIN_PIT_std",
        "BIN_PUBV_mean", "BIN_PUBV_std",
        "BIN_SPP2_mean", "BIN_SPP2_std",
        "BIN_SMP_mean", "BIN_SMP_std",
        "BIN_SERE_mean", "BIN_SERE_std",
        "BIN_STPR_mean", "BIN_STPR_std",
        "BIN_SUAV_mean", "BIN_SUAV_std",
        "BIN_SYNE_mean", "BIN_SYNE_std",
        "BIN_TPP3_mean", "BIN_TPP3_std",
        "BIN_TIME_mean", "BIN_TIME_std",
        "BIN_NOMZ_mean", "BIN_NOMZ_std",
        "BIN_BYPA_mean", "BIN_BYPA_std",
        "BIN_PRED_mean", "BIN_PRED_std",
        "BIN_TOBJ_mean", "BIN_TOBJ_std",
        "BIN_TSUB_mean", "BIN_TSUB_std",
        "BIN_THVC_mean", "BIN_THVC_std",
        "BIN_NN_mean", "BIN_NN_std",
        "BIN_DEMP_mean", "BIN_DEMP_std",
        "BIN_DEMO_mean", "BIN_DEMO_std",
        "BIN_WHQU_mean", "BIN_WHQU_std",
        "BIN_EMPH_mean", "BIN_EMPH_std",
        "BIN_HDG_mean", "BIN_HDG_std",
        "BIN_WZPAST_mean", "BIN_WZPAST_std",
        "BIN_THAC_mean", "BIN_THAC_std",
        "BIN_PEAS_mean", "BIN_PEAS_std",
        "BIN_ANDC_mean", "BIN_ANDC_std",
        "BIN_PRESP_mean", "BIN_PRESP_std",
        "BIN_PROD_mean", "BIN_PROD_std",
        "BIN_SPAU_mean", "BIN_SPAU_std",
        "BIN_SPIN_mean", "BIN_SPIN_std",
        "BIN_THATD_mean", "BIN_THATD_std",
        "BIN_WHOBJ_mean", "BIN_WHOBJ_std",
        "BIN_WHSUB_mean", "BIN_WHSUB_std",
        "BIN_WHCL_mean", "BIN_WHCL_std",
        "BIN_ART_mean", "BIN_ART_std",
        "BIN_AUXB_mean", "BIN_AUXB_std",
        "BIN_CAP_mean", "BIN_CAP_std",
        "BIN_SCONJ_mean", "BIN_SCONJ_std",
        "BIN_CCONJ_mean", "BIN_CCONJ_std",
        "BIN_DET_mean", "BIN_DET_std",
        "BIN_EMOJ_mean", "BIN_EMOJ_std",
        "BIN_EMOT_mean", "BIN_EMOT_std",
        "BIN_EXCL_mean", "BIN_EXCL_std",
        "BIN_HASH_mean", "BIN_HASH_std",
        "BIN_INF_mean", "BIN_INF_std",
        "BIN_UH_mean", "BIN_UH_std",
        "BIN_NUM_mean", "BIN_NUM_std",
        "BIN_LAUGH_mean", "BIN_LAUGH_std",
        "BIN_PRP_mean", "BIN_PRP_std",
        "BIN_PREP_mean", "BIN_PREP_std",
        "BIN_NNP_mean", "BIN_NNP_std",
        "BIN_QUES_mean", "BIN_QUES_std",
        "BIN_QUOT_mean", "BIN_QUOT_std",
        "BIN_AT_mean", "BIN_AT_std",
        "BIN_SBJP_mean", "BIN_SBJP_std",
        "BIN_URL_mean", "BIN_URL_std",
        "BIN_WH_mean", "BIN_WH_std",
        "BIN_INDA_mean", "BIN_INDA_std",
        "BIN_ACCU_mean", "BIN_ACCU_std",
        "BIN_PGAS_mean", "BIN_PGAS_std",
        "BIN_CMADJ_mean", "BIN_CMADJ_std",
        "BIN_SPADJ_mean", "BIN_SPADJ_std",
        "BIN_X_mean", "BIN_X_std"
    ]

def create_dataset(output_file, dataset_paths, biber_features):
    with open(output_file, 'w') as jsonl_file:
        for name, corpus_path in dataset_paths.items():
            corpus_fp = os.path.join(corpus_path, 'corpus.jsonl')
            line_count = 0
            valid_count = 0
            
            print(f"\nProcessing {corpus_fp}")
            with open(corpus_fp, 'r') as file:
                for line in file:
                    line_count += 1
                    if line_count % 1000000 == 0:
                        print(f"Processed {line_count:,} lines...")
                    
                    data = json.loads(line)
                    text = data.get('fullText', '')
                    binary_encodings = data.get('biber_tagged', {}).get('binary', [])

                    if not text or not binary_encodings:
                        continue

                    valid_count += 1
                    output_obj = {
                        'text': text,
                        'features': dict(zip(biber_features, binary_encodings))
                    }
                    jsonl_file.write(json.dumps(output_obj) + '\n')
            
            print(f"\nDataset: {name}")
            print(f"Total lines: {line_count:,}")
            print(f"Valid lines: {valid_count:,}")
            print(f"Filtered lines: {line_count - valid_count:,}\n")

def read_and_shuffle(file_path, chunksize=50000):
    chunks = []
    for chunk in pd.read_json(file_path, lines=True, chunksize=chunksize):
        chunks.append(chunk.sample(frac=1))
    return pd.concat(chunks).reset_index(drop=True)
  
  
def save_to_jsonl(df, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    df.to_json(file_path, orient='records', lines=True)

def split_and_save_data(input_file, chunk_size=50000):
    print(f"Processing data in chunks of {chunk_size:,}...")
    
    # Initialize file writers
    train_file = '/shared/3/projects/hiatus/tagged_data/binary_train.jsonl'
    dev_file = '/shared/3/projects/hiatus/tagged_data/binary_dev.jsonl'
    test_file = '/shared/3/projects/hiatus/tagged_data/binary_test.jsonl'
    
    # Create directories if they don't exist
    for file_path in [train_file, dev_file, test_file]:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Process in chunks
    with open(train_file, 'w') as train_f, \
         open(dev_file, 'w') as dev_f, \
         open(test_file, 'w') as test_f:
        
        for chunk in pd.read_json(input_file, lines=True, chunksize=chunk_size):
            # Shuffle the chunk
            chunk = chunk.sample(frac=1, random_state=42)
            
            # Calculate split indices for 80/10/10
            n = len(chunk)
            train_idx = int(n * 0.8)
            dev_idx = int(n * 0.9)
            
            # Split the chunk
            train_chunk = chunk.iloc[:train_idx]
            dev_chunk = chunk.iloc[train_idx:dev_idx]
            test_chunk = chunk.iloc[dev_idx:]
            
            # Write to respective files
            train_chunk.to_json(train_f, orient='records', lines=True)
            dev_chunk.to_json(dev_f, orient='records', lines=True)
            test_chunk.to_json(test_f, orient='records', lines=True)
    
    print("Data splitting and saving completed.")

def main():
    output_file = '/shared/3/projects/hiatus/tagged_data/binary_multi_label_dataset.jsonl'
    # dataset_paths = get_dataset_paths()
    # biber_features = get_biber_features()
    
    # create_dataset(output_file, dataset_paths, biber_features)
    
    split_and_save_data(output_file)
    
    print("Data splitting and saving completed.")

if __name__ == "__main__":
    main()