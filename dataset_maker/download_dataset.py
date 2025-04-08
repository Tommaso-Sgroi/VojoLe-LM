from huggingface_hub import snapshot_download

def download_full_dataset():
    folder = snapshot_download(
                    "HuggingFaceFW/fineweb-2",
                    repo_type="dataset",
                    local_dir="./data/fineweb2/",
                    # download the Ita filtered + test data
                    allow_patterns=["data/ita_Latn/train/*", "data/ita_Latn/test/*"],)
    return folder

TRAIN_SPLIT, TEST_SPLIT = 'train', 'test'
def get_dataset_stream(split:str):
    from datasets import load_dataset
    # get Croatian data
    fw = load_dataset("HuggingFaceFW/fineweb-2", name="ita_Latn", split=split, streaming=True)
    return fw
