from huggingface_hub import snapshot_download
folder = snapshot_download(
                "HuggingFaceFW/fineweb-2",
                repo_type="dataset",
                local_dir="./data/fineweb2/",
                # download the Ita filtered + test data
                allow_patterns=["data/ita_Latn/train/*", "data/ita_Latn/test/*"],)
