import os.path


## parse dataset

def parse_parquet_dataset(data_files: str = "./data/fineweb-2/data/ita_Latn/test/*.parquet"):
    # If you want to use the dataset immediately and efficiently stream the data as you iterate over the dataset
    def load_samples():
        from datasets import load_dataset
        fw = load_dataset(
            "parquet",  # format
            data_files=data_files,  # point to local files
            split="train",
            streaming=True  # efficient memory usage
        )
        # return dataset
        return [{"id": entry["id"], "text": entry["text"]} for entry in fw]


    data = load_samples()
    return data

def dataset_to_jsonl(dataset: list[dict[str, str]], write_path: str):
    """
    Convert this dict into a jsonl: [{"id": entry["id"], "text": entry["text"]}]
    """
    import json
    with(open(write_path, "w", encoding="utf-8")) as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")


def split_dataset_train_test(dataset: list[dict[str, str]], write_dir_path: str = None):
    import random, os.path as path
    dataset_copy = dataset.copy()

    random.seed(0xeff293294aaaef32accac12312afbb4768594, version=2)
    random.shuffle(dataset_copy)
    random.shuffle(dataset_copy)

    dataset_train_split = dataset_copy[:int(len(dataset_copy) * 0.80)]
    dataset_test_split = dataset_copy[int(len(dataset_copy) * 0.80):]

    assert (len(dataset_train_split) + len(dataset_test_split)) == len(dataset)

    if write_dir_path is not None:
        dataset_to_jsonl(dataset_train_split, path.join(write_dir_path, 'train.jsonl'))
        dataset_to_jsonl(dataset_test_split, path.join(write_dir_path, 'test.jsonl'))

    return dataset_train_split, dataset_test_split

def load_dataset(ds_path: str = './data/fineweb-2/data/ita_Latn') -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    :param ds_path: directory path where train.jsonl and test.jsonl files are located
    :return: train and test datasets in form of list[dict[str, str]]
    """
    import os.path as path
    import json
    # train_d, test_d = [], []
    import zipfile

    if path.exists(ds_path) and os.path.isfile(ds_path):
        with(open(path.join(ds_path), "r", encoding="utf-8")) as f:
            return [json.loads(line) for line in f.readlines()], []

    if not path.exists(path.join(ds_path, 'train.jsonl')) or not path.exists(path.join(ds_path, 'test.jsonl')):
        with zipfile.ZipFile(path.join(ds_path, 'ita_ds.zip'), 'r') as zip_ref:
            zip_ref.extractall(ds_path)

    with(open(path.join(ds_path, 'train.jsonl'), "r", encoding="utf-8")) as f:
        train_d = [json.loads(line) for line in f.readlines()]
    with(open(path.join(ds_path, 'test.jsonl'), "r", encoding="utf-8")) as f:
        test_d = [json.loads(line) for line in f.readlines()]

    return train_d, test_d

def load_clean_mc4_dataset(ds_path: str = "./data/clean_mc4_it/clean_mc4_it.jsonl"):
    from os import path
    import json

    with(open(path.join(ds_path), "r", encoding="utf-8")) as f:
        ds = []
        for id, line in enumerate(f.readlines()):
            ds.append(json.loads(line))
            ds[-1]['id'] = id
        return ds, []

def load_tatoeba_dataset(ds_path: str = "./data2/t5_finetune/tatoeba_it_15k.tsv"):
    from os import path
    with(open(path.join(ds_path), "r", encoding="utf-8")) as f:
        return f.readlines()


if __name__ == '__main__':
    # dataset = parse_parquet_dataset()
    # dataset_to_jsonl(dataset, './data/fineweb-2/data/ita_Latn/ita_Latn.jsonl')
    # split_dataset_train_test(dataset, write_dir_path='./data/fineweb-2/data/ita_Latn/')
    ds = load_dataset('./data/fineweb-2/data/ita_Latn/')
    print('Train set split:', len(ds[0]))
    print('Test set split:', len(ds[1]))
    pass



