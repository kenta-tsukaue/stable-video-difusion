from datasets import load_dataset

def getData(config):
    config.dataset_name = "Nexdata/Emotional_Video_Data"
    dataset = load_dataset(config.dataset_name, split="train")
    return dataset