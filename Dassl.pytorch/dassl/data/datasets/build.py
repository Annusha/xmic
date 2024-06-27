from dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg, split='train', dataset_name=None):
    avai_datasets = DATASET_REGISTRY.registered_names()
    if dataset_name is None:
        dataset_name = cfg.DATASET.NAME
    check_availability(dataset_name, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(dataset_name))
    if 'Ego4D' in dataset_name:
        return DATASET_REGISTRY.get(dataset_name)(cfg, split)
    else:
        return DATASET_REGISTRY.get(dataset_name)(cfg)
