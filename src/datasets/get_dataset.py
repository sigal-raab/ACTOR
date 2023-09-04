def get_dataset(name="ntu13"):
    if name == "ntu13":
        from .ntu13 import NTU13
        return NTU13
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanact12_oneclass":
        from .humanact12poses_oneclass import HumanAct12Poses
        return HumanAct12Poses
    elif name == "mixamo":
        from .mixamoposes import MixamoPoses
        return MixamoPoses
    elif name == "mixamo_euler":
        from .mixamoposes_euler import MixamoPoses_Euler
        return MixamoPoses_Euler


def get_datasets(parameters):
    name = parameters["dataset"]

    DATA = get_dataset(name)
    dataset = DATA(split="train", **parameters)

    train = dataset

    # test: shallow copy (share the memory) but set the other indices
    from copy import copy
    test = copy(train)
    test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets
