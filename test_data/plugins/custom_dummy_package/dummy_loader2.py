from trapper.data import DatasetLoader


@DatasetLoader.register("dummy_dataset_loader_inside_package2")
class DummyDatasetLoader2(DatasetLoader):
    pass
