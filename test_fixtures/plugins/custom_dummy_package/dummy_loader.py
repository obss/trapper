from trapper.data import DatasetLoader


@DatasetLoader.register("dummy_dataset_loader_inside_package")
class DummyDatasetLoader(DatasetLoader):
    pass
