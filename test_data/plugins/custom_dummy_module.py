from trapper.data import DatasetLoader


@DatasetLoader.register("dummy_dataset_loader_inside_module")
class DummyDatasetLoader3(DatasetLoader):
    pass
