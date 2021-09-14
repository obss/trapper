from pathlib import Path
from typing import Iterable, Union

from trapper.data import DatasetReader
from trapper.data.dataset_readers import IndexedInstance


@DatasetReader.register("dummy_dataset_reader_inside_package2")
class DummyDatasetReader2(DatasetReader):
    def _read(self, file_path: Union[Path, str]) -> Iterable[IndexedInstance]:
        pass

    def text_to_instance(self, *inputs) -> IndexedInstance:
        pass
