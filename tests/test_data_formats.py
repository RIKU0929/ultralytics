# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.data.base import BaseDataset
from ultralytics.data.utils import IMG_FORMATS


class DummyDataset(BaseDataset):
    def get_labels(self):
        return [{} for _ in self.im_files]

    def build_transforms(self, hyp):
        return lambda x, y=None: (x, y)


def test_get_img_files_includes_fits(tmp_path):
    fits_file = tmp_path / "example.fits"
    fts_file = tmp_path / "example.fts"
    jpg_file = tmp_path / "example.jpg"
    text_file = tmp_path / "notes.txt"

    for file in (fits_file, fts_file, jpg_file, text_file):
        file.touch()

    dataset = DummyDataset(img_path=str(tmp_path), cache=False, augment=False)

    assert {"fits", "fts"}.issubset(IMG_FORMATS)
    assert str(fits_file) in dataset.im_files
    assert str(fts_file) in dataset.im_files
    assert str(text_file) not in dataset.im_files
