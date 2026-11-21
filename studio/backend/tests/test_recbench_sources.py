from pathlib import Path
import pytest
from studio.backend.executor import provision
from studio.backend.executor.recbench_sources import PUBLISHED_PARQUET
from utils.config_init import DataInit


@pytest.mark.parametrize('processor', sorted(PUBLISHED_PARQUET))
def test_official_downloads_use_existing_splits(processor, tmp_path, monkeypatch):
    monkeypatch.setattr(DataInit, 'get', lambda *a, **kw: '')
    monkeypatch.setattr(provision, 'DOWNLOADS', tmp_path)
    calls = []
    def download(name, urls):
        calls.append((name, urls))
        return tmp_path / name
    monkeypatch.setattr(provision, 'download', download)
    path, published = provision.raw_recbench(processor)
    assert published
    assert path == tmp_path / 'legommenders' / processor
    assert {Path(name).stem for name, _ in calls} == {'items', 'users', 'train', 'valid', 'test'}
    assert all(PUBLISHED_PARQUET[processor][Path(name).stem] in urls[0] for name, urls in calls)


def test_local_published_splits_need_no_finetune_or_validation_list(tmp_path, monkeypatch):
    for name in ('items', 'users', 'train', 'valid', 'test'):
        (tmp_path / f'{name}.parquet').touch()
    monkeypatch.setattr(DataInit, 'get', lambda *a, **kw: str(tmp_path))
    assert provision.raw_recbench('booksrb') == (tmp_path, True)


def test_official_cache_does_not_reuse_legacy_split_cache():
    data = {'item': {'inputs': ['item_id']}}
    assert 'published-v1' in provision.prepared_directory(data, 'booksrb', 'cmn').name
