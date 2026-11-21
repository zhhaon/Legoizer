import time
from unittest.mock import patch
from studio.backend import resources as r


def test_states_and_config():
    assert r.inspect('item.cnn')['state'] == 'unnecessary'
    assert r.inspect('item.externalhf')['state'] == 'configure'
    with patch.object(r, 'snapshot', return_value='/cached'):
        assert r.inspect('item.bertbase')['state'] == 'ready'
    with patch.object(r, 'snapshot', side_effect=OSError('missing')):
        assert r.inspect('item.bertbase')['state'] == 'missing'
    spec = r.get_component('dataset.mind')
    path, data = r.dataset_config(spec, {x.name:x.default for x in spec.parameters})
    assert data['item']['ut'] == 'data/mind/items'
    with patch.object(r.p, 'tables_available', return_value=False):
        assert r.inspect('dataset.mind')['state'] == 'missing'
    with patch.object(r.p, 'tables_available', return_value=True), patch.object(r, 'dataset_embeddings', return_value=[]):
        assert r.inspect('dataset.mind')['state'] == 'ready'


def test_acquisition_deduplicates_and_reports_failure():
    import subprocess
    import threading
    gate = threading.Event()
    def run(*args, **kwargs):
        gate.wait(2)
        return subprocess.CompletedProcess(args, 1, stdout='Download unavailable')
    with patch.object(r.p, 'tables_available', return_value=False), patch('subprocess.run', side_effect=run) as worker:
        first = r.acquire('dataset.mind')
        second = r.acquire('dataset.mind')
        assert first['state'] == second['state'] == 'downloading'
        gate.set()
        for _ in range(100):
            status = r.inspect('dataset.mind')
            if status['state'] == 'failed': break
            time.sleep(.02)
        assert status['state'] == 'failed'
        assert 'Download unavailable' in status['message']
        assert worker.call_count == 1
    r._jobs.clear()


def test_existing_resources_never_start_download():
    with patch.object(r.p, 'tables_available', return_value=True), patch.object(r, 'dataset_embeddings', return_value=[]), patch('subprocess.run') as worker:
        assert r.acquire('dataset.mind')['state'] == 'ready'
        assert r.acquire('item.cnn')['state'] == 'unnecessary'
        worker.assert_not_called()


def test_dataset_is_not_ready_until_required_embedding_exists(tmp_path, monkeypatch):
    embedding = tmp_path / 'token-embeddings.npy'
    monkeypatch.setattr(r.p, 'tables_available', lambda *_: True)
    monkeypatch.setattr(r, 'dataset_embeddings', lambda *_: [{'path': str(embedding)}])
    assert r.inspect('dataset.mind')['state'] == 'missing'
    embedding.touch()
    assert r.inspect('dataset.mind')['state'] == 'ready'


def test_delete_dataset_and_restore_download_state(tmp_path, monkeypatch):
    from studio.backend.executor import executor
    monkeypatch.setattr(r.p, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(r.p, 'CACHE', tmp_path / 'resources')
    monkeypatch.setattr(executor, 'list_runs', lambda: [])
    data = {'item': {'ut': 'data/test/items', 'inputs': ['title']},
            'user': {'ut': 'data/test/users'},
            'inter': {key: 'data/test/' + key for key in ('train', 'dev', 'test')}}
    monkeypatch.setattr(r, 'dataset_config', lambda *args: ('config/data/mind.yaml', data))
    monkeypatch.setattr(r, 'dataset_embeddings', lambda *args: [])
    monkeypatch.setattr(r.p, 'prepared_directory', lambda *args: tmp_path / 'resources/datasets/mind-test')
    for name in r.p.dataset_paths(data):
        folder = tmp_path / name
        folder.mkdir(parents=True)
        (folder / 'meta.json').write_text('{"features":[{"name":"title"}]}')
        (folder / 'data.pkl').write_bytes(b'test')
    assert r.inspect('dataset.mind')['state'] == 'ready'
    assert r.remove('dataset.mind')['state'] == 'missing'
    assert not (tmp_path / 'data/test/items').exists()


def test_delete_rejects_active_run_and_external_paths(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from studio.backend.executor import executor
    import pytest
    monkeypatch.setattr(executor, 'list_runs', lambda: [SimpleNamespace(status='running', error='')])
    with pytest.raises(ValueError, match='Stop active'):
        r.remove('dataset.mind')
    with pytest.raises(ValueError, match='Only project-managed'):
        r._remove_owned(tmp_path / 'unmanaged')
