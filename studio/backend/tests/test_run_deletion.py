from types import SimpleNamespace
import pytest
from studio.backend.executor import executor as e


def test_delete_persisted_record(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(e, '_RUNS', {'test': SimpleNamespace(run_id='test', status='completed')})
    record = tmp_path / 'test.json'
    record.write_text('{}')
    checkpoint = tmp_path / 'keep.pt'
    checkpoint.write_bytes(b'checkpoint')
    e.delete_run('test')
    assert not record.exists()
    assert e.get_run('test') is None
    assert checkpoint.exists()
    with pytest.raises(KeyError): e.delete_run('test')


def test_active_run_cannot_be_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(e, 'RUNS_DIR', tmp_path)
    monkeypatch.setattr(e, '_RUNS', {'test': SimpleNamespace(run_id='test', status='running')})
    with pytest.raises(ValueError): e.delete_run('test')
    assert e.get_run('test') is not None
