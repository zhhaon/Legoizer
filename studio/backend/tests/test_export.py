from copy import deepcopy
import json


def test_export_is_portable_and_does_not_compile_files(client, naml_graph, monkeypatch):
    from studio.backend.compiler import compiler
    monkeypatch.setattr(compiler, '_write_generated', lambda *args: (_ for _ in ()).throw(AssertionError('export wrote files')))
    graph = deepcopy(naml_graph)
    graph['train'].update(lr=0.004, epochs=7, accumulate_batch=3, seed=42)
    response = client.post('/api/graph/export', json=graph)
    assert response.status_code == 200
    body = response.json()
    assert body['exp']['policy']['epoch'] == 7
    assert body['exp']['policy']['accumulate_batch'] == 3
    assert body['exp']['policy']['lr'] == 0.004
    assert body['plan']['data'] == body['data']
    assert body['plan']['embed'] == body['embed']
    assert '${' not in json.dumps(body)
    assert len(body['revision']) == 40


def test_export_rejects_invalid_graph(client, naml_graph):
    graph = deepcopy(naml_graph)
    graph['nodes'] = []
    graph['edges'] = []
    assert client.post('/api/graph/export', json=graph).status_code == 422
