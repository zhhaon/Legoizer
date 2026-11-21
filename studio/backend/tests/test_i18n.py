"""Translation references preserve existing English API contracts and graph data."""
import json
from fastapi.testclient import TestClient
from studio.backend.app import app
from studio.backend.schemas.graph import GraphJson
from studio.backend.validator.validator import validate


def test_validation_references_are_serializable_and_keep_english():
    graph = GraphJson()
    original = graph.model_dump()
    report = validate(graph)
    issue = next(i for i in report.errors if i.code == 'empty_graph')
    assert issue.message == 'The model canvas is empty.'
    assert issue.translations['message']['key'].startswith('validation.empty_graph.message.')
    assert issue.translations['reason']['key']
    assert issue.translations['fix']['key']
    json.dumps(report.model_dump(mode='json'))
    assert graph.model_dump() == original


def test_http_error_preserves_detail_and_adds_language_neutral_reference():
    with TestClient(app) as client:
        response = client.get('/api/templates/not-a-template')
    assert response.status_code == 404
    assert response.json()['detail'] == 'Unknown template'
    assert response.json()['translation']['key'].startswith('backend.')
