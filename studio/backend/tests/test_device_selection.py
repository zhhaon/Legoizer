from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from studio.backend.executor import executor
from studio.backend.schemas.graph import GraphJson


def test_cuda_selection_command(monkeypatch, naml_graph):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: [(100, 1000), (800, 1000), (300, 1000)][index])
    graph = GraphJson.model_validate(naml_graph)
    config = SimpleNamespace(data_config="config/data/mind.yaml", generated_file="model.yaml",
                             embed_config="config/embed/null.yaml", exp_config="exp.yaml")
    for device, index, expected in [("cpu", None, "-1"), ("auto", None, "0"),
                                     ("cuda", None, "1"), ("cuda", 0, "0"), ("cuda", 2, "2")]:
        graph.train.device, graph.train.cuda_index = device, index
        command = executor._build_command(graph, config, script=Path("trainer.py"))
        assert command[command.index("--cuda") + 1] == expected
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    graph.train.device = "auto"
    command = executor._build_command(graph, config, script=Path("trainer.py"))
    assert command[command.index("--cuda") + 1] == "-1"
    with pytest.raises(ValueError, match="unavailable"):
        executor._automatic_cuda_index()


def test_environment_lists_gpu_names(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"Test GPU {index}")
    report = executor.environment_report()
    assert report["cuda_devices"] == [{"index": 0, "name": "Test GPU 0"}, {"index": 1, "name": "Test GPU 1"}]
