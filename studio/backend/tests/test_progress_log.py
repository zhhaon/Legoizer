from studio.backend.executor.progress_log import ProgressLog


def test_progress_across_chunks_and_normal_logs():
    rows = []
    def emit(line, replace):
        if replace and rows:
            rows[-1] = line
        else:
            rows.append(line)
    log = ProgressLog(emit)
    for chunk in ["start\n\rItem Cach", "ing 10%\rItem Caching 20%", "\rItem Caching 100%\n", "\rTraining 10%\rTraining 50%, loss: 0.2\r", "Training 100%\nvalidation\npartial"]:
        log.feed(chunk)
    log.finish()
    assert rows == ["start", "Item Caching 100%", "Training 100%", "validation", "partial"]


def test_crlf_ansi_and_tqdm_clear():
    events = []
    log = ProgressLog(lambda line, replace: events.append((line, replace)))
    log.feed("hello\r\n\r\x1b[32m10%\x1b[0m\r100%\r     \r\nnext\n")
    log.finish()
    assert events == [("hello", False), ("10%", False), ("100%", True), ("next", False)]


def test_polling_resends_mutable_tail(monkeypatch):
    from studio.backend.executor import executor
    from studio.backend.schemas.runs import RunRecord
    record = RunRecord(run_id="progress-test", log_lines=["start", "10%"])
    monkeypatch.setitem(executor._RUNS, record.run_id, record)
    first = executor.logs_since(record.run_id, 0)
    executor._append_log(record, "20%", replace=True)
    update = executor.logs_since(record.run_id, first["next"])
    assert update["next"] == first["next"] == 2
    assert update["replace_from"] == 1
    assert update["lines"] == ["20%"]
    executor._append_log(record, "done")
    assert executor.logs_since(record.run_id, 2)["lines"] == ["20%", "done"]


def test_training_progress_enabled(naml_graph):
    from studio.backend.compiler.compiler import build_exp_config
    from studio.backend.schemas.graph import GraphJson
    assert build_exp_config(GraphJson.model_validate(naml_graph))["policy"]["disable_tqdm"] is False


def test_real_process_progress_arrives_before_exit():
    import sys
    import threading
    import time
    from studio.backend.executor import executor
    from studio.backend.schemas.runs import RunRecord
    record = RunRecord(run_id="process-progress-test")
    source = "import sys,time; print('start',flush=True); sys.stdout.write('\\rTraining 10%\\r'); sys.stdout.flush(); time.sleep(1); sys.stdout.write('Training 100%\\n'); sys.stdout.flush()"
    result = []
    worker = threading.Thread(target=lambda: result.append(executor._run_step(record, [sys.executable, "-u", "-c", source], {})))
    worker.start()
    try:
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline and "Training 10%" not in record.log_lines:
            time.sleep(0.01)
        assert "Training 10%" in record.log_lines
        assert worker.is_alive()
    finally:
        worker.join(timeout=5)
    assert result == [0]
    assert record.log_lines[-2:] == ["start", "Training 100%"]
