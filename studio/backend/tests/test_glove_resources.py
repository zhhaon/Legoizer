"""Cold-start GloVe export and tokenization must share one canonical vocabulary."""
import numpy as np
import pytest
from unitok import UniTok, VocabularyHub

from embedder.glove_embedder import GloVeEmbedder


def test_export_then_load_and_reexport_in_one_process(tmp_path, monkeypatch):
    vectors = tmp_path / 'glove.txt'
    vectors.write_text('hello 0.1 0.2\nworld 0.3 0.4\n')
    archive = tmp_path / 'glove.zip'
    archive.touch()
    monkeypatch.setattr(GloVeEmbedder, 'FILE_DIR', str(tmp_path))
    monkeypatch.setattr(GloVeEmbedder, 'FILE_PATH', str(vectors))
    monkeypatch.setattr(GloVeEmbedder, 'ZIP_PATH', str(archive))
    with UniTok():
        weights = GloVeEmbedder().get_embeddings()
        first = GloVeEmbedder.get_glove_vocab()
        second = GloVeEmbedder.get_glove_vocab()
        assert first is second is VocabularyHub.get('glove')
        assert list(first) == ['hello', 'world']
        assert first['world'] == 1
        np.testing.assert_allclose(weights, [[.1, .2], [.3, .4]])
        np.testing.assert_array_equal(GloVeEmbedder().get_embeddings(), weights)


def test_missing_vocab_does_not_register_an_empty_vocab(tmp_path, monkeypatch):
    monkeypatch.setattr(GloVeEmbedder, 'FILE_DIR', str(tmp_path))
    with UniTok():
        with pytest.raises(FileNotFoundError):
            GloVeEmbedder.get_glove_vocab()
        assert not VocabularyHub.has('glove')


def test_missing_nltk_data_is_acquired_in_the_project_cache(tmp_path, monkeypatch):
    import nltk
    import unitok
    from studio.backend.executor import provision

    monkeypatch.setattr(provision, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(provision, 'CACHE', tmp_path / 'resources')
    monkeypatch.setattr(nltk.data, 'path', [])
    monkeypatch.setattr(nltk.data, 'find', lambda *_: (_ for _ in ()).throw(LookupError()))
    calls = []
    monkeypatch.setattr(nltk, 'download', lambda name, **kwargs: calls.append((name, kwargs)) or True)
    monkeypatch.setattr(GloVeEmbedder, 'FILE_DIR', str(tmp_path / 'embeddings'))
    (tmp_path / 'embeddings').mkdir()
    (tmp_path / 'embeddings/glove.vocab').touch()
    monkeypatch.setattr(GloVeEmbedder, 'get_glove_vocab', lambda: 'canonical-vocabulary')
    monkeypatch.setattr(unitok, 'GloVeTokenizer', lambda vocab: vocab)
    assert provision.tokenizer('glove') == 'canonical-vocabulary'
    assert calls[0][0] == 'punkt_tab'
    assert calls[0][1]['download_dir'] == str(tmp_path / 'resources/nltk')
    assert (tmp_path / 'resources/nltk').stat().st_mode & 0o077 == 0
