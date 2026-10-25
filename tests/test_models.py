import torch

from climatesense.models import MAFNet, SequenceClassifier

MOD = {"climate": [0, 1, 2, 9], "air": [3, 4, 10], "dryness": [5, 6, 7, 8, 11]}
MISS = {"climate": 9, "air": 10, "dryness": 11}


def _x(B=8, L=14, F=12):
    x = torch.randn(B, L, F)
    x[:, :, 9:] = 0.0  # missing flags off
    return x


def test_mafnet_forward_shapes_and_probabilities():
    torch.manual_seed(0)
    m = MAFNet(MOD, MISS, n_ctx=8, d_model=32, hidden=32, heads=2).eval()
    logits, w = m(_x(), torch.randn(8, 8))
    assert logits.shape == (8, 4)
    assert w.shape == (8, 4)  # climate, air, dryness, context
    p = torch.softmax(logits, -1)
    torch.testing.assert_close(p.sum(-1), torch.ones(8))
    torch.testing.assert_close(w.sum(-1), torch.ones(8))


def test_attention_mask_excludes_missing_modality():
    torch.manual_seed(0)
    m = MAFNet(MOD, MISS, n_ctx=0, d_model=32, hidden=32, heads=2, token_self_attention=True).eval()
    x = _x()
    x[:4, :, 10] = 1.0  # air missing for the whole window in the first 4 samples
    _, w = m(x)
    assert torch.all(w[:4, 1] == 0)
    assert torch.all(w[4:, 1] > 0)


def test_concat_fusion_and_encoders():
    for enc in ["gru", "lstm", "transformer"]:
        m = MAFNet(MOD, MISS, n_ctx=8, d_model=16, hidden=16, encoder=enc, fusion="concat").eval()
        logits, w = m(_x(), torch.randn(8, 8))
        assert logits.shape == (8, 4) and w is None


def test_sequence_classifiers():
    for kind in ["rnn", "lstm", "gru", "transformer"]:
        m = SequenceClassifier(12, 8, hidden=16, kind=kind, layers=2).eval()
        logits, _ = m(_x(), torch.randn(8, 8))
        assert logits.shape == (8, 4)


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m = MAFNet(MOD, MISS, n_ctx=8, d_model=16, hidden=16).eval()
    torch.save(m.state_dict(), tmp_path / "m.pt")
    m2 = MAFNet(MOD, MISS, n_ctx=8, d_model=16, hidden=16).eval()
    m2.load_state_dict(torch.load(tmp_path / "m.pt"))
    x, c = _x(), torch.randn(8, 8)
    torch.testing.assert_close(m(x, c)[0], m2(x, c)[0])
