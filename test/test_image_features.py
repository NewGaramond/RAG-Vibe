import types
from PIL import Image
import src.ingest.image_features as imf

class DummyEmb:
    def __init__(self, *a, **k): pass

def test_build_text_blob():
    t = imf.build_text_blob("A bar chart of sales.", ["chart","sales","bar"])
    assert "Caption:" in t and "Tags:" in t

def test_dedupe_hash_same_image(tmp_path, monkeypatch):
    # Create two identical images; phash must match
    img = Image.new("RGB", (300, 200), (128, 128, 128))
    h1 = imf._perceptual_hash(img)
    h2 = imf._perceptual_hash(img.copy())
    assert h1 == h2

def test_caption_fallback_to_text(monkeypatch):
    # Force captioner to return non-JSON; ensure we still get a caption string
    monkeypatch.setattr(imf, "ChatOpenAI", lambda **k: types.SimpleNamespace(
        invoke=lambda msgs: types.SimpleNamespace(content="A diagram with 3 boxes linked by arrows.")
    ))
    out = imf.caption_image_with_openai(Image.new("RGB", (200,200), (255,255,255)),
                                        imf.ImageIngestConfig(openai_api_key="sk", embed_model="", vectordb_dir="", collection=""))
    assert isinstance(out.get("caption",""), str)

