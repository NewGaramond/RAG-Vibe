# src/ingest/image_features.py
from __future__ import annotations
import io, os, base64, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
import imagehash

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


@dataclass
class ImageIngestConfig:
    openai_api_key: str
    embed_model: str
    vectordb_dir: str
    collection: str
    caption_model: str = "gpt-4o-mini"  # vision-capable
    min_width: int = 150
    min_height: int = 150
    max_images_per_pdf: int = 24
    images_dir: str = "storage/images"
    thumbs_dir: str = "storage/images/thumbs"
    caption_max_tokens: int = 250


def _ensure_dirs(cfg: ImageIngestConfig):
    os.makedirs(cfg.images_dir, exist_ok=True)
    os.makedirs(cfg.thumbs_dir, exist_ok=True)
    os.makedirs(cfg.vectordb_dir, exist_ok=True)


def _img_from_xref(doc: fitz.Document, xref: int) -> Optional[Image.Image]:
    try:
        pix = fitz.Pixmap(doc, xref)
        if pix.n >= 5:  # CMYK or with alpha
            pix = fitz.Pixmap(fitz.csRGB, pix)
        data = pix.tobytes("png")
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _save_png(img: Image.Image, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path, format="PNG", optimize=True)


def _thumb(img: Image.Image, size=(320, 320)) -> Image.Image:
    t = img.copy()
    t.thumbnail(size)
    return t


def _perceptual_hash(img: Image.Image) -> str:
    return str(imagehash.phash(img))


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_images_from_pdf(pdf_path: str, cfg: ImageIngestConfig) -> List[Tuple[int, Image.Image, Tuple[int,int]]]:
    """Return list of (page_number, PIL_Image, (w,h)) filtered by size."""
    out: List[Tuple[int, Image.Image, Tuple[int,int]]] = []
    with fitz.open(pdf_path) as doc:
        for pno, page in enumerate(doc, start=1):
            img_list = page.get_images(full=True)
            for i, (xref, *_rest) in enumerate(img_list):
                pil = _img_from_xref(doc, xref)
                if pil is None:
                    continue
                w, h = pil.size
                if w < cfg.min_width or h < cfg.min_height:
                    continue
                out.append((pno, pil, (w, h)))
                if len(out) >= cfg.max_images_per_pdf:
                    return out
    return out


def caption_image_with_openai(img: Image.Image, cfg: ImageIngestConfig) -> Dict[str, Any]:
    """Ask the vision model for a factual caption and 5-10 tags."""
    llm = ChatOpenAI(model=cfg.caption_model, api_key=cfg.openai_api_key, temperature=0)
    prompt = (
        "Describe this image factually in 1–2 sentences. "
        "Then provide 5–10 short, comma-separated tags (nouns). "
        "Do not speculate beyond visible content. Respond as JSON with keys "
        '{"caption": "...", "tags": ["...","..."]}.'
    )
    data_url = _image_to_data_url(img)
    msgs = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]}
    ]
    txt = llm.invoke(msgs).content.strip()
    # Be defensive: if model didn't return JSON, wrap it
    try:
        import json
        parsed = json.loads(txt)
        cap = str(parsed.get("caption", ""))[: cfg.caption_max_tokens]
        tags = [t.strip() for t in (parsed.get("tags", []) or [])][:10]
    except Exception:
        cap = txt[: cfg.caption_max_tokens]
        tags = []
    return {"caption": cap, "tags": tags}


def upsert_image_record(
    cfg: ImageIngestConfig,
    text: str,
    metadata: Dict[str, Any],
):
    """Embed + upsert a single image record to Chroma."""
    embeddings = OpenAIEmbeddings(model=cfg.embed_model, api_key=cfg.openai_api_key)
    vs = Chroma(
        collection_name=cfg.collection,
        persist_directory=cfg.vectordb_dir,
        embedding_function=embeddings,
    )
    doc = Document(page_content=text, metadata=metadata)
    vs.add_documents([doc])  # persist happens on close; Chroma handles
    # Note: using same collection as text; filter by modality via metadata if needed.


def build_text_blob(caption: str, tags: List[str], extra_ocr_text: str = "") -> str:
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if tags:
        parts.append("Tags: " + ", ".join(tags))
    if extra_ocr_text:
        parts.append("OCR: " + extra_ocr_text)
    return "\n".join(parts).strip()


def ingest_pdf_images(pdf_path: str, cfg: ImageIngestConfig) -> List[Dict[str, Any]]:
    """Extract, dedupe, caption and index images from a single PDF."""
    _ensure_dirs(cfg)
    images = extract_images_from_pdf(pdf_path, cfg)
    results: List[Dict[str, Any]] = []

    # Prepare de-dup cache across this run
    seen_hashes: set[str] = set()

    file_name = os.path.basename(pdf_path)
    doc_id = os.path.splitext(file_name)[0]

    for idx, (page, img, (w, h)) in enumerate(images, start=1):
        phash = _perceptual_hash(img)
        if phash in seen_hashes:
            continue
        seen_hashes.add(phash)

        # Save originals and thumbs (optional but nice for UI)
        image_name = f"{doc_id}_p{page}_i{idx}.png"
        full_path = os.path.join(cfg.images_dir, image_name)
        _save_png(img, full_path)
        _save_png(_thumb(img), os.path.join(cfg.thumbs_dir, image_name))

        # Caption + tags
        cap = caption_image_with_openai(img, cfg)
        text_blob = build_text_blob(cap.get("caption",""), cap.get("tags", []), "")

        metadata = {
            "modality": "image",
            "file_name": file_name,
            "doc_id": doc_id,
            "page": page,
            "width": w,
            "height": h,
            "image_hash": phash,
            "image_path": full_path,
            "thumb_path": os.path.join(cfg.thumbs_dir, image_name),
        }

        upsert_image_record(cfg, text_blob, metadata)
        results.append({"page": page, "caption": cap.get("caption",""), "tags": cap.get("tags", []), "metadata": metadata})

    return results
