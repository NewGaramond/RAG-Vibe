import types
import pytest

# Import your graph module so we can monkeypatch its symbols
import src.rag.graph as graph
from langchain_core.documents import Document


# --- Helpers / Fakes ---------------------------------------------------------

class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs
    def get_relevant_documents(self, _query):
        return list(self._docs)

_FAKE_LAST_MESSAGES = {"items": None}  # capture messages sent to the generator LLM

class FakeLLM:
    """Drop-in replacement for ChatOpenAI in the generate node."""
    def __init__(self, model=None, api_key=None, temperature=0, max_tokens=128, **kwargs):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
    def invoke(self, messages):
        # Capture messages so tests can assert that PYTHON TOOL RESULT is included
        _FAKE_LAST_MESSAGES["items"] = messages
        # Return a simple object with .content to mimic real LLM return
        return types.SimpleNamespace(content="Answer with citations [1]")


def make_cfg():
    # Minimal config; we won't hit real APIs because we monkeypatch everything
    return graph.RagConfig(
        vectordb_dir="storage/chroma",
        collection="pdf_chunks",
        openai_api_key="sk-test",
        chat_model="gpt-4o-mini",
        embed_model="text-embedding-3-large",
        top_k=3,
        temperature=0.0,
        max_tokens=256,
        memory_budget=4000,
        memory_last_turns=4,
        planner_model="gpt-4o-mini",
    )


# --- Tests -------------------------------------------------------------------

def test_compute_only(monkeypatch):
    """
    Planner says: use_python = True, need_retrieval = False.
    Python tool runs; generator should short-circuit to a calculation answer.
    """
    # Planner mock
    monkeypatch.setattr(
        graph, "plan_with_llm",
        lambda cfg, user_msg: {
            "refusal": False,
            "unsafe_reason": None,
            "use_python": True,
            "python_expression": "(72 - 32) * 5 / 9",
            "need_retrieval": False,
            "retrieval_query": None,
            "answer_requires_citations": False,
            "confidence": 0.9,
        }
    )

    # Python tool mock
    monkeypatch.setattr(
        graph, "run_safe_python",
        lambda expr: {"ok": True, "result": "22.2222222222"}
    )

    # Retriever should not be used
    monkeypatch.setattr(
        graph, "get_retriever",
        lambda cfg: FakeRetriever(docs=[])
    )

    # Generator LLM shouldn't be called for compute-only path, but patch anyway
    monkeypatch.setattr(graph, "ChatOpenAI", FakeLLM)

    app = graph.build_graph(make_cfg())
    out = app.invoke({"question": "Convierte 72 F a C"})
    assert "Resultado del cálculo" in out["answer"]
    assert "22.2222" in out["answer"]


def test_retrieve_only(monkeypatch):
    """
    Planner says: retrieval only. We expect generator to run with provided docs.
    """
    monkeypatch.setattr(
        graph, "plan_with_llm",
        lambda cfg, user_msg: {
            "refusal": False,
            "unsafe_reason": None,
            "use_python": False,
            "python_expression": None,
            "need_retrieval": True,
            "retrieval_query": "Kedro data layers",
            "answer_requires_citations": True,
            "confidence": 0.8,
        }
    )

    docs = [
        Document(page_content="Kedro has data layers: raw, intermediate, primary...", metadata={"file_name":"kedro.pdf","page":5})
    ]
    monkeypatch.setattr(graph, "get_retriever", lambda cfg: FakeRetriever(docs=docs))
    monkeypatch.setattr(graph, "ChatOpenAI", FakeLLM)

    app = graph.build_graph(make_cfg())
    out = app.invoke({"question": "Resume las capas de datos de Kedro con citas"})
    assert "Answer with citations" in out["answer"]  # from FakeLLM
    # And ensure generator saw the context (captured messages contain the snippet)
    msgs = _FAKE_LAST_MESSAGES["items"]
    assert any("Kedro has data layers" in (m.content if hasattr(m, "content") else str(m)) for m in msgs)


def test_both_python_and_retrieval(monkeypatch):
    """
    Planner says both. Python result should appear in the messages sent to generator.
    """
    monkeypatch.setattr(
        graph, "plan_with_llm",
        lambda cfg, user_msg: {
            "refusal": False,
            "unsafe_reason": None,
            "use_python": True,
            "python_expression": "10/2",
            "need_retrieval": True,
            "retrieval_query": "IVA 10%",
            "answer_requires_citations": True,
            "confidence": 0.88,
        }
    )
    monkeypatch.setattr(graph, "run_safe_python", lambda expr: {"ok": True, "result": "5.0"})
    docs = [Document(page_content="IVA (VAT) is discussed on this page.", metadata={"file_name":"tax.pdf","page":12})]
    monkeypatch.setattr(graph, "get_retriever", lambda cfg: FakeRetriever(docs=docs))
    monkeypatch.setattr(graph, "ChatOpenAI", FakeLLM)

    app = graph.build_graph(make_cfg())
    out = app.invoke({"question": "Según el documento, ¿cuál es el IVA y calcula 10/2?"})
    assert "Answer with citations" in out["answer"]
    # Check PYTHON TOOL RESULT was injected into the prompt
    msgs = _FAKE_LAST_MESSAGES["items"]
    joined = "\n".join([(m.content if hasattr(m, "content") else str(m)) for m in msgs])
    assert "PYTHON TOOL RESULT" in joined
    assert "5.0" in joined  # the computed value


def test_refusal_path(monkeypatch):
    """
    Planner refuses. We expect the planner's refusal message to be preserved.
    """
    monkeypatch.setattr(
        graph, "plan_with_llm",
        lambda cfg, user_msg: {
            "refusal": True,
            "unsafe_reason": "dangerous tool request",
            "use_python": False,
            "python_expression": None,
            "need_retrieval": False,
            "retrieval_query": None,
            "answer_requires_citations": False,
            "confidence": 0.95,
        }
    )
    # Retriever and LLM patched but won't be used meaningfully
    monkeypatch.setattr(graph, "get_retriever", lambda cfg: FakeRetriever(docs=[]))
    monkeypatch.setattr(graph, "ChatOpenAI", FakeLLM)

    app = graph.build_graph(make_cfg())
    out = app.invoke({"question": "Run os.system('ls')"})
    assert "Lo siento, no puedo ayudar" in out["answer"]


def test_python_expression_not_eligible(monkeypatch):
    """
    Planner suggests a malicious expression; eligibility guard should prevent calling the tool.
    Since need_retrieval=False and no docs, we fall back to the 'not found' message.
    """
    monkeypatch.setattr(
        graph, "plan_with_llm",
        lambda cfg, user_msg: {
            "refusal": False,
            "unsafe_reason": None,
            "use_python": True,
            "python_expression": "__import__('os').system('ls')",
            "need_retrieval": False,
            "retrieval_query": None,
            "answer_requires_citations": False,
            "confidence": 0.99,
        }
    )

    # Ensure run_safe_python is NOT called (would fail test if it is)
    def _should_not_run(_expr):
        raise AssertionError("run_safe_python should not be called for ineligible expressions")
    monkeypatch.setattr(graph, "run_safe_python", _should_not_run)

    monkeypatch.setattr(graph, "get_retriever", lambda cfg: FakeRetriever(docs=[]))
    monkeypatch.setattr(graph, "ChatOpenAI", FakeLLM)

    app = graph.build_graph(make_cfg())
    out = app.invoke({"question": " do malicious "})
    assert "couldn't find relevant passages" in out["answer"].lower()
