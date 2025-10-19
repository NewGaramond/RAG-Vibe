# Safe Python Expression Tool

A tiny, **whitelisted expression evaluator** for small, deterministic calculations inside your RAG graph. It interprets a **single Python expression** under a strict AST allow-list—**no imports, no attributes, no loops, no comprehensions, no assignments**—and returns a compact string result.

Designed to pair with the planner node in `src/rag/graph.py`, which decides when to call this tool.

---

## What it can do

* Arithmetic: `+ - * / // %`
* Unary ops: `+x`, `-x`, `not x`
* Comparisons & boolean logic: `== != < <= > >= and or in not in`
* Indexing & slicing: `x[i]`, `x[i:j:k]`
* Literal containers: lists, tuples, sets, dicts (bounded sizes)
* Whitelisted functions: `len`, `sum`, `min`, `max`, `sorted`, `round`, `abs`

**Won’t do:** imports, file/OS/network access, attributes (`obj.attr`), function defs, lambdas, f-strings, comprehensions, loops, assignments, dunder names, keyword args in calls.

---

## Public API

```python
from src.python_tool import run_safe_python

out = run_safe_python("(72 - 32) * 5 / 9")
# {"ok": True, "result": "22.22222222222222"}

bad = run_safe_python("__import__('os').system('ls')")
# {"ok": False, "error": "Dunder access is not allowed"}
```

**Return shape**

* Success → `{"ok": True, "result": "<repr-capped>"}`
* Failure → `{"ok": False, "error": "<reason>"}`

---

## Security model

1. **AST parse** in `mode="eval"` (expression-only).
2. **Node limit** (`MAX_AST_NODES`) to bound complexity.
3. **Strict visitor allow-list**:

   * Literals: bool, int, float, str (size-capped)
   * Containers: list/tuple/set/dict (size-capped)
   * Operators: small set of arithmetic/boolean/comparison ops
   * Names: only constants (`True/False/None`) + whitelisted function names
   * Calls: **bare whitelisted names only**, **no attributes**, **no kwargs**
4. **Global dunder ban**: any `__` is rejected.
5. **Collection size guards** inside wrappers for `sum/min/max/sorted`.

---

## Configuration knobs (top of file)

```python
MAX_AST_NODES = 200         # cap expression complexity
MAX_STR_LEN = 2000          # cap output + string literal length
MAX_COLLECTION_LEN = 1000   # cap literal list/dict/set sizes

ALLOW_FUNCTIONS = {
    "len": len,
    "sum": lambda x: sum(_ensure_iter(x)),
    "min": lambda x: min(_ensure_iter(x)),
    "max": lambda x: max(_ensure_iter(x)),
    "sorted": lambda x: sorted(_ensure_iter(x)),
    "round": round,
    "abs": abs,
}

ALLOW_CONSTANT_NAMES = {"True": True, "False": False, "None": None}
```

> `sum/min/max/sorted` wrappers require a list/tuple and enforce size limits.

---

## Allowed syntax (quick table)

| Category       | Allowed | Examples                              |
| -------------- | :-----: | ------------------------------------- |
| Literals       |    ✅    | `3.14`, `True`, `{"a": 1}`            |
| Arithmetic     |    ✅    | `7 % 3`, `10 // 3`                    |
| Unary          |    ✅    | `-5`, `not (1 < 0)`                   |
| Boolean        |    ✅    | `a and b`, `x or y`                   |
| Comparisons    |    ✅    | `"a" in ["a","b"]`, `1 <= 2 < 3`      |
| Index/Slice    |    ✅    | `[1,2,3][0]`, `s[:10]`, `d["k"]`      |
| Calls          |    ✅    | `len([1,2])`, `round(3.14159, 2)`     |
| Names          |    ✅    | `True/False/None` + whitelisted funcs |
| Attributes     |    ❌    | `obj.attr`                            |
| Imports/Exec   |    ❌    | `__import__`, `eval`, `exec`, `open`  |
| Comprehensions |    ❌    | `[x for x in y]`                      |
| Assign/Defs    |    ❌    | `x = 5`, `lambda x: x`                |
| Loops/IfExpr   |    ❌    | `for`, `while`, `a if b else c`       |
| Kwargs         |    ❌    | `sorted(x, key=...)`                  |

---

## Usage examples

```python
run_safe_python("len([1, 2, 3])")                 # ok → '3'
run_safe_python("sum([10, 20, 30]) / 3")          # ok → '20.0'
run_safe_python("sorted([3,1,2])")                # ok → '[1, 2, 3]'
run_safe_python("round(3.14159, 3)")              # ok → '3.142'
run_safe_python("not (3 in [1,2,3])")             # ok → 'False'
run_safe_python("['a','b','c'][1:]")              # ok → "['b', 'c']"

run_safe_python("{i:i*i for i in range(3)}")      # error (comprehensions)
run_safe_python("open('/etc/passwd').read()")     # error (name/attribute)
run_safe_python("__import__('os').system('ls')")  # error (dunder ban)
run_safe_python("sorted((1,2), key=lambda x:-x)") # error (kwargs + lambda)
run_safe_python("min(10)")                        # error (expects list/tuple)
```

---

## Integrating with the graph

In `src/rag/graph.py`, the planner decides when to call the tool and provides a **single expression**. The graph performs a lightweight pre-eligibility check before calling:

```python
from src.python_tool import run_safe_python

expr = "(72 - 32) * 5 / 9"
tool = run_safe_python(expr)
python_result = tool["result"] if tool["ok"] else f"(python declined: {tool['error']})"
# Pass python_result as context (not as a citation source).
```

---

## Extending safely

* **More math (pure only):**

  ```python
  import math
  ALLOW_FUNCTIONS.update({
      "ceil": math.ceil,
      "floor": math.floor,
      "sqrt": math.sqrt,
  })
  ```
* **Tighten bounds:** lower `MAX_AST_NODES`, `MAX_STR_LEN`, `MAX_COLLECTION_LEN`.
* **i18n:** map error messages if you need localized UX (return shape unchanged).

---

## Testing

```python
# tests/test_python_tool.py
import pytest
from src.python_tool import run_safe_python

@pytest.mark.parametrize("expr, ok", [
    ("(72 - 32) * 5 / 9", True),
    ("len([1,2,3])", True),
    ("__import__('os')", False),
    ("[x for x in range(3)]", False),
])
def test_safe_python(expr, ok):
    out = run_safe_python(expr)
    assert out["ok"] is ok

def test_caps_and_limits():
    assert run_safe_python("['x' * 10]")["ok"]
    huge = "'" + ("a" * 5000) + "'"
    err = run_safe_python(huge)
    assert (not err["ok"]) and "String literal too long" in err["error"]
```

Run with:

```bash
pytest -q
```

---

## Troubleshooting

* **Name not allowed** → used a non-whitelisted function/variable.
* **Attribute access is not allowed** → calls like `obj.method(...)` are blocked.
* **Keyword arguments not allowed in MVP** → only positional args are accepted.
* **Expression/List/Tuple/Set/Dict too large** → hit a safety cap.
* **Type errors with sum/min/max/sorted** → they require a list/tuple, not generators/sets.

---

## Changelog

* **v0.2** — Size caps, boolean ops, membership tests, clearer errors.
* **v0.1** — Minimal allow-listed evaluator + safe wrappers for list/tuple funcs.
