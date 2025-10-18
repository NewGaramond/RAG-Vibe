# src/tools/python_tool.py
from __future__ import annotations
import ast
import math
from typing import Any, Dict, List, Optional, Union

# --- Config knobs (can be mirrored in configs/app.yaml) ---
MAX_AST_NODES = 200           # prevent giant expressions
MAX_STR_LEN = 2000            # cap output and string literal sizes
MAX_COLLECTION_LEN = 1000     # cap literal list/dict/set sizes
ALLOW_FUNCTIONS = {
    "len": len,
    "sum": lambda x: sum(_ensure_iter(x)),
    "min": lambda x: min(_ensure_iter(x)),
    "max": lambda x: max(_ensure_iter(x)),
    "sorted": lambda x: sorted(_ensure_iter(x)),
    "round": round,
    # Optional: allow a small slice of math safely
    "abs": abs,
}
ALLOW_CONSTANT_NAMES = {"True": True, "False": False, "None": None}

def _ensure_iter(x: Any):
    if isinstance(x, (list, tuple)):
        if len(x) > MAX_COLLECTION_LEN:
            raise ValueError("Collection too large")
        return x
    raise TypeError("Expected a list or tuple")

ALLOWED_BINOPS = (
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub, ast.Not,)
ALLOWED_CMPOPS = (
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

class UnsafeExpression(Exception):
    pass

class _NodeLimiter(ast.NodeVisitor):
    """Counts AST nodes to prevent pathological inputs."""
    def __init__(self, limit: int):
        self.limit = limit
        self.count = 0
    def generic_visit(self, node):
        self.count += 1
        if self.count > self.limit:
            raise UnsafeExpression("Expression too large")
        super().generic_visit(node)

class SafeEvaluator(ast.NodeVisitor):
    """
    Minimal expression interpreter over a strict AST allow-list.
    No attribute access, no comprehensions, no loops, no calls
    except to whitelisted functions by bare name.
    """
    def __init__(self, allow_functions: Dict[str, Any] | None = None):
        self.allow_functions = allow_functions or ALLOW_FUNCTIONS

    # ---- Entry point ----
    def eval(self, expr: str) -> Any:
        if "__" in expr:
            raise UnsafeExpression("Dunder access is not allowed")
        tree = ast.parse(expr, mode="eval")
        # Size guard
        _NodeLimiter(MAX_AST_NODES).visit(tree)
        return self.visit(tree.body)

    # ---- Literals ----
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and len(node.value) > MAX_STR_LEN:
            raise UnsafeExpression("String literal too long")
        if isinstance(node.value, (int, float, bool, str)) or node.value is None:
            return node.value
        raise UnsafeExpression("Unsupported constant type")

    def visit_List(self, node: ast.List):
        if len(node.elts) > MAX_COLLECTION_LEN:
            raise UnsafeExpression("List too large")
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node: ast.Tuple):
        if len(node.elts) > MAX_COLLECTION_LEN:
            raise UnsafeExpression("Tuple too large")
        return tuple(self.visit(e) for e in node.elts)

    def visit_Set(self, node: ast.Set):
        if len(node.elts) > MAX_COLLECTION_LEN:
            raise UnsafeExpression("Set too large")
        return {self.visit(e) for e in node.elts}

    def visit_Dict(self, node: ast.Dict):
        if len(node.keys) > MAX_COLLECTION_LEN:
            raise UnsafeExpression("Dict too large")
        return {self.visit(k): self.visit(v) for k, v in zip(node.keys, node.values)}

    # ---- Names (constants + whitelisted functions) ----
    def visit_Name(self, node: ast.Name):
        if node.id in ALLOW_CONSTANT_NAMES:
            return ALLOW_CONSTANT_NAMES[node.id]
        if node.id in self.allow_functions:
            # Return a callable wrapper so Call can use it
            return self.allow_functions[node.id]
        # everything else is blocked
        raise UnsafeExpression(f"Name '{node.id}' is not allowed")

    # ---- Operators ----
    def visit_BinOp(self, node: ast.BinOp):
        if not isinstance(node.op, ALLOWED_BINOPS):
            raise UnsafeExpression("Operator not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return self._apply_binop(node.op, left, right)

    def _apply_binop(self, op, left, right):
        if isinstance(op, ast.Add):       return left + right
        if isinstance(op, ast.Sub):       return left - right
        if isinstance(op, ast.Mult):      return left * right
        if isinstance(op, ast.Div):       return left / right
        if isinstance(op, ast.FloorDiv):  return left // right
        if isinstance(op, ast.Mod):       return left % right
        raise UnsafeExpression("Operator not handled")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            raise UnsafeExpression("Unary operator not allowed")
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd): return +operand
        if isinstance(node.op, ast.USub): return -operand
        if isinstance(node.op, ast.Not):  return not operand
        raise UnsafeExpression("Unary operator not handled")

    def visit_BoolOp(self, node: ast.BoolOp):
        if not isinstance(node.op, (ast.And, ast.Or)):
            raise UnsafeExpression("Boolean operator not allowed")
        values = [self.visit(v) for v in node.values]
        if isinstance(node.op, ast.And):
            res = True
            for v in values:
                res = res and v
                if not res: break
            return res
        else:
            res = False
            for v in values:
                res = res or v
                if res: break
            return res

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, ALLOWED_CMPOPS):
                raise UnsafeExpression("Comparison operator not allowed")
            right = self.visit(comparator)
            if isinstance(op, ast.Eq) and not (left == right): return False
            if isinstance(op, ast.NotEq) and not (left != right): return False
            if isinstance(op, ast.Lt) and not (left < right): return False
            if isinstance(op, ast.LtE) and not (left <= right): return False
            if isinstance(op, ast.Gt) and not (left > right): return False
            if isinstance(op, ast.GtE) and not (left >= right): return False
            if isinstance(op, ast.In) and not (left in right): return False
            if isinstance(op, ast.NotIn) and not (left not in right): return False
            left = right
        return True

    # ---- Indexing ----
    def visit_Subscript(self, node: ast.Subscript):
        value = self.visit(node.value)
        sl = self.visit(node.slice)
        return value[sl]

    def visit_Slice(self, node: ast.Slice):
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step  = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

    def visit_Index(self, node: ast.Index):  # Py<3.9 compatibility, harmless here
        return self.visit(node.value)

    # ---- Calls (whitelist names only, no attributes) ----
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Attribute):
            raise UnsafeExpression("Attribute access is not allowed")
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        if node.keywords:
            raise UnsafeExpression("Keyword arguments not allowed in MVP")
        # size guards on typical list-taking functions inside wrapper lambdas
        result = func(*args)
        return result

    # Explicitly block everything else
    def visit_Attribute(self, node):    raise UnsafeExpression("Attribute access is not allowed")
    def visit_Lambda(self, node):       raise UnsafeExpression("Lambdas are not allowed")
    def visit_ListComp(self, node):     raise UnsafeExpression("Comprehensions are not allowed")
    def visit_SetComp(self, node):      raise UnsafeExpression("Comprehensions are not allowed")
    def visit_DictComp(self, node):     raise UnsafeExpression("Comprehensions are not allowed")
    def visit_GeneratorExp(self, node): raise UnsafeExpression("Comprehensions are not allowed")
    def visit_IfExp(self, node):        raise UnsafeExpression("Ternary expressions are not allowed in MVP")

def run_safe_python(expression: str, allow_functions: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main API: returns a dict with 'ok', 'result' or 'error'.
    """
    evaluator = SafeEvaluator(allow_functions=allow_functions)
    try:
        result = evaluator.eval(expression.strip())
        # stringify conservatively, cap length
        s = repr(result)
        if len(s) > MAX_STR_LEN:
            s = s[:MAX_STR_LEN] + "…"
        return {"ok": True, "result": s}
    except (UnsafeExpression, ValueError, TypeError, ZeroDivisionError) as e:
        return {"ok": False, "error": str(e)}
