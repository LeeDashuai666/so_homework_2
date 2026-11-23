#!/usr/bin/env python3
"""Source-linked Z3 verification of prefix storage and captured decode contracts.

Executes selected real Python methods over symbolic tensors, not CUDA tensors.
Neural arithmetic, allocator correctness and faithful CUDA replay are contracts.
"""
import argparse
import ast
from contextlib import nullcontext
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import shutil
import sys
import time
from types import MethodType, SimpleNamespace

import z3 as z

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
RUNTIME = HERE.parent / "runtime.py"
INT = z.IntSort()
ARRAY = z.ArraySort(INT, INT)
BOOL_ARRAY = z.ArraySort(INT, z.BoolSort())
MINIMUM = -338953138925153547590470800371487866880
LENGTH = z.Int("prefix_length")
CAPACITY = 1024
LAYERS = 28
NEXT = z.Function("opaque_next_token", INT, INT, ARRAY, ARRAY, ARRAY, INT)
KEY = z.Function("opaque_key_write", INT, INT, INT, ARRAY, ARRAY, ARRAY, INT)
VALUE = z.Function("opaque_value_write", INT, INT, INT, ARRAY, ARRAY, ARRAY, INT)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


class Heap:
    """Distinct Python storage objects model successful nonaliasing allocations."""
    def __init__(self, prefix):
        self.prefix, self.cells, self.next_address = prefix, {}, 1000
        self.clone_alias = False

    def tensor(self, expr, array=False, end=None, shape=None):
        self.next_address += 1
        address = self.next_address
        self.cells[address] = expr
        return Tensor(self, address, array, end, shape)


class Tensor:
    def __init__(self, heap, address, array=False, end=None, shape=None):
        self.heap, self.address, self.array, self.end = heap, address, array, end
        self.shape = shape or ((1, 8, CAPACITY, 128) if array else (1, 1))

    @property
    def expr(self):
        return self.heap.cells[self.address]

    @expr.setter
    def expr(self, value):
        self.heap.cells[self.address] = value

    def __getitem__(self, index):
        if self.array and isinstance(index, tuple) and len(index) >= 3 and isinstance(index[2], slice):
            return Tensor(self.heap, self.address, True, index[2].stop, self.shape)
        return self

    def clone(self):
        if self.heap.clone_alias:
            return self
        value = self.expr
        if self.array and self.end is not None:
            i = z.Int("clone_index")
            value = z.Lambda(i, z.If(z.And(i >= 0, i < self.end), z.Select(value, i), 0))
        return self.heap.tensor(value, self.array, self.end, self.shape)

    def copy_(self, other):
        value = unwrap(other)
        if self.array and self.end is not None:
            i = z.Int("copy_index")
            value = z.Lambda(i, z.If(z.And(i >= 0, i < self.end), z.Select(value, i), z.Select(self.expr, i)))
        self.expr = value
        return self

    def fill_(self, value):
        self.expr = z.K(INT, unwrap(value)) if self.array else unwrap(value)
        return self

    def zero_(self):
        return self.fill_(z.IntVal(0))

    def add_(self, value):
        self.expr = self.expr + unwrap(value)
        return self

    def index_copy_(self, dimension, index, values):
        assert dimension == 2
        self.expr = z.Store(self.expr, unwrap(index), unwrap(values))
        return self

    def __add__(self, other):
        return self.heap.tensor(self.expr + unwrap(other))

    def __le__(self, other):
        i = z.Int("mask_index")
        return self.heap.tensor(z.Lambda(i, z.Select(self.expr, i) <= unwrap(other)), True)

    def view(self, *args):
        return self

    def expand(self, *args):
        return self

    def to(self, *args):
        return self

    def argmax(self, *args):
        return self

    def __bool__(self):
        raise TypeError("A tensor-dependent Python branch is outside the graph contract")


def unwrap(value):
    return value.expr if isinstance(value, Tensor) else value


class SymbolicTorch:
    bfloat16 = "bf16"

    def __init__(self, heap):
        self.heap = heap
        stream = SimpleNamespace(wait_stream=lambda *args: None)
        self.cuda = SimpleNamespace(Stream=lambda: stream, current_stream=lambda: stream,
                                    stream=lambda *args: nullcontext(), graph=lambda *args: nullcontext(),
                                    synchronize=lambda: None, CUDAGraph=lambda: SimpleNamespace())

    def arange(self, length, **kwargs):
        assert length == 1, "The verified decoder appends exactly one token"
        return self.heap.tensor(z.IntVal(0))

    def where(self, condition, yes, no):
        i = z.Int("where_index")
        return self.heap.tensor(z.Lambda(i, z.If(z.Select(condition.expr, i), int(yes), int(no))), True)

    def finfo(self, dtype):
        return SimpleNamespace(min=MINIMUM)


def method(source, class_name, name, torch, replace=None):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name))
    if replace:
        original = ast.unparse(node)
        mutated = replace(original)
        assert mutated != original, "Mutation did not change the extracted implementation"
        node = ast.parse(mutated).body[0]
    node.decorator_list = []
    node.returns = None
    for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
        arg.annotation = None
    namespace = {"torch": torch, "time": time}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "<symbolic-source>", "exec"), namespace)
    return namespace[name]


class Engine:
    def __init__(self, prefix, source, cache_source, cursor=None, delta=None, token=None, state=None, mutations=None):
        self.heap = Heap(prefix)
        self.torch = SymbolicTorch(self.heap)
        mutations = mutations or {}
        self.heap.clone_alias = mutations.get("clone_alias", False)
        self.work = SimpleNamespace(layers=[])
        self.variables = []
        for layer in range(LAYERS):
            values = []
            for kind in ("key", "value"):
                value = state[layer][len(values)] if state else z.Array(f"{prefix}_{kind}_{layer}", INT, INT)
                self.variables.append(value)
                values.append(self.heap.tensor(value, True))
            length = cursor if cursor is not None else z.Int(f"{prefix}_cursor")
            obj = SimpleNamespace(keys=values[0], values=values[1], cumulative_length=self.heap.tensor(length),
                                  is_initialized=True, device="cuda")
            obj.update = MethodType(method(cache_source, "StaticLayer", "update", self.torch), obj)
            obj.reset = MethodType(method(cache_source, "CacheLayerMixin", "reset", self.torch), obj)
            self.work.layers.append(obj)
        self.cursor = self.work.layers[0].cumulative_length.expr
        self.delta = self.heap.tensor(delta if delta is not None else z.Int(f"{prefix}_delta"))
        self.token = self.heap.tensor(token if token is not None else z.Int(f"{prefix}_token"))
        self.input_delta, self.input_token = self.delta.expr, self.token.expr
        i = z.Int("column_index")
        self.columns = self.heap.tensor(z.Lambda(i, i), True)
        self.model = SimpleNamespace(lm_head=lambda value: value)
        self.lm = self.language
        self.work.reset = lambda: [layer.reset() for layer in self.work.layers]
        self.store = SimpleNamespace(clear=lambda: None)
        for name in ("_save", "_restore", "_decode", "build_graph"):
            setattr(self, name, MethodType(method(source, "Runtime", name, self.torch, mutations.get(name)), self))

    def language(self, input_ids, position_ids, attention_mask, past_key_values, use_cache):
        token, position, mask = input_ids.expr, position_ids.expr, attention_mask.expr
        self.last_position, self.last_mask = position, mask
        # The deterministic neural step is opaque. All storage writes below execute
        # the installed Transformers StaticLayer.update method itself.
        for index, layer in enumerate(self.work.layers):
            key, value = layer.keys.expr, layer.values.expr
            args = (index, token, position, key, value, mask)
            new_key = self.heap.tensor(KEY(*args), shape=(1, 8, 1, 128))
            new_value = self.heap.tensor(VALUE(*args), shape=(1, 8, 1, 128))
            layer.update(new_key, new_value)
        last = self.work.layers[-1]
        predicted = NEXT(token, position, mask, last.keys.expr, last.values.expr)
        return SimpleNamespace(last_hidden_state=self.heap.tensor(predicted))

    def bindings(self):
        return [tensor.address for layer in self.work.layers for tensor in (layer.keys, layer.values, layer.cumulative_length)] + [
            self.token.address, self.delta.address, self.columns.address]

    def outputs(self):
        return [self.token.expr, self.last_position, self.last_mask] + [
            tensor.expr for layer in self.work.layers for tensor in (layer.keys, layer.values, layer.cumulative_length)]


def disjunction(expressions):
    return z.Or(*[z.simplify(expr) for expr in expressions])


def check_pair(output, name, title, safe, faulty, witnesses):
    records = []
    for kind, (premises, violation, variables) in [("safe", safe), ("negative", faulty)]:
        solver = z.Solver()
        solver.set(timeout=20000)
        solver.add(*premises)
        base = solver.check()
        assert base == z.sat, (name, kind, "vacuous or unknown premises", base)
        premise_file = output / f"{name}-{kind}-premises.smt2"
        premise_file.write_text(solver.to_smt2())
        solver.add(violation)
        result = solver.check()
        expected = z.unsat if kind == "safe" else z.sat
        query = output / f"{name}-{kind}.smt2"
        query.write_text(solver.to_smt2())
        counterexample = None
        if result == z.sat:
            model = solver.model()
            counterexample = {key: str(model.eval(value, model_completion=True)) for key, value in variables.items()}
        record = {"kind": kind, "result": str(result), "expected": str(expected), "premises": str(base),
                  "query": query.name, "query_sha256": sha(query), "premises_file": premise_file.name,
                  "premises_sha256": sha(premise_file), "counterexample": counterexample}
        records.append(record)
        assert result == expected, (name, kind, result, counterexample)
    print(name, "UNSAT / negative SAT", flush=True)
    return {"id": name, "title": title, "results": records, "source_mapping": witnesses}


def proofs(source, cache_source):
    cases = []
    # P1: equality of the *structural* cache identity, under a collision-free hash
    # abstraction. This is not a cryptographic proof of SHA256.
    a = z.Ints("model_a pixel_a grid_a prefix_tokens_a prefix_positions_a")
    b = z.Ints("model_b pixel_b grid_b prefix_tokens_b prefix_positions_b")
    kv = z.Function("prefix_kv_from_context", *([INT] * 5), INT)
    fields = [x == y for x, y in zip(a, b)]
    variables = {"positions_A": a[-1], "positions_B": b[-1], "KV_A": kv(*a), "KV_B": kv(*b)}
    cases.append(("P1-key", "缓存身份覆盖前缀状态的全部上下文", (fields, kv(*a) != kv(*b), variables),
                  (fields[:-1] + [a[-1] != b[-1]], kv(*a) != kv(*b), variables),
                  ["Runtime.__init__: identity", "Runtime.prepare: pixel/grid/prefix-token/prefix-position key inputs"]))
    # P2: one-layer induction lemma, for every integer prefix length and every
    # layer satisfying the premise, rather than a sample of neural weights.
    boundary, current, dependency = z.Ints("boundary current_position dependency_position")
    previous = z.Array("suffix_dependency_previous_layer", INT, z.BoolSort())
    premise = [boundary > 0, current >= 0, current < boundary,
               z.ForAll(dependency, z.Implies(z.And(dependency >= 0, dependency < boundary), z.Not(z.Select(previous, dependency))))]
    bad = z.Exists(dependency, z.And(dependency >= 0, dependency <= current, z.Select(previous, dependency)))
    unmasked = z.Exists(dependency, z.And(dependency >= 0, dependency <= boundary, z.Select(previous, dependency)))
    variables = {"boundary": boundary, "position": current, "suffix_dependency_at_boundary": z.Select(previous, boundary)}
    cases.append(("P2-causality", "因果依赖逐层保持前缀不依赖后续问题", (premise, bad, variables),
                  (premise + [z.Select(previous, boundary)], unmasked, variables),
                  ["Causal-attention/local-MLP contract; induction applies across 28 language layers",
                   "Runtime.prepare: no visual token after boundary; Runtime._language: prefix-only end"]))
    index = z.Int("arbitrary_prefix_cell")
    bounds = [LENGTH > 0, LENGTH < CAPACITY, index >= 0, index < LENGTH]
    # P3: execute the real clone/reset/restore operations, in all 28 layers.
    def isolation(alias):
        engine = Engine("isolation", source, cache_source, cursor=LENGTH, mutations={"clone_alias": alias})
        before = [tensor.expr for layer in engine.work.layers for tensor in (layer.keys, layer.values)]
        saved = engine._save(LENGTH)
        engine.work.reset()
        engine._restore(saved, LENGTH)
        # Universally arbitrary future K/V, rather than just one zeroing example.
        # A later request may overwrite even the entire working prefix.
        future = []
        for number, layer in enumerate(engine.work.layers):
            for kind, tensor in [("key", layer.keys), ("value", layer.values)]:
                value = z.Array(f"future_{kind}_{number}", INT, INT)
                future.append(value)
                tensor.copy_(engine.heap.tensor(value, True))
        after = [tensor.expr for pair in saved for tensor in pair]
        bad = disjunction(z.Select(old, index) != z.Select(new, index) for old, new in zip(before, after))
        premise = bounds + ([z.Select(before[0], index) != 0, z.Select(future[0], index) == 0] if alias else [])
        return premise, bad, {"prefix_length": LENGTH, "cell": index,
                             "original_K": z.Select(before[0], index), "saved_K_after_requests": z.Select(after[0], index)}
    cases.append(("P3-isolation", "工作区重置及后续请求不改写持久前缀", isolation(False), isolation(True),
                  ["Runtime._save/_restore", "Transformers.CacheLayerMixin.reset", "negative: clone returns alias"]))
    def restore(off_by_one):
        mutation = (lambda text: text.replace("fill_(length)", "fill_(length + 1)")) if off_by_one else None
        engine = Engine("restore", source, cache_source, cursor=z.IntVal(0), mutations={"_restore": mutation})
        payload = [(engine.heap.tensor(z.Array(f"saved_k_{i}", INT, INT), True, LENGTH),
                    engine.heap.tensor(z.Array(f"saved_v_{i}", INT, INT), True, LENGTH)) for i in range(LAYERS)]
        engine._restore(payload, LENGTH)
        violations = []
        for layer, (key, value) in zip(engine.work.layers, payload):
            violations += [z.Select(layer.keys.expr, index) != z.Select(key.expr, index),
                           z.Select(layer.values.expr, index) != z.Select(value.expr, index), layer.cumulative_length.expr != LENGTH]
        return bounds, disjunction(violations), {"prefix_length": LENGTH, "restored_cursor": engine.work.layers[0].cumulative_length.expr}
    cases.append(("P4-restore", "28层K/V恢复准确且游标回到前缀边界", restore(False), restore(True),
                  ["Runtime._restore: slices.copy_ and cumulative_length.fill_"]))
    # G1: derive the admission predicate from the actual prepare() Assert AST.
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    prepare = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    guard = next(n.test for n in ast.walk(prepare) if isinstance(n, ast.Assert) and "self.max_new_tokens" in ast.unparse(n.test))
    prompt, step = z.Ints("prompt_length decode_step")
    generation = read(PROJECT / "configs/protocol.lock.json")["runtime"]["generation"]["max_new_tokens"]
    env = {"cpu": {"input_ids": SimpleNamespace(shape=(1, prompt))},
           "self": SimpleNamespace(max_new_tokens=generation, max_length=CAPACITY)}
    admitted = eval(compile(ast.Expression(copy.deepcopy(guard)), "<actual-admission-guard>", "eval"), env)
    common = [prompt > 0, step >= 0, step < generation - 1]
    bad = z.Or(prompt + step < 0, prompt + step >= CAPACITY)
    variables = {"prompt_length": prompt, "decode_step": step, "write_slot": prompt + step, "capacity": z.IntVal(CAPACITY)}
    cases.append(("G1-capacity", "真实准入守卫保证最多63步decode写入不越界", (common + [admitted], bad, variables),
                  (common + [prompt <= CAPACITY], bad, variables),
                  ["Runtime.prepare: extracted admission assert", "Runtime.generate: at most max_new_tokens-1 decode calls",
                   "fixed_steps diagnostic calls must be <=64; first token comes from prefill"]))
    cursor, delta, token = z.Ints("live_cursor live_delta live_token")
    valid_decode = [cursor > 0, cursor < CAPACITY, delta >= -CAPACITY, delta <= CAPACITY,
                    token >= 0, token < 151936]
    def decode_engine(prefix="decode", **kwargs):
        return Engine(prefix, source, cache_source, cursor=cursor, delta=delta, token=token, **kwargs)
    def visibility(unmask):
        mutation = (lambda text: text.replace("self.columns <= cursor", "self.columns <= cursor + 1")) if unmask else None
        engine = decode_engine(mutations={"_decode": mutation})
        engine._decode()
        column = z.Int("observed_column")
        actual = z.Select(engine.last_mask, column) == 0
        expected = column < engine.work.layers[0].cumulative_length.expr
        return valid_decode + [column >= 0, column < CAPACITY], actual != expected, {
            "cursor_before_write": cursor, "cursor_after_write": engine.work.layers[0].cumulative_length.expr,
            "column": column, "visible": actual}
    cases.append(("G2-mask", "单步写入后mask恰好覆盖有效KV，不暴露未来槽位", visibility(False), visibility(True),
                  ["Runtime._decode: device cursor / columns<=cursor / torch.where", "StaticLayer.update: add_ and index_copy_"]))
    def replay(freeze_cursor):
        captured = Engine("capture", source, cache_source)
        captured._decode()
        live = decode_engine("live")
        live._decode()
        substitutions = list(zip(captured.variables, live.variables)) + [(captured.input_token, token)]
        if not freeze_cursor:
            substitutions += [(captured.cursor, cursor), (captured.input_delta, delta)]
        replayed = [z.substitute(expr, substitutions) for expr in captured.outputs()]
        bad = disjunction(a != b for a, b in zip(replayed, live.outputs()))
        premises = valid_decode + [captured.cursor > 0, captured.cursor < CAPACITY,
                                   captured.input_delta >= -CAPACITY, captured.input_delta <= CAPACITY]
        if freeze_cursor:
            premises += [captured.cursor != cursor]
        return premises, bad, {"captured_cursor": captured.cursor, "live_cursor": cursor,
                                "captured_delta": captured.input_delta, "live_delta": delta,
                                "replay_position": replayed[1], "eager_position": live.last_position}
    cases.append(("G3-replay", "固定绑定下图数据流重放与eager单步迁移相同", replay(False), replay(True),
                  ["Runtime._decode and StaticLayer.update executed symbolically",
                   "Graph replay substitutes CURRENT buffer contents into the captured tensor DAG",
                   "negative: freeze capture-time cursor/delta; CUDA faithfully replaying the DAG is an assumption"]))
    def addresses(rebind):
        mutation = (lambda text: text.replace("self.token.copy_(logits[:, -1].argmax(-1).view(1, 1))",
                                              "self.token = logits[:, -1].argmax(-1).view(1, 1)")) if rebind else None
        engine = decode_engine(mutations={"_decode": mutation})
        original = engine.bindings()
        saved = engine._save(LENGTH)
        engine._restore(saved, LENGTH)
        engine._decode()
        changed = z.BoolVal(original != engine.bindings())
        return [LENGTH > 0, LENGTH < CAPACITY], changed, {
            "captured_token_address": z.IntVal(original[-3]), "current_token_address": z.IntVal(engine.bindings()[-3])}
    cases.append(("G4-addresses", "恢复与解码保持全部KV、游标及输入缓冲地址", addresses(False), addresses(True),
                  ["Runtime._restore/_decode", "StaticLayer.update/reset", "negative: replace token tensor instead of copy_"]))
    def rollback(skip_restore):
        mutation = (lambda text: text.rsplit("self._restore(snapshot, length)", 1)[0] +
                    text.rsplit("self._restore(snapshot, length)", 1)[1]) if skip_restore else None
        engine = decode_engine(mutations={"build_graph": mutation})
        snapshot = [(layer.keys.expr, layer.values.expr) for layer in engine.work.layers]
        original_token, original_delta = engine.token.expr, engine.delta.expr
        bindings = engine.bindings()
        # _prefill is the contract boundary: a successful prefill yields these KV,
        # token and delta contents. The rest of build_graph is the actual source.
        engine._prefill = lambda request, prefix: None
        request = SimpleNamespace(inputs={"input_ids": SimpleNamespace(shape=(1, cursor))})
        engine.build_graph(request)
        column = z.Int("rollback_column")
        violations = [engine.token.expr != original_token, engine.delta.expr != original_delta,
                      z.BoolVal(engine.bindings() != bindings)]
        for layer, (key, value) in zip(engine.work.layers, snapshot):
            violations += [layer.cumulative_length.expr != cursor,
                           z.Select(layer.keys.expr, column) != z.Select(key, column),
                           z.Select(layer.values.expr, column) != z.Select(value, column)]
        premises = [cursor > 0, cursor + generation <= CAPACITY, column >= 0, column < cursor]
        return premises, disjunction(violations), {"prefill_length": cursor,
                "length_after_build": engine.work.layers[0].cumulative_length.expr,
                "token_before": original_token, "token_after": engine.token.expr}
    cases.append(("G5-rollback", "预热和图捕获后恢复有效KV、首token与游标", rollback(False), rollback(True),
                  ["Runtime.build_graph: actual 3 warmups, restore, capture, restore, token.copy_",
                   "Prefill success and CUDA stream/capture ordering are contracts"]))
    return cases


def source_contract(source):
    tree = ast.parse(source)
    runtime = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runtime")
    methods = {n.name: ast.unparse(n) for n in runtime.body if isinstance(n, ast.FunctionDef)}
    prepare = methods["prepare"]
    required = ["locations.numel() == 1", "boundary < cpu['input_ids'].shape[1]",
                "not cpu['mm_token_type_ids'][:, boundary:].any()", "row['pixel_sha256']",
                "metadata['grid_thw']", "cpu['input_ids'][:, :boundary]", "positions[:, :, :boundary]"]
    assert all(part in prepare for part in required), "Prefix-key or boundary contract changed"
    assert all(name in methods["__init__"] for name in ["'model'", "'processor'", "'dtype'", "'attention'", "'transformers'"])
    assert "range(3)" in methods["build_graph"]
    assert "(fixed_steps or self.max_new_tokens) - 1" in methods["generate"]
    assert "tokens[-1] in self.eos" in methods["generate"]
    assert "self.work.reset()" in methods["_prefill"]
    assert "self._language(request, start=request.boundary)" in methods["_prefill"]
    assert "self.delta.copy_(request.delta.reshape(1))" in methods["_prefill"]
    assert "early_initialization(1," in methods["__init__"]
    assert not any(word in methods["_decode"] for word in [".item(", ".cpu(", "if "])
    return {"prefix_components_checked": required, "decode_host_scalar_or_python_branch": False,
            "methods_executed": ["_save", "_restore", "_decode", "build_graph", "StaticLayer.update", "CacheLayerMixin.reset"]}


def replay_existing(output):
    summary = read(output / "summary.json")
    assert summary["passed"]
    for path, expected in summary["source_sha256"].items():
        actual = (Path(metadata.distribution("transformers").locate_file("transformers/cache_utils.py"))
                  if path == "@transformers/cache_utils.py" else PROJECT / path)
        assert sha(actual) == expected, ("Proof source changed", path)
    count = 0
    for case in summary["cases"]:
        for record in case["results"]:
            for filename, digest, expected in [(record["query"], record["query_sha256"], record["expected"]),
                                               (record["premises_file"], record["premises_sha256"], "sat")]:
                path = output / filename
                assert sha(path) == digest
                solver = z.Solver(); solver.set(timeout=20000); solver.from_file(str(path))
                assert str(solver.check()) == expected, filename
                count += 1
    print(json.dumps({"passed": True, "replayed_smt2_files": count, "safety_unsat": len(summary["cases"]),
                      "negative_sat": len(summary["cases"]), "scope": summary["scope"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(); output = args.output.resolve()
    if args.replay:
        replay_existing(output); return
    if output.exists() and any(output.iterdir()):
        parser.error("Use --replay or a new output directory; existing proof artifacts are preserved")
    output.mkdir(parents=True, exist_ok=True)
    source = RUNTIME.read_text()
    cache_path = Path(metadata.distribution("transformers").locate_file("transformers/cache_utils.py"))
    cache_source = cache_path.read_text()
    model_config = PROJECT / "env/qwen3-vl-2b-config.json"
    text_config = read(model_config)["text_config"]
    assert text_config["num_hidden_layers"] == LAYERS and text_config["vocab_size"] == 151936
    assert read(PROJECT / "configs/protocol.lock.json")["runtime"]["generation"]["max_new_tokens"] == 64
    assert sha(RUNTIME) == read(HERE.parent / "results/pilot/manifest.json")["source_sha256"]["runtime.py"]
    contract = source_contract(source)
    records = [check_pair(output, *case) for case in proofs(source, cache_source)]
    dependencies = [RUNTIME, Path(__file__).resolve(), PROJECT / "configs/protocol.lock.json", PROJECT / "configs/vlm-assets.lock.json", model_config]
    source_hashes = {str(path.relative_to(PROJECT)): sha(path) for path in dependencies}
    source_hashes["@transformers/cache_utils.py"] = sha(cache_path)
    summary = {"passed": True, "created_at": datetime.now(timezone.utc).isoformat(),
               "scope": "Source-linked symbolic state/control/dataflow contracts; causal-dependency induction; not BF16 neural arithmetic or CUDA driver/kernel verification",
               "python": sys.version, "z3_version": z.get_version_string(), "z3_solver_package": metadata.version("z3-solver"),
               "transformers": metadata.version("transformers"), "capacity": CAPACITY, "layers": LAYERS,
               "max_new_tokens": 64, "safe_unsat": len(records), "negative_sat": len(records),
               "nonvacuity_sat_checks": 2 * len(records), "source_sha256": source_hashes,
               "source_contract": contract, "cases": records,
               "assumptions": ["Successful disjoint allocation, correct slice/clone/copy_/index_copy_ tensor semantics",
                               "Structural cache identities are collision-free abstractions of SHA256; identical deterministic preprocessing/visual features",
                               "Language layers obey causal attention, token-local operations and supplied mask",
                               "B=1, 28 initialized layers, capacity1024, output limit64; fixed_steps<=64",
                               "Valid prepared positions fit int64; tested mRoPE delta is modeled within [-1024,1024]",
                               "CUDA streams preserve ordering and CUDA Graph faithfully replays captured tensor operations at bound addresses",
                               "Neural step is a deterministic uninterpreted function; BF16 numerical differences and answer accuracy are measured separately"]}
    write(output / "summary.json", summary)
    shutil.copy2(RUNTIME, output / "runtime-snapshot.py")
    shutil.copy2(Path(__file__), output / "verifier-snapshot.py")
    shutil.copy2(cache_path, output / "transformers-cache-snapshot.py")
    write(output / "environment.json", {key: summary[key] for key in ["python", "z3_version", "z3_solver_package", "transformers"]})
    table = "\n".join(f"| {case['id']} | {case['title']} | UNSAT | SAT |" for case in records)
    faults = {"P1-key": "缓存键漏掉前缀位置", "P2-causality": "取消因果可见范围", "P3-isolation": "clone退化为共享存储",
              "P4-restore": "恢复游标偏移1", "G1-capacity": "准入不预留生成预算", "G2-mask": "mask多开放一个未来槽位",
              "G3-replay": "冻结捕获时的游标和delta", "G4-addresses": "重新绑定token张量", "G5-rollback": "省去捕获后的KV恢复"}
    examples = "\n".join(f"| {case['id']} | {faults[case['id']]} | `{json.dumps(case['results'][1]['counterexample'], ensure_ascii=False)}` |"
                         for case in records)
    report = f"""# 前缀KV与CUDA Graph形式化验证\n\n{len(records)}项安全性质均为UNSAT，{len(records)}项故障负控制均为SAT；另有{2*len(records)}项前提可满足性检查通过。每项SMT2、前提SMT2和具体反例均保存。\n\n| 性质 | 证明对象 | 安全查询 | 负控制 |\n| --- | --- | --- | --- |\n{table}\n\n求解器z3-solver {summary['z3_solver_package']}，Transformers {summary['transformers']}。原型runtime SHA256为`{sha(RUNTIME)}`。\n\n验证直接抽取并执行Runtime的_save、_restore、_decode、build_graph及已安装StaticLayer.update、CacheLayerMixin.reset，在符号张量语义下构造状态迁移。P2是任意前缀长度／任意满足前提的语言层的因果依赖归纳步；28层通过归纳应用。所有28层存储在P3/P4和G类迁移中展开。单个KV标量代表任意头／通道的元素，复制与索引操作对这些元素逐坐标成立。\n\nG3把实际_decode生成的符号张量数据流保存为捕获图，再用当前绑定缓冲的值代入；与重新执行源码得到的eager迁移比较。负控制冻结捕获时游标和delta，能够找到不同长度请求使用旧位置的反例。G5执行真实建图控制流；成功prefill、流顺序和CUDA捕获语义是接口前提。\n\n形式化结论覆盖前缀身份、因果依赖、所有权隔离、恢复、静态容量、mask、地址和图数据流迁移。SHA256无碰撞、PyTorch张量算子及CUDA Graph按接口执行属于可信前提；神经网络数值以确定性未解释函数表示，不据此证明BF16逐位等价或GPU驱动／kernel正确。已有GPU状态检查、真实答案和ANLS门槛补充验证具体设备执行。\n\n## 复现\n\n从仓库根目录执行`./stage_2/experiments/cross_layer_v2/run.sh formal`重放全部36份SMT2。重新建模使用同一入口的`--output 新目录`，不覆盖已保存证据。无需GPU、模型权重、网络或额外求解服务。环境复用stage_2/.venv中的z3-solver=={summary['z3_solver_package']}；输出含版本、源码哈希和快照。\n"""
    report += "\n## 故障负控制与反例\n\n以下是抽象状态反例，地址数字为符号堆的地址标识。它们表示移除保护后的错误条件，不是正常GPU实验发生了这些故障。\n\n| 性质 | 注入故障 | 求解器反例 |\n| --- | --- | --- |\n" + examples + "\n"
    (output / "形式化验证报告.md").write_text(report)
    print(json.dumps({"passed": True, "safe_unsat": len(records), "negative_sat": len(records), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
