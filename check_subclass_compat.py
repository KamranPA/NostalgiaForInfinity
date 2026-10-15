#!/usr/bin/env python3
"""Static compatibility check: custom subclass vs. its (freshly synced) upstream base.

No freqtrade / TA-Lib needed (pure AST), so it runs in seconds on a bare runner.

Usage: check_subclass_compat.py SUB.py SubClass BASE.py BaseClass [...more quadruples]
Exit 0 = OK (warnings may still print), exit 1 = hard failure.

HARD failures (would break or silently disable the subclass):
  1. A class attribute the subclass overrides no longer exists in the base
     (override would silently do nothing, e.g. a renamed config attribute).
  2. A method the subclass overrides exists in the base but its positional
     parameter names/order changed (wrapper's super() call would break).
  3. The subclass calls self.<x>() where x looks like a base indicator helper
     (contains "indicators" or starts with "btc_") but neither class defines it.
WARNINGS (data-quality only, never block): column-name string literals read
by the subclass (safe_get("...")) that no longer appear anywhere in the base.
"""
import ast
import re
import sys

# freqtrade IStrategy callbacks: legitimately absent from the base file.
ISTRATEGY_METHODS = {
    "version", "order_filled", "adjust_trade_position", "bot_loop_start",
    "custom_exit", "custom_stake_amount", "confirm_trade_entry",
    "confirm_trade_exit", "leverage", "custom_entry_price", "custom_exit_price",
    "populate_indicators", "populate_entry_trend", "populate_exit_trend",
    "check_entry_timeout", "check_exit_timeout", "__init__",
}


def load_class(path, name):
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node, src
    raise SystemExit(f"class {name} not found in {path}")


def members(cls):
    attrs, funcs = set(), {}
    for n in cls.body:
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    attrs.add(t.id)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            attrs.add(n.target.id)
        elif isinstance(n, ast.FunctionDef):
            a = n.args
            funcs[n.name] = [x.arg for x in a.posonlyargs + a.args if x.arg != "self"]
    return attrs, funcs


def self_calls(cls):
    out = set()
    for n in ast.walk(cls):
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "self"):
            out.add(n.func.attr)
    return out


def check(sub_path, sub_name, base_path, base_name):
    sub, sub_src = load_class(sub_path, sub_name)
    base, base_src = load_class(base_path, base_name)
    s_attrs, s_funcs = members(sub)
    b_attrs, b_funcs = members(base)
    errors, warns = [], []

    for a in sorted(s_attrs):
        if a not in b_attrs and a not in b_funcs and not a.startswith("_"):
            # attribute that only the subclass owns (e.g. guard_*) is fine;
            # only flag overrides of things the base is expected to own.
            if a.startswith(("guard_",)):
                continue
            errors.append(f"attribute '{a}' overridden in {sub_name} but missing in {base_name}")
    for f, params in s_funcs.items():
        if f in b_funcs:
            if params != b_funcs[f]:
                errors.append(f"method '{f}' signature changed: {sub_name}{params} vs {base_name}{b_funcs[f]}")
        elif f not in ISTRATEGY_METHODS and not f.startswith("_"):
            errors.append(f"method '{f}' overridden in {sub_name} but missing in {base_name}")
    for c in sorted(self_calls(sub)):
        risky = "indicators" in c or c.startswith("btc_")
        if risky and c not in s_funcs and c not in b_funcs:
            errors.append(f"{sub_name} calls self.{c}() but neither class defines it")

    for col in sorted(set(re.findall(r'safe_get\(\s*\w+\s*,\s*"([^"]+)"', sub_src))):
        # Timeframe suffixes (_1h, _4h ...) are appended dynamically by the
        # base, so look for the stem (e.g. BTC_EMA_20) rather than the full name.
        stem = re.sub(r"_(5m|15m|1h|4h|1d)$", "", col)
        if stem not in base_src:
            warns.append(f"column '{col}' read by {sub_name} no longer appears in {base_name}")
    return errors, warns


def main(argv):
    if len(argv) < 4 or len(argv) % 4:
        raise SystemExit(__doc__)
    bad = False
    for i in range(0, len(argv), 4):
        sp, sn, bp, bn = argv[i:i + 4]
        errors, warns = check(sp, sn, bp, bn)
        print(f"== {sn} vs {bn}: {len(errors)} error(s), {len(warns)} warning(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        for w in warns:
            print(f"  warn:  {w}")
        bad |= bool(errors)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
