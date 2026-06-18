import glob
import os

import pytest

from lucid.errors import TypeError_
from lucid.parser import parse
from lucid.typechecker import typecheck

EX_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")
EXAMPLES = sorted(glob.glob(os.path.join(EX_DIR, "*.lucid")))


@pytest.mark.parametrize("path", EXAMPLES)
def test_examples_typecheck(path):
    with open(path) as f:
        typecheck(parse(f.read()))


# Each of these MUST be rejected by the static checker.
BAD = {
    "type_mismatch": "fn @f () -> #Int = do return true ; end @f ;",
    "set_immutable": "fn @f ($x : #Int) -> #Int = do let $y : #Int = 1 ; set $y = 2 ; return $y ; end @f ;",
    "set_param": "fn @f ($x : #Int) -> #Int = do set $x = 2 ; return $x ; end @f ;",
    "no_return": "fn @f ($x : #Int) -> #Int = do let $y : #Int = 1 ; end @f ;",
    "recursion": "fn @f ($x : #Int) -> #Int = do return @f($x) ; end @f ;",
    "forward_ref": "fn @a () -> #Int = do return @b() ; end @a ; fn @b () -> #Int = do return 1 ; end @b ;",
    "undeclared_var": "fn @f () -> #Int = do return $z ; end @f ;",
    "nonexhaustive": "variant #C = A() | B() ; fn @f ($c : #C) -> #Int = do return match #C $c of case A() -> 1 end match ; end @f ;",
    "dup_case": "variant #C = A() | B() ; fn @f ($c : #C) -> #Int = do return match #C $c of case A() -> 1 case A() -> 2 case B() -> 3 end match ; end @f ;",
    "dead_code": "fn @f () -> #Int = do return 1 ; let $y : #Int = 2 ; end @f ;",
    "bad_builtin_arg": "fn @f ($s : #Str) -> #Int = do return add($s, 1) ; end @f ;",
    "arity_mismatch": "fn @f () -> #Int = do return add(1) ; end @f ;",
    "wrong_field": "record #P = { %x : #Int } ; fn @f ($p : #P) -> #Int = do return get %y($p) ; end @f ;",
    "cond_branch_mismatch": "fn @f ($b : #Bool) -> #Int = do return cond $b then 1 else true end cond ; end @f ;",
    "shadowing": "fn @f ($x : #Int) -> #Int = do let $x : #Int = 1 ; return $x ; end @f ;",
    "foreach_not_list": "fn @f ($x : #Int) -> #Int = do foreach $i : #Int in $x do end foreach ; return 0 ; end @f ;",
    "map_wrong_fn": "fn @bad ($x : #Bool) -> #Int = do return 1 ; end @bad ; fn @f ($xs : #List[#Int]) -> #List[#Int] = do return map($xs, @bad) ; end @f ;",
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_illtyped_rejected(name):
    with pytest.raises(TypeError_):
        typecheck(parse(BAD[name]))


def test_match_must_bind_correct_payload_count():
    src = ("variant #C = A(#Int) ; "
           "fn @f ($c : #C) -> #Int = do return match #C $c of case A() -> 1 end match ; end @f ;")
    with pytest.raises(TypeError_):
        typecheck(parse(src))
