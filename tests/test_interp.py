import pytest

from lucid.errors import ResourceError, RuntimeError_
from lucid.interp import ExecConfig, run
from lucid.parser import parse
from lucid.typechecker import typecheck
from lucid.values import RecordVal, VariantVal


def _run(src, inputs, cfg=None):
    m = parse(src)
    typecheck(m)
    return run(m, inputs, cfg).value


def test_sum():
    src = ("fn @sum ($xs : #List[#Int]) -> #Int = do var $a : #Int = 0 ; "
           "foreach $x : #Int in $xs do set $a = add($a, $x) ; end foreach ; return $a ; end @sum ;")
    assert _run(src, [[1, 2, 3, 4]]) == 10
    assert _run(src, [[]]) == 0


def test_factorial_via_fold():
    src = ("fn @m ($a : #Int, $b : #Int) -> #Int = do return mul($a, $b) ; end @m ; "
           "fn @fact ($n : #Int) -> #Int = do return foldl(range(1, add($n, 1)), 1, @m) ; end @fact ;")
    assert _run(src, [5]) == 120
    assert _run(src, [0]) == 1


def test_clamp():
    src = "fn @c ($x : #Int, $lo : #Int, $hi : #Int) -> #Int = do return min(max($x, $lo), $hi) ; end @c ;"
    assert _run(src, [15, 0, 10]) == 10
    assert _run(src, [-5, 0, 10]) == 0
    assert _run(src, [5, 0, 10]) == 5


def test_records_and_fields():
    src = ("record #P = { %x : #Int, %y : #Int } ; "
           "fn @mh ($p : #P, $q : #P) -> #Int = do return add(abs(sub(get %x($p), get %x($q))), "
           "abs(sub(get %y($p), get %y($q)))) ; end @mh ;")
    p = RecordVal("P", (("x", 1), ("y", 2)))
    q = RecordVal("P", (("x", 4), ("y", 6)))
    assert _run(src, [p, q]) == 7


def test_variant_match():
    src = ("variant #O = Some(#Int) | None() ; "
           "fn @u ($o : #O, $d : #Int) -> #Int = do return match #O $o of "
           "case Some($v : #Int) -> $v case None() -> $d end match ; end @u ;")
    assert _run(src, [VariantVal("O", "Some", (9,)), 0]) == 9
    assert _run(src, [VariantVal("O", "None", ()), -1]) == -1


def test_div_mod_by_zero_total():
    src = "fn @f ($a : #Int) -> #Int = do return add(div($a, 0), mod($a, 0)) ; end @f ;"
    assert _run(src, [10]) == 0  # both defined as 0


def test_cond_is_lazy():
    # else branch divides by zero in value terms but is not taken
    src = "fn @f ($b : #Bool) -> #Int = do return cond $b then 1 else div(1, 0) end cond ; end @f ;"
    assert _run(src, [True]) == 1


def test_determinism_same_inputs():
    src = "fn @f ($xs : #List[#Int]) -> #Int = do return length(reverse($xs)) ; end @f ;"
    a = _run(src, [[1, 2, 3]])
    b = _run(src, [[1, 2, 3]])
    assert a == b == 3


def test_resource_bound_int_blowup():
    src = ("fn @b ($n : #List[#Int]) -> #Int = do var $a : #Int = 2 ; "
           "foreach $x : #Int in $n do set $a = mul($a, $a) ; end foreach ; return $a ; end @b ;")
    with pytest.raises(ResourceError):
        _run(src, [list(range(200))])


def test_resource_bound_list_blowup():
    src = ("fn @b ($n : #List[#Int]) -> #Int = do var $a : #List[#Int] = $n ; "
           "foreach $x : #Int in $n do set $a = concat_list($a, $a) ; end foreach ; "
           "return length($a) ; end @b ;")
    with pytest.raises(ResourceError):
        _run(src, [list(range(40))], ExecConfig(max_list=2000))


def test_trace_records_steps():
    src = ("fn @f ($xs : #List[#Int]) -> #Int = do var $a : #Int = 0 ; "
           "foreach $x : #Int in $xs do set $a = add($a, $x) ; end foreach ; return $a ; end @f ;")
    m = parse(src)
    typecheck(m)
    res = run(m, [[1, 2]], ExecConfig(trace=True))
    events = [e["event"] for e in res.trace]
    assert "call" in events and "return" in events and "iter" in events
    assert res.value == 3
