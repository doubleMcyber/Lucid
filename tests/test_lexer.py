import pytest

from lucid import lexer as lx
from lucid.errors import LexError


def kinds(src):
    return [t.kind for t in lx.tokenize(src) if t.kind != lx.EOF]


def test_sigil_classes_are_context_free():
    toks = lx.tokenize("$x @f #Int %field")
    assert [(t.kind, t.value) for t in toks[:4]] == [
        (lx.LOCAL, "x"), (lx.FUNC, "f"), (lx.TYPE, "Int"), (lx.FIELD, "field"),
    ]


def test_bare_word_is_name_regardless_of_role():
    # 'end' and 'add' are both just NAME tokens; role is a parse concern.
    assert kinds("end add foreach") == [lx.NAME, lx.NAME, lx.NAME]


def test_int_literal():
    toks = lx.tokenize("123 0")
    assert toks[0].kind == lx.INT and toks[0].value == "123"


def test_no_negative_literal_minus_is_only_arrow():
    with pytest.raises(LexError):
        lx.tokenize("-3")
    # arrow is fine
    assert lx.tokenize("->")[0].kind == lx.ARROW


def test_malformed_number_touching_letters():
    with pytest.raises(LexError):
        lx.tokenize("12abc")


def test_raw_string_basic():
    toks = lx.tokenize('"hello"')
    assert toks[0].kind == lx.STR and toks[0].value == "hello"


def test_raw_string_with_embedded_quote_uses_hash_fence():
    # #"...."# lets the content contain a bare quote
    toks = lx.tokenize('#"a "b" c"#')
    assert toks[0].kind == lx.STR
    assert toks[0].value == 'a "b" c'


def test_hash_disambiguates_type_vs_string():
    toks = lx.tokenize('#Int #"s"#')
    assert toks[0].kind == lx.TYPE and toks[0].value == "Int"
    assert toks[1].kind == lx.STR and toks[1].value == "s"


def test_unterminated_string():
    with pytest.raises(LexError):
        lx.tokenize('"oops')


def test_whitespace_is_insignificant():
    a = kinds("add ( $x , $y )")
    b = kinds("add($x,$y)")
    assert a == b
