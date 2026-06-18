# Lucid Language Specification v0.1.0

Lucid is a model-native language: locally decodable, whitespace-insensitive,
free of overloaded tokens and optional syntax, with a single canonical form per
program and a bijective surface↔AST mapping. This document is the normative
description of v1; it matches the implementation in `lucid/`.

## 1. Design invariants (PRD §7)

- **No context-dependent lexing.** A token's class is fixed by its own leading
  character(s): `$`→local, `@`→function, `#`→type (or a string fence), `%`→field,
  a digit→int, a bare word→`NAME` (keyword/builtin/tag), a quote/`#"`→string.
- **Whitespace is non-semantic.** It only separates tokens. The canonical printer
  fixes a layout; the parser ignores it.
- **No infix operators / no precedence.** Every operation is a prefix named call:
  `add($a, $b)`, never `$a + $b`. Negative integers use `neg(...)`.
- **No optional syntax.** Mandatory `;` terminators; explicit block open (`do`/
  `{`/`[`) and self-naming close (`end @f`, `end foreach`, `end if`, `end match`,
  `end cond`).
- **Kind sigils.** `$` local, `@` function, `#` type, `%` field — an identifier's
  kind is read off the token itself.
- **Explicit types everywhere.** Every binding, parameter, loop variable, and
  match binder is annotated.
- **Escape-free strings.** Raw, self-delimiting hash-fenced literals (below).
- **Single canonical form / bijective.** Exactly one text per AST;
  `parse(print(ast)) == ast` and `print(parse(canonical)) == canonical`.

## 2. Lexical grammar

```
LOCAL  ::= '$' ident
FUNC   ::= '@' ident
TYPE   ::= '#' ident                  ; '#' immediately followed by a letter/_
FIELD  ::= '%' ident
NAME   ::= ident                      ; bare word: keyword | builtin | variant tag
INT    ::= digit+                     ; non-negative only
STR    ::= '#'{k} '"' chars '"' '#'{k}   ; k>=0; closer is '"' then k '#'
ident  ::= (letter | '_') (letter | digit | '_')*
```

**String literals** are raw and escape-free. A literal opens with `k≥0` `#`
followed by `"`, and closes with `"` followed by the same `k` `#`. The content is
taken verbatim. The canonical spelling uses the minimal `k` such that the content
cannot contain the closing delimiter (so `plain` → `"plain"`, `a"b` → `#"a"b"#`).
`-` is **not** an operator and appears only inside `->`.

## 3. Concrete grammar

```
program      ::= decl*
decl         ::= record_decl | variant_decl | fn_decl

record_decl  ::= 'record' TYPE '=' '{' [field (',' field)*] '}' ';'
field        ::= FIELD ':' type
variant_decl ::= 'variant' TYPE '=' ctor ('|' ctor)* ';'
ctor         ::= NAME '(' [type (',' type)*] ')'
fn_decl      ::= 'fn' FUNC '(' [param (',' param)*] ')' '->' type
                 '=' 'do' stmt* 'end' FUNC ';'        ; closing FUNC must match
param        ::= LOCAL ':' type

type         ::= '#Int' | '#Bool' | '#Str'
               | '#List' '[' type ']'
               | '#Fn' '[' '(' [type (',' type)*] ')' '->' type ']'
               | TYPE                                  ; user type, declared earlier

stmt         ::= let | var | set | foreach | ifstmt | return
let          ::= 'let' LOCAL ':' type '=' expr ';'
var          ::= 'var' LOCAL ':' type '=' expr ';'
set          ::= 'set' LOCAL '=' expr ';'
foreach      ::= 'foreach' LOCAL ':' type 'in' expr 'do' stmt* 'end' 'foreach' ';'
ifstmt       ::= 'if' expr 'do' stmt* 'else' stmt* 'end' 'if' ';'
return       ::= 'return' expr ';'

expr         ::= INT | STR | 'true' | 'false'
               | LOCAL                                 ; $x
               | FUNC                                  ; @f as a value
               | FUNC '(' args ')'                     ; @f(...) user call
               | NAME '(' args ')'                     ; builtin call
               | 'list' '[' type ']' '(' args ')'
               | 'new' TYPE '{' [finit (',' finit)*] '}'
               | 'get' FIELD '(' expr ')'
               | 'tag' TYPE NAME '(' args ')'
               | 'match' TYPE expr 'of' arm+ 'end' 'match'
               | 'cond' expr 'then' expr 'else' expr 'end' 'cond'
args         ::= [expr (',' expr)*]
finit        ::= FIELD '=' expr
arm          ::= 'case' NAME '(' [binder (',' binder)*] ')' '->' expr
binder       ::= LOCAL ':' type
```

Keywords: `fn do end let var set foreach in if else return record variant
true false cond then match of case list new tag get`.

## 4. Type system (PRD §7.4)

Static, strong, monomorphic: no implicit coercion, no subtyping, no user-facing
overloading, no generics in v1. Types: `#Int`, `#Bool`, `#Str`, `#List[T]`,
records (nominal), tagged variants (nominal), and first-class function types
`#Fn[(T,…)->R]`. User types are **non-recursive** (a field/payload may only
reference earlier-declared types), which keeps every value finite.

The only polymorphism is in the built-in list intrinsics (`length`, `map`,
`foldl`, …), resolved by a one-shot structural matcher.

Static rules enforced: declare-before-use, **no recursion** (a function or `@f`
value may only reference earlier functions ⇒ the call graph is a DAG), guaranteed
return on all paths, no dead code after a terminator, exhaustive `match`, `set`
only on `var` bindings, and no shadowing.

## 5. Semantics (PRD §7.5)

A program is a pure function from inputs to an output (no ambient I/O, no
nondeterminism). It is **total**: the DAG call graph plus finite `foreach` (the
only iteration) guarantee termination. The interpreter additionally enforces
resource bounds (work, integer magnitude, string/list size, call depth); a run
that trips a bound raises `ResourceError` and is discarded by Loom. Built-ins are
total — integer `div`/`mod` by zero are defined to be `0`; partial list access is
exposed only via total `*_or` variants.

## 6. Built-ins

Arithmetic `add sub mul div mod neg abs min max`; comparisons
`lt_int le_int gt_int ge_int eq_int ne_int`; booleans `and or not xor eq_bool
ne_bool`; strings `concat len_str eq_str ne_str to_str_int to_str_bool`; lists
`length is_empty append concat_list reverse get_or head_or range map filter
foldl`; application `apply1 apply2`.

## 7. Canonical form

2-space indentation; one statement per line; declarations separated by a blank
line; expressions on a single line; record-literal fields in declaration order;
minimal string fence. The printer in `lucid/printer.py` is the normative
definition; `print(parse(text))` for any valid `text` yields the canonical form.

## 8. Example

```
fn @sum ($xs : #List[#Int]) -> #Int = do
  var $acc : #Int = 0 ;
  foreach $x : #Int in $xs do
    set $acc = add($acc, $x) ;
  end foreach ;
  return $acc ;
end @sum ;
```
