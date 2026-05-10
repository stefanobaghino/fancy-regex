// Copyright 2026 The Fancy Regex Authors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//! Seek pre-filter: build a simplified approximation of a pattern used to
//! skip to plausible match positions before handing off to the backtracking VM.

use alloc::format;
use alloc::string::String;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as Map;
#[cfg(feature = "std")]
use std::collections::HashMap as Map;

use crate::analyze::Info;
use crate::compile::MAX_SUBROUTINE_RECURSION_DEPTH;
use crate::{write_quantifier, Absent, Assertion, Expr};

/// Returns `true` if the expression tree contains a `StartText` or `EndText` assertion anywhere.
///
/// This is used to decide whether it is safe to inline a group body when approximating a backref:
/// text-anchor assertions in an inlined body would be evaluated at the wrong string position and
/// could cause false negatives (missing valid match positions).
pub(crate) fn expr_contains_positional_anchor(expr: &Expr) -> bool {
    match expr {
        // StartText (^), EndText ($), StartLine ((?m)^), EndLine ((?m)$) are positional anchors
        // that depend on absolute position in the string.  When a group containing such anchors
        // is inlined for a backref approximation, the anchors would be evaluated at the wrong
        // position (the backref position rather than the original capture position), potentially
        // producing false negatives.
        Expr::Assertion(
            Assertion::StartText
            | Assertion::EndText
            | Assertion::EndTextIgnoreTrailingNewlines { .. }
            | Assertion::StartLine { .. }
            | Assertion::EndLine { .. },
        ) => true,
        _ => expr.children_iter().any(expr_contains_positional_anchor),
    }
}

/// Emit a permissive size placeholder — `(?s:.)+` or `(?s:.){N,}` — into `buf`.
///
/// Used when a hard node cannot be represented in the seek pattern but has a known minimum
/// size.  The placeholder is an over-approximation that avoids false negatives: it admits any
/// run of characters of at least `min_size` length.
pub(crate) fn emit_min_size_placeholder(buf: &mut String, min_size: usize, precedence: u8) {
    if min_size == 0 {
        return; // zero width: safe to drop without emitting anything
    }
    if precedence > 2 {
        buf.push_str("(?:");
    }
    buf.push_str("(?s:.)");
    match min_size {
        1 => buf.push('+'),
        n => {
            buf.push('{');
            crate::push_usize(buf, n);
            buf.push_str(",}");
        }
    }
    if precedence > 2 {
        buf.push(')');
    }
}

/// Write the seek-pattern approximation for `info` into `buf`.
///
/// Easy (non-hard) subtrees are serialised verbatim via [`Expr::to_str`].
/// Hard nodes are handled as follows:
/// - `Backref` / `SubroutineCall`: inline the referenced group's body (up to
///   `MAX_SUBROUTINE_RECURSION_DEPTH`). For backrefs, positional anchors (`^`, `$`, `\b`) etc.
///   are dropped from the inlined body because backreferences only capture text, not positions.
///   If inlining is not possible, a permissive `(?s:.){min_size,}` placeholder is emitted
///   instead of silently dropping, to maintain correctness.
/// - `LookAround`, `KeepOut`, `ContinueFromPreviousMatchEnd`, `BacktrackingControlVerb`,
///   `BackrefExistsCondition`, `Absent`: dropped (emit nothing — safe over-approximation,
///   their `min_size` is 0).
/// - `AtomicGroup`: keep the contents, discard the atomicity wrapper.
/// - `Group`: keep the contents, discard the capture group wrapper.
/// - `Conditional`: emit the union of the true and false branches.
/// - `GeneralNewline`: replaced with an explicit alternation.
/// - `Assertion`: anchors and word-boundary assertions are emitted; half-boundaries are
///   dropped (over-approximation — `\b` would be too restrictive).
pub(crate) fn build_seek_pattern<'a>(
    info: &Info<'a>,
    group_info_map: &Map<usize, &'a Info<'a>>,
    depth: usize,
    buf: &mut String,
    precedence: u8,
) {
    build_seek_pattern_impl(info, group_info_map, depth, buf, precedence, false);
}

pub(crate) fn build_seek_pattern_impl<'a>(
    info: &Info<'a>,
    group_info_map: &Map<usize, &'a Info<'a>>,
    depth: usize,
    buf: &mut String,
    precedence: u8,
    drop_positional_anchors: bool,
) {
    // Drop positional anchors at this node when requested (used when inlining for a backref).
    if drop_positional_anchors {
        if let Expr::Assertion(
            Assertion::StartText
            | Assertion::EndText
            | Assertion::StartLine { .. }
            | Assertion::EndLine { .. },
        ) = info.expr
        {
            return;
        }
        // \Z is an odd one out, although it is a zero-width assertion, for the purposes of seeking
        // it still needs to consume the newlines
        if let Expr::Assertion(Assertion::EndTextIgnoreTrailingNewlines { crlf }) = info.expr {
            if *crlf {
                buf.push_str(r"[\r\n]*");
            } else {
                buf.push_str(r"\n*");
            }
            return;
        }
    }

    if !info.hard {
        // Easy subtree — use to_str directly when no positional-anchor dropping is needed,
        // or when the subtree contains no positional anchors (so dropping is a no-op).
        if !drop_positional_anchors || !expr_contains_positional_anchor(info.expr) {
            info.expr.to_str(buf, precedence);
            return;
        }
        // The easy subtree contains positional anchors that need to be stripped.
        // Fall through to the match below, which recurses via build_seek_pattern_impl.
    }

    match info.expr {
        Expr::Empty | Expr::DefineGroup { .. } => {}
        Expr::Assertion(assertion) => {
            // Emit all assertion types, approximating half-word-boundaries with \b.
            // Note: easy positional assertions (StartText, EndText, StartLine, EndLine) are
            // handled by the to_str early return above (drop_positional_anchors=false) or by
            // the early drop at the top of this function (drop_positional_anchors=true).
            // Only hard assertions reach this arm.
            match assertion {
                Assertion::EndTextIgnoreTrailingNewlines { crlf: false } => buf.push_str(r"\n*$"),
                Assertion::EndTextIgnoreTrailingNewlines { crlf: true } => {
                    buf.push_str(r"[\r\n]*$")
                }
                Assertion::WordBoundary => buf.push_str(r"\b"),
                Assertion::NotWordBoundary => buf.push_str(r"\B"),
                // Full word boundaries: \< and \> — overapproximate with \b.
                Assertion::LeftWordBoundary | Assertion::RightWordBoundary => buf.push_str(r"\b"),
                // Half-boundaries (\b{start-half}, \b{end-half}) can match in positions where
                // \b does not (e.g. before punctuation), so \b would be an under-approximation.
                // Drop them (emit nothing) which is a safe over-approximation.
                Assertion::LeftWordHalfBoundary | Assertion::RightWordHalfBoundary => {}
                // Easy positional anchors — handled by early return / early drop above.
                // They can only reach here when drop_positional_anchors=false and the subtree
                // is easy (handled by to_str) but the Group/Concat parent fell through to match.
                // Emit them faithfully.
                Assertion::StartText => buf.push('^'),
                Assertion::EndText => buf.push('$'),
                Assertion::StartLine { crlf: false } => buf.push_str("(?m:^)"),
                Assertion::StartLine { crlf: true } => buf.push_str("(?Rm:^)"),
                Assertion::EndLine { crlf: false } => buf.push_str("(?m:$)"),
                Assertion::EndLine { crlf: true } => buf.push_str("(?Rm:$)"),
            }
        }
        Expr::Concat(_) => {
            if precedence > 1 {
                buf.push_str("(?:");
            }
            for child in &info.children {
                build_seek_pattern_impl(
                    child,
                    group_info_map,
                    depth,
                    buf,
                    2,
                    drop_positional_anchors,
                );
            }
            if precedence > 1 {
                buf.push(')');
            }
        }
        Expr::Alt(_) => {
            if precedence > 0 {
                buf.push_str("(?:");
            }
            let mut first = true;
            for child in &info.children {
                if !first {
                    buf.push('|');
                }
                build_seek_pattern_impl(
                    child,
                    group_info_map,
                    depth,
                    buf,
                    1,
                    drop_positional_anchors,
                );
                first = false;
            }
            if precedence > 0 {
                buf.push(')');
            }
        }
        Expr::Group(_) => {
            // Drop the capture group wrapper; keep the contents.
            if !info.children.is_empty() {
                build_seek_pattern_impl(
                    &info.children[0],
                    group_info_map,
                    depth,
                    buf,
                    precedence,
                    drop_positional_anchors,
                );
            }
        }
        Expr::Repeat { lo, hi, greedy, .. } => {
            if precedence > 2 {
                buf.push_str("(?:");
            }
            if !info.children.is_empty() {
                build_seek_pattern_impl(
                    &info.children[0],
                    group_info_map,
                    depth,
                    buf,
                    3,
                    drop_positional_anchors,
                );
            }
            write_quantifier(buf, *lo, *hi, *greedy);
            if precedence > 2 {
                buf.push(')');
            }
        }
        Expr::Backref { group, casei }
        | Expr::BackrefWithRelativeRecursionLevel { group, casei, .. } => {
            // Inline the body of the referenced capture group, wrapping with (?i:...) when
            // the backref is case-insensitive so the approximation remains correct.
            //
            // Positional anchors are dropped from the inlined body: a backref matches the
            // captured text, not a position, so anchors from the group definition do not hold
            // at the backref's position.
            //
            // For BackrefWithRelativeRecursionLevel, the relative_level is ignored here:
            // for seeking purposes, the group body is the same regardless of recursion level.
            //
            // If inlining is not possible (depth limit, group not in map), emit a permissive
            // placeholder so that no match positions are incorrectly skipped.
            if depth < MAX_SUBROUTINE_RECURSION_DEPTH {
                if let Some(group_info) = group_info_map.get(group) {
                    if !group_info.children.is_empty() {
                        let child = &group_info.children[0];
                        if *casei {
                            let mut inner = String::new();
                            build_seek_pattern_impl(
                                child,
                                group_info_map,
                                depth + 1,
                                &mut inner,
                                // Precedence 0 (alternation) so content inside (?i:...) is unambiguous.
                                0,
                                true,
                            );
                            if !inner.is_empty() {
                                buf.push_str("(?i:");
                                buf.push_str(&inner);
                                buf.push(')');
                            }
                        } else {
                            build_seek_pattern_impl(
                                child,
                                group_info_map,
                                depth + 1,
                                buf,
                                precedence,
                                true,
                            );
                        }
                        return;
                    }
                    // Empty group (no children): min_size=0, nothing to emit.
                    return;
                }
            }
            // Could not inline (depth limit or group not found): emit a permissive placeholder
            // so we don't falsely skip positions where the backref's target could have matched.
            emit_min_size_placeholder(buf, info.min_size, precedence);
        }
        Expr::SubroutineCall(target_group) => {
            // Inline the body of the target group, honouring the recursion depth limit.
            if depth < MAX_SUBROUTINE_RECURSION_DEPTH {
                if let Some(group_info) = group_info_map.get(target_group) {
                    if !group_info.children.is_empty() {
                        build_seek_pattern_impl(
                            &group_info.children[0],
                            group_info_map,
                            depth + 1,
                            buf,
                            precedence,
                            drop_positional_anchors,
                        );
                        return;
                    }
                    return;
                }
            }
            emit_min_size_placeholder(buf, info.min_size, precedence);
        }
        // LookAround is zero-width — drop it.
        Expr::LookAround(_, _) => {}
        Expr::AtomicGroup(_) => {
            // Keep the contents; atomicity is invisible to the seek approximation.
            if !info.children.is_empty() {
                build_seek_pattern_impl(
                    &info.children[0],
                    group_info_map,
                    depth,
                    buf,
                    precedence,
                    drop_positional_anchors,
                );
            }
        }
        Expr::GeneralNewline { unicode } => {
            // Replace \R with an explicit alternation accepted by regex-automata.
            if *unicode {
                buf.push_str(r"(?:\r\n|[\n\x0B\x0C\r\x85\u{2028}\u{2029}])");
            } else {
                buf.push_str(r"(?:\r\n|[\n\x0B\x0C\r])");
            }
        }
        Expr::Conditional { .. } => {
            // Emit the union of (condition + true_branch) and (false_branch).
            //
            // Including the condition prefix in the true alternative ensures that consuming
            // conditions (e.g. `(?(a)b|c)` where `a` is matched and consumed) are
            // over-approximated correctly.  For zero-width conditions (e.g. backref-exists
            // checks) the condition emits nothing and the result degenerates to
            // `true_branch | false_branch`, preserving the original behaviour.
            let mut cond_pat = String::new();
            let mut true_pat = String::new();
            let mut false_pat = String::new();
            if !info.children.is_empty() {
                build_seek_pattern_impl(
                    &info.children[0],
                    group_info_map,
                    depth,
                    &mut cond_pat,
                    2,
                    drop_positional_anchors,
                );
            }
            if info.children.len() >= 2 {
                build_seek_pattern_impl(
                    &info.children[1],
                    group_info_map,
                    depth,
                    &mut true_pat,
                    2,
                    drop_positional_anchors,
                );
            }
            if info.children.len() >= 3 {
                build_seek_pattern_impl(
                    &info.children[2],
                    group_info_map,
                    depth,
                    &mut false_pat,
                    1,
                    drop_positional_anchors,
                );
            }
            // Build the "condition then true-branch" alternative.
            let cond_true = if cond_pat.is_empty() {
                true_pat.clone()
            } else if true_pat.is_empty() {
                cond_pat.clone()
            } else {
                format!("(?:{}{})", cond_pat, true_pat)
            };
            match (cond_true.is_empty(), false_pat.is_empty()) {
                (true, true) => {}
                (true, false) => buf.push_str(&false_pat),
                (false, true) => buf.push_str(&cond_true),
                (false, false) => {
                    if precedence > 0 {
                        buf.push_str("(?:");
                    }
                    buf.push_str(&cond_true);
                    buf.push('|');
                    buf.push_str(&false_pat);
                    if precedence > 0 {
                        buf.push(')');
                    }
                }
            }
        }
        // Absent repeater `(?~expr)` matches any content not containing `expr`.
        // Approximate with `(?s:.*)` (any characters, including newline) since the repeater
        // can consume an arbitrary number of characters.
        Expr::Absent(Absent::Repeater(_)) => buf.push_str("(?s:.*)"),
        // Absent expression `(?~|absent|exp)` matches `exp` subject to the absent constraint.
        // Approximate by seeking only for `exp`, dropping the constraint — this is a safe
        // over-approximation since any position where `exp` matches (ignoring the constraint)
        // is a superset of positions where the full expression matches.
        Expr::Absent(crate::Absent::Expression { .. }) => {
            if info.children.len() >= 2 {
                build_seek_pattern_impl(
                    &info.children[1],
                    group_info_map,
                    depth,
                    buf,
                    precedence,
                    drop_positional_anchors,
                );
            }
        }
        // Zero-width / control nodes — drop them (all have min_size = 0).
        Expr::KeepOut
        | Expr::ContinueFromPreviousMatchEnd
        | Expr::BacktrackingControlVerb(_)
        | Expr::BackrefExistsCondition { .. }
        | Expr::Absent(_) => {}
        // Easy leaf nodes (Literal, Any, Delegate) are always handled by the `!info.hard`
        // early return above and never reach here.  Listed explicitly so that adding a new
        // Expr variant produces a compile error until the seek-pattern case is handled.
        Expr::Literal { .. } | Expr::Any { .. } | Expr::Delegate { .. } => {
            info.expr.to_str(buf, precedence)
        }
        // These variants cause a compile error during analysis and are therefore unreachable
        // after a successful `analyze()` call.
        Expr::AstNode(..) => {
            unreachable!("unexpected expr variant after analysis")
        }
    }
}

/// Returns `true` if `pattern` contains at least one character that provides useful filtering
/// (i.e. a literal character, character class, or anchor rather than just wildcards/quantifiers).
///
/// This is a conservative approximation: any pattern containing `[`, `\`, `^`, `$`, or an
/// alphanumeric character is considered useful. The primary goal is to reject fully
/// unconstrained patterns such as `.*`.
pub fn seek_pattern_is_useful(pattern: &str) -> bool {
    pattern
        .bytes()
        .any(|b| matches!(b, b'[' | b'\\' | b'^' | b'$' | b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analyze::{analyze, AnalyzeContext};
    use crate::compile::populate_group_info_map;

    /// Build the seek pattern for a regex string and return it.
    fn get_seek_pattern(re: &str) -> String {
        let tree = Expr::parse_tree(re).unwrap();
        let info = analyze(&tree, AnalyzeContext::default()).unwrap();
        let mut group_info_map = Map::new();
        populate_group_info_map(&mut group_info_map, &info);
        let mut buf = String::new();
        build_seek_pattern(&info, &group_info_map, 0, &mut buf, 0);
        buf
    }

    #[test]
    fn seek_pattern_backref_no_anchor() {
        // Simple backref with no positional anchors: group body is inlined verbatim.
        // The `(?:...)` wrapping comes from Concat precedence handling when stripping groups.
        assert_eq!(get_seek_pattern(r"(abc)\1"), "(?:abc)(?:abc)");
    }

    #[test]
    fn seek_pattern_backref_with_start_anchor() {
        // Backref to a group whose body starts with `^`: the anchor should be dropped from the
        // inlined backref position but kept at the group definition site.
        // `(^a)\1` — seek = `(?:^a)` (group) + `(?:a)` (backref, anchor dropped)
        assert_eq!(get_seek_pattern(r"(^a)\1"), "(?:^a)(?:a)");
    }

    #[test]
    fn seek_pattern_backref_with_start_anchor_variable_length() {
        // The group has a start anchor plus a variable-length content.
        // Anchor is dropped when inlining for the backref.
        assert_eq!(get_seek_pattern(r"(^a+)\1"), "(?:^a+)(?:a+)");
    }

    #[test]
    fn seek_pattern_backref_with_end_anchor() {
        // Backref to a group ending with `$`: the anchor is dropped in the inline position.
        assert_eq!(get_seek_pattern(r"(a$)\1"), "(?:a$)(?:a)");
    }

    #[test]
    fn seek_pattern_backref_only_anchor() {
        // Backref to a group that is only a positional anchor.
        // The group emits `^` (min_size=0); the backref drops `^` and emits nothing.
        assert_eq!(get_seek_pattern(r"(^)\1"), "^");
    }

    #[test]
    fn seek_pattern_lookahead_dropped() {
        // Lookahead is zero-width and is dropped; only the literal is kept.
        assert_eq!(get_seek_pattern(r"(?=foo)bar"), "bar");
    }

    #[test]
    fn seek_pattern_casei_literal_backref_with_anchor() {
        // Group contains a case-insensitive literal and a start anchor.
        // The anchor is dropped during backref inlining; the casei literal emits (?i:...).
        assert_eq!(get_seek_pattern(r"(?i:(^a))\1"), "(?:^(?i:a))(?:(?i:a))");
    }

    #[test]
    fn seek_pattern_easy_expr_preserved() {
        // An easy (non-hard) expression is serialised verbatim via to_str.
        assert_eq!(get_seek_pattern(r"abc"), "abc");
        assert_eq!(get_seek_pattern(r"a|b"), "a|b");
        assert_eq!(get_seek_pattern(r"a+b*c?"), "a+b*c?");
    }

    #[test]
    fn seek_pattern_end_text_ignore_trailing_newlines_non_crlf() {
        // \Z (non-CRLF mode) — approximate with `\n*$` so the seek only visits positions
        // near end-of-text.
        assert_eq!(get_seek_pattern(r"abc\Z"), r"abc\n*$");
    }

    #[test]
    fn seek_pattern_end_text_ignore_trailing_newlines_crlf() {
        // \Z in CRLF mode — approximate with `(?:\r?\n)*$`.
        assert_eq!(get_seek_pattern(r"(?R)abc\Z"), r"abc[\r\n]*$");
    }

    #[test]
    fn seek_pattern_end_text_ignore_trailing_newlines_only() {
        // \Z alone (hard assertion) — just the seek pattern with no surrounding literal.
        assert_eq!(get_seek_pattern(r"\Z"), r"\n*$");
    }

    #[test]
    fn seek_pattern_backref_with_end_text_ignore_trailing_newlines() {
        // Backref to a group that ends with \Z: \Z is a positional anchor and should be
        // dropped when inlining the group body for the backref.
        // `(a\Z)\1` — seek = `(?:a\n*$)` (group) + `(?:a\n*)` (backref, \Z anchor dropped, newline matching kept)
        assert_eq!(get_seek_pattern(r"(a\Z)\1"), r"(?:a\n*$)(?:a\n*)");
    }

    // ------------------------------------------------------------------
    // Diagnostic tests for PR #249 (seek false-negatives on top-level
    // (?=X) lookaheads). Each test:
    //   1. Builds the seek over-approximation via build_seek_pattern.
    //   2. Runs that approximation through META, PikeVM, and lazy DFA on
    //      the failing haystack from the syntest.
    //   3. Reports the leftmost positions side-by-side and asserts the
    //      approximation, when correctly searched, includes the position
    //      where the original pattern matches.
    //
    // Verdict guide:
    //   - All three engines agree on a position past expected_leftmost
    //     => over-approximation is too narrow (the bug is in seek.rs).
    //   - META differs from PikeVM/DFA => regex-automata META leftmost bug.
    //   - All three agree on expected_leftmost => bug is elsewhere
    //     (vm.rs Insn::Seek handling, or syntect-side).

    fn diagnostic_for(case: &str, pattern: &str, haystack: &str, expected_leftmost: usize) {
        let mut tree = Expr::parse_tree(pattern).expect("parse_tree");
        // Apply optimize_trailing_lookahead — this is the transform that turns
        // a top-level `(?=X)` into `Concat([Group(Empty), X])`, which is the
        // exact shape PR #249 disables seek for.
        let requires_capture_group_fixup = crate::optimize::optimize(&mut tree);
        let info = analyze(
            &tree,
            AnalyzeContext {
                explicit_capture_group_0: requires_capture_group_fixup,
                ..AnalyzeContext::default()
            },
        )
        .expect("analyze");
        let mut group_info_map = Map::new();
        populate_group_info_map(&mut group_info_map, &info);
        let mut buf = String::new();
        build_seek_pattern(&info, &group_info_map, 0, &mut buf, 0);
        let approximation = &buf;

        let meta_pos = regex_automata::meta::Regex::new(approximation)
            .ok()
            .and_then(|r| r.find(haystack.as_bytes()))
            .map(|m| m.start());

        let pikevm_pos = {
            use regex_automata::nfa::thompson::pikevm::PikeVM;
            PikeVM::new(approximation).ok().and_then(|pv| {
                let mut cache = pv.create_cache();
                pv.find_iter(&mut cache, regex_automata::Input::new(haystack.as_bytes()))
                    .next()
                    .map(|m| m.start())
            })
        };

        let dfa_pos = {
            use regex_automata::hybrid::regex::Regex as HybridRegex;
            HybridRegex::new(approximation).ok().and_then(|r| {
                let mut cache = r.create_cache();
                r.find(&mut cache, regex_automata::Input::new(haystack.as_bytes()))
                    .map(|m| m.start())
            })
        };

        eprintln!("\n========== {} ==========", case);
        eprintln!("pattern (original):");
        eprintln!("  {}", pattern);
        eprintln!("approximation (seek string):");
        eprintln!("  {}", approximation);
        eprintln!("haystack ({} bytes):", haystack.len());
        eprintln!("  {:?}", haystack);
        eprintln!("expected leftmost: {}", expected_leftmost);
        eprintln!("META:   {:?}", meta_pos);
        eprintln!("PikeVM: {:?}", pikevm_pos);
        eprintln!("DFA:    {:?}", dfa_pos);
    }

    // Pattern from C++.sublime-syntax line 846 ('generic-type' context, the
    // SECOND lookahead — without the trailing \s*(\(|\{) that line 787 has).
    // Line 846 is the one that triggers the seek divergence in syntest.
    //
    // Captured verbatim from syntect's regex.rs::search instrumentation while
    // running `cargo run --example syntest -- testdata/Packages/C++` with
    // .seek(true): syntect's variable resolver expands [[:alpha:]] → \p{L}.
    //
    // Failing assertion: testdata/Packages/C++/syntax_test_cpp.cpp:843, col 29
    // (the first `<` in `<A<B<C>>()`) is mis-scoped as
    // keyword.operator.comparison.c instead of
    // punctuation.definition.generic.begin.c++.
    const CPP_GENERIC_TYPE_LOOKAHEAD: &str = concat!(
        r"(?=(?!template)",
        r"(?:::\s*)?(?:\b[\p{L}_][\p{L}\p{N}_]*\b\s*::\s*)*(?:template\s+)?\b[\p{L}_][\p{L}\p{N}_]*\b",
        r"\s*",
        r"<(?:[^(){}&;*^%=<>-]+(?:<(?:[^(){}&;*^%=<>-]+(?:<[^(){}&;*^%=<>-]*>)?)?\s*>)?)?",
        r"[^(){}&;*^%=<>-]*(?:\([^(){}&;*^%=<>-]*\))?[^(){}&;*^%=<>-]*>",
        r")",
    );

    // Line 843 of syntax_test_cpp.cpp + trailing newline (syntect feeds the
    // line including its '\n' to the regex).
    const CPP_HAYSTACK: &str =
        "    auto f = [](std::function<A<B<C>>()> g) { return g(); };\n";
    // Position of 's' in `std::function` — where path_lookahead first matches.
    // Confirmed by syntect's seek=false call: returns Some(16).
    const CPP_EXPECTED: usize = 16;

    #[test]
    fn diagnostic_seek_cpp_template_angle_bracket() {
        diagnostic_for(
            "C++ generic-type lookahead",
            CPP_GENERIC_TYPE_LOOKAHEAD,
            CPP_HAYSTACK,
            CPP_EXPECTED,
        );
    }

    // Objective-C++.sublime-syntax line 46 defines `generic_lookahead` with the
    // same body as C++. The failing line in syntax_test_objc++.mm:830 is C++
    // code embedded in an .mm file; the same pattern + same haystack span.
    #[test]
    fn diagnostic_seek_objcpp_template_angle_bracket() {
        diagnostic_for(
            "ObjC++ generic-type lookahead (== C++)",
            CPP_GENERIC_TYPE_LOOKAHEAD,
            CPP_HAYSTACK,
            CPP_EXPECTED,
        );
    }

    // Pattern captured verbatim from syntect's regex.rs::search instrumentation
    // while running `cargo run --example syntest -- testdata/Packages/Makefile`
    // with .seek(true). The original syntect form is `(?={{rule_lookahead}})`
    // = `(?={{just_eat}}{{ruleassign}})` from Makefile.sublime-syntax line 122.
    //
    // The (?x) is INSIDE the (?=...) group (between `(?=` and the body) and
    // the ASCII-art comments are preserved — both matter, because the way
    // (?x) interacts with the comment text affects how the over-approximation
    // is built. Use a raw byte-equivalent form rather than reconstructing.
    //
    // Failing assertions: testdata/Packages/Makefile/syntax_test_makefile.mak
    // lines 732 ($(a:b=c) : d) and 740 ($(X:a=b) : w ;) — expect-rule context
    // not pushed.
    const MAKEFILE_RULE_LOOKAHEAD: &str = "(?=(?x)              # ignore whitespace in this regex\n  [^()=]*        #       level 0\n  (?:\\(        # start level 1                      __\n    [^()]*       #       level 1          _______    /*_>-<\n    (?:\\(      # start level 2      ___/ _____ \\__/ /\n      [^()]*     #       level 2     <____/     \\____/\n      (?:\\(    # start level 3     is like snek... (by Valerie Haecky)\n        [^()]*   #       level 3\n        (?:\\(  # start level 4\n          [^()]* #       level 4\n        \\))? #   end level 4\n        [^()]*   #       level 3\n      \\))?   #   end level 3\n      [^()]*     #       level 2\n      (?:\\(    # start level 3\n        [^()]*   #       level 3\n        (?:\\(  # start level 4\n          [^()]* #       level 4\n        \\))? #   end level 4\n        [^()]*   #       level 3\n      \\))?   #   end level 3\n      [^()]*     #       level 2\n    \\))?     #   end level 2\n    [^()]*       #       level 1\n    (?:\\(      # start level 2\n      [^()]*     #       level 2\n      (?:\\(    # start level 3\n        [^()]*   #       level 3\n        (?:\\(  # start level 4\n          [^()]* #       level 4\n        \\))? #   end level 4\n        [^()]*   #       level 3\n      \\))?   #   end level 3\n      [^()]*     #       level 2\n      (?:\\(    # start level 3\n        [^()]*   #       level 3\n        (?:\\(  # start level 4\n          [^()]* #       level 4\n        \\))? #   end level 4\n        [^()]*   #       level 3\n      \\))?   #   end level 3\n      [^()]*     #       level 2\n      (?:\\(    # start level 3\n        [^()]*   #       level 3\n        (?:\\(  # start level 4\n          [^()]* #       level 4\n        \\))? #   end level 4\n        [^()]*   #       level 3\n      \\))?   #   end level 3\n      [^()]*     #       level 2\n    \\))?     #   end level 2\n    [^()]*       #       level 1\n  \\))?       #   end level 1\n  [^()=]*        #       level 0\n:(?!=))";

    // Includes trailing newline that syntect feeds along with the line.
    const MAKEFILE_HAYSTACK: &str = "$(a:b=c) : d\n";
    // Confirmed by syntect's seek=false call: returns Some(0).
    const MAKEFILE_EXPECTED: usize = 0;

    #[test]
    fn diagnostic_seek_makefile_rule_lookahead() {
        diagnostic_for(
            "Makefile rule_lookahead",
            MAKEFILE_RULE_LOOKAHEAD,
            MAKEFILE_HAYSTACK,
            MAKEFILE_EXPECTED,
        );
    }

    // Sanity check: reproduce the failure at the regex level on this checkout
    // (fancy-regex main, with the buggy seek). For each case, .seek(false)
    // should find expected_leftmost; .seek(true) should miss it (or find a
    // wrong position) — confirming the bug is in seek, not elsewhere.
    fn sanity_for(case: &str, pattern: &str, haystack: &str, expected_leftmost: usize) {
        use crate::RegexBuilder;
        let off = RegexBuilder::new(pattern)
            .oniguruma_mode(true)
            .build()
            .expect("build seek=false")
            .find_from_pos(haystack, 0)
            .expect("find seek=false")
            .map(|m| m.start());
        let on = RegexBuilder::new(pattern)
            .oniguruma_mode(true)
            .seek(true)
            .build()
            .expect("build seek=true")
            .find_from_pos(haystack, 0)
            .expect("find seek=true")
            .map(|m| m.start());
        eprintln!("--- sanity {} ---", case);
        eprintln!("expected leftmost: {}", expected_leftmost);
        eprintln!("seek=false: {:?}", off);
        eprintln!("seek=true:  {:?}", on);
    }

    #[test]
    fn diagnostic_sanity_cpp() {
        sanity_for("C++", CPP_GENERIC_TYPE_LOOKAHEAD, CPP_HAYSTACK, CPP_EXPECTED);
    }

    #[test]
    fn diagnostic_sanity_makefile() {
        sanity_for(
            "Makefile",
            MAKEFILE_RULE_LOOKAHEAD,
            MAKEFILE_HAYSTACK,
            MAKEFILE_EXPECTED,
        );
    }

    // Probe the original pattern at every position from 0 to 32 (or len).
    // For each position p, report whether the original pattern matches when
    // anchored at p (i.e., does the lookahead succeed if the parser is at p?).
    fn probe_positions(case: &str, pattern: &str, haystack: &str) {
        use crate::RegexBuilder;
        let re_off = RegexBuilder::new(pattern)
            .oniguruma_mode(true)
            .build()
            .expect("build");
        let re_on = RegexBuilder::new(pattern)
            .oniguruma_mode(true)
            .seek(true)
            .build()
            .expect("build seek");
        eprintln!("\n--- positions {} ---", case);
        let limit = haystack.len().min(45);
        for p in 0..=limit {
            let off = re_off
                .find_from_pos(haystack, p)
                .ok()
                .and_then(|m| m)
                .map(|m| m.start());
            let on = re_on
                .find_from_pos(haystack, p)
                .ok()
                .and_then(|m| m)
                .map(|m| m.start());
            if off != on {
                eprintln!(
                    "  pos {:3}: seek=false → {:?}  seek=true → {:?}  *** DIFFER ***",
                    p, off, on
                );
            } else {
                eprintln!("  pos {:3}: {:?}", p, off);
            }
        }
    }

    #[test]
    fn diagnostic_probe_cpp() {
        probe_positions("C++", CPP_GENERIC_TYPE_LOOKAHEAD, CPP_HAYSTACK);
    }

    #[test]
    fn diagnostic_probe_makefile() {
        probe_positions("Makefile", MAKEFILE_RULE_LOOKAHEAD, MAKEFILE_HAYSTACK);
    }

    // syntect calls captures_from_pos(&text[..end], begin) — the haystack is
    // clipped at `end`. Try the C++ case at clipping boundaries that bracket
    // the actual `(` at pos 37 (the `(` the lookahead body needs to see).
    // Probe full captures, not just start position. Bug could manifest as
    // different capture group bounds even when find_from_pos starts agree.
    #[test]
    fn diagnostic_probe_cpp_captures() {
        use crate::RegexBuilder;
        let re_off = RegexBuilder::new(CPP_GENERIC_TYPE_LOOKAHEAD)
            .oniguruma_mode(true)
            .build()
            .expect("build");
        let re_on = RegexBuilder::new(CPP_GENERIC_TYPE_LOOKAHEAD)
            .oniguruma_mode(true)
            .seek(true)
            .build()
            .expect("build seek");
        eprintln!("\n--- C++ captures ---");
        for begin in [0usize, 16, 29, 30] {
            let off = re_off.captures_from_pos(CPP_HAYSTACK, begin).expect("c off");
            let on = re_on.captures_from_pos(CPP_HAYSTACK, begin).expect("c on");
            let off_caps: alloc::vec::Vec<_> = off
                .as_ref()
                .map(|c| (0..c.len()).map(|i| c.get(i).map(|m| (m.start(), m.end()))).collect())
                .unwrap_or_default();
            let on_caps: alloc::vec::Vec<_> = on
                .as_ref()
                .map(|c| (0..c.len()).map(|i| c.get(i).map(|m| (m.start(), m.end()))).collect())
                .unwrap_or_default();
            let differ = if off_caps != on_caps { " *** DIFFER ***" } else { "" };
            eprintln!("  begin={:3}: seek=false → {:?}", begin, off_caps);
            eprintln!("  begin={:3}: seek=true  → {:?}{}", begin, on_caps, differ);
        }
    }

    #[test]
    fn diagnostic_probe_cpp_clipped() {
        use crate::RegexBuilder;
        let re_off = RegexBuilder::new(CPP_GENERIC_TYPE_LOOKAHEAD)
            .oniguruma_mode(true)
            .build()
            .expect("build");
        let re_on = RegexBuilder::new(CPP_GENERIC_TYPE_LOOKAHEAD)
            .oniguruma_mode(true)
            .seek(true)
            .build()
            .expect("build seek");
        eprintln!("\n--- C++ clipped haystack ---");
        for end in [30usize, 37, 38, 39, 40, 44, 45, 60] {
            for begin in [0usize, 16, 29] {
                if begin > end {
                    continue;
                }
                let clipped = &CPP_HAYSTACK[..end];
                let off = re_off
                    .find_from_pos(clipped, begin)
                    .ok()
                    .and_then(|m| m)
                    .map(|m| m.start());
                let on = re_on
                    .find_from_pos(clipped, begin)
                    .ok()
                    .and_then(|m| m)
                    .map(|m| m.start());
                let differ = if off != on { " *** DIFFER ***" } else { "" };
                eprintln!(
                    "  end={:3} begin={:3}: seek=false → {:?}  seek=true → {:?}{}",
                    end, begin, off, on, differ
                );
            }
        }
    }
}
