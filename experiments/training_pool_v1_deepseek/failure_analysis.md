# Official failure analysis

This analysis covers the nine predictions that the official SWE-bench harness
marked unresolved. The labels compare the generated patch, issue statement,
official target-test outcome, and training-split gold patch. Gold patches are
used here only for diagnosis; they were not available to the Agent during
generation.

## Aggregate diagnosis

| Category | Count | Meaning |
| --- | ---: | --- |
| Wrong abstraction layer | 4 | The patch treated a nearby symptom instead of the state owner or shared layer responsible for the behavior. |
| Incomplete behavioral coverage | 4 | The patch addressed part of the report but omitted required modes, side effects, or API behavior. |
| Incorrect specification algorithm | 1 | The implementation was plausible but did not follow the algorithm explicitly required by the issue. |

All nine patches applied successfully. They caused zero official regression-test
failures and modified no test files, but failed issue-specific target tests.
Consequently, these are model/verification failures rather than confirmed
environment failures.

## Case findings

### `pydicom__pydicom-800` — incorrect specification algorithm

The Agent recognized that the `2.25` branch should produce a UUID-style numeric
UID, but derived the value from SHA-512. The required behavior for `prefix=None`
was a fresh UUID4 integer representation. The implementation was structurally
reasonable yet semantically different from the requested algorithm, and the
single target test failed.

Training lesson: when a report names a standard construction, translate that
construction into an explicit invariant before choosing an implementation.

### `pydicom__pydicom-816` — incomplete behavioral coverage

The Agent special-cased the literal `"Not Supplied"` and mapped it to the default
encoding. The required behavior was broader: handle any codec lookup failure,
warn and fall back in permissive mode, and preserve strict-validation behavior.
All three target tests failed.

Training lesson: prefer exception- and policy-based behavior over examples
copied literally from a bug report.

### `pydicom__pydicom-901` — incomplete behavioral coverage

The Agent removed the import-time default handler, which addressed the unwanted
side effect, but also changed the debug configuration without preserving the
public API. The intended design keeps a `NullHandler` by default and lets
`debug(default_handler=True)` install a stream handler on demand. All five
target tests failed. The Agent's final public test exercised an unrelated UID
fallback and therefore provided no issue-specific evidence.

Training lesson: verify both sides of an API lifecycle: the undesired default
behavior and the required opt-in behavior.

### `pylint-dev__astroid-1978` — incomplete behavioral coverage

The Agent wrapped attribute access with Python warning suppression. The problem
also covered text written directly to stdout and stderr by third-party objects;
the intended implementation captures those streams and reports the output via
logging. The single target test failed.

Training lesson: enumerate distinct side-effect channels instead of treating
all diagnostics as `warnings` module events.

### `pylint-dev__astroid-2307` — wrong abstraction layer

The Agent globally changed `TypeVar._infer()` to yield its bound. The expected
fix was localized in the typing brain, where Astroid constructs a richer
synthetic TypeVar representation. Editing the generic node inference layer
changed the wrong semantics and the target test failed.

Training lesson: trace framework-specific synthetic inference before changing
a generic AST node method.

### `pylint-dev__astroid-927` — wrong abstraction layer

The Agent restored inference paths around one statement-inference call site.
The underlying defect was mutable state sharing in `InferenceContext.clone()`;
the cloned context needed independent `path` and `inferred` containers. The
symptom-site workaround passed one of three target tests but failed the other
two.

Training lesson: when failures depend on recursion or traversal order, inspect
state ownership and copying before adding local resets.

### `sqlfluff__sqlfluff-1517` — wrong abstraction layer

The Agent changed `Delimited` grammar matching to retain segments when a
delimiter was missing. The issue required malformed parser output to be turned
into a user-facing `SQLParseError` in parser helpers. The modified matching
semantics did not satisfy the error-handling contract, and the target test
failed.

Training lesson: distinguish recovery/reporting requirements from grammar
matching behavior.

### `sqlfluff__sqlfluff-2625` — incomplete behavioral coverage

The Agent raised selected logger thresholds for machine-readable output modes.
The required CLI behavior included a `--write-output` path and correct routing
of serialized output to a file or stdout without contaminating it. Logger
suppression addressed only one output source; all five target tests failed.

Training lesson: derive a complete input/output matrix for CLI issues: format,
destination, logging, exit behavior, and file-writing modes.

### `sqlfluff__sqlfluff-4041` — wrong abstraction layer

The Agent deduplicated violations only while formatting CLI output, using rule,
line, and position as identity. The required behavior belonged in the linter
core so every consumer receives deduplicated violations, with identity that
also accounts for description and fixes. The target core-level test failed.

Training lesson: if multiple consumers observe the same faulty collection,
repair the collection owner rather than one renderer.

## Cross-case implications

The next Agent improvement should focus on verification quality rather than
patch syntax. Useful changes are:

1. Require the Agent to write down the behavioral contract and affected modes
   before editing.
2. Prefer targeted issue reproductions over broad existing regression files.
3. Add a state-ownership and abstraction-layer check before modifying generic
   framework internals.
4. Treat a passing public regression command as necessary but insufficient
   evidence of resolution.

For post-training, retain the seven officially resolved exact trajectories as
strict SFT candidates. Keep the nine original trajectories as rejected examples
and create corrected, officially verified train-only replays before forming
chosen/rejected preference pairs. Do not copy a gold diff into a synthetic
assistant answer and call it a trajectory: the chosen example must preserve a
coherent tool-use and verification sequence.
