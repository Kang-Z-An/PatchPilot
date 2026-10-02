# Third-party notices and benchmark provenance

The root MIT license covers PatchPilot's original implementation, documentation,
and toy fixtures. It does not relicense third-party issue text, code fragments,
patches, datasets, dependencies, or container images.

## SWE-bench

Benchmark records originate from the official
[SWE-bench](https://huggingface.co/datasets/princeton-nlp/SWE-bench) and
[SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
datasets. Credit the dataset authors and the authors of the original
repository issues and pull requests when reusing this material. Dataset schema
and harness source are maintained in
[SWE-bench/SWE-bench](https://github.com/SWE-bench/SWE-bench).

- `benchmarks/swebench/*.json`: historical official instance records, including
  upstream gold and test patches where present. The Agent adapter reads only
  instance ID, repository, base commit, and issue text. These oracle fields must
  not be fed into Agent prompts or tools.
- `benchmarks/training_pool_v1/instances/*.json`: selected records with
  `patch` and `test_patch` removed. Their `source` metadata records the dataset,
  revision, and original split.
- Selection manifests and experimental summaries are PatchPilot's annotations.
  They do not alter upstream ground truth or grant rights to upstream code.
- Full checkouts, Docker images, raw conversations, and model weights are not
  distributed in this repository or the Python wheel.

The training pool uses SWE-bench revision
`e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6` and excludes Verified IDs from
revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`.
The older pilot selection did not pin a dataset revision; its versioned JSON
records and base commits define the retained inputs.

## Upstream code represented in the historical pilot

License texts below were read from the pinned upstream Git commits present in
the recorded pilot checkouts. They retain their original notices. Review the
upstream file headers and applicable terms when redistributing or adapting an
individual patch; the PatchPilot MIT license is not a replacement.

| Upstream repository | License source at recorded commit | Retained notice |
| --- | --- | --- |
| [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | [a3e2897bfaf9](https://github.com/matplotlib/matplotlib/blob/a3e2897bfaf9eaac1d6649da535c4e721c89fa69/LICENSE/LICENSE) | [License text](third_party/licenses/matplotlib__matplotlib.txt) |
| [psf/requests](https://github.com/psf/requests) | [39d0fdd9096f](https://github.com/psf/requests/blob/39d0fdd9096f7dceccbc8f82e1eda7dd64717a8e/LICENSE) | [License text](third_party/licenses/psf__requests.txt) |
| [pydata/xarray](https://github.com/pydata/xarray) | [a41edc7bf530](https://github.com/pydata/xarray/blob/a41edc7bf5302f2ea327943c0c48c532b12009bc/LICENSE) | [License text](third_party/licenses/pydata__xarray.txt) |
| [pylint-dev/pylint](https://github.com/pylint-dev/pylint) | [40cc2ffd7887](https://github.com/pylint-dev/pylint/blob/40cc2ffd7887959157aaf469e09585ec2be7f528/LICENSE) | [License text](third_party/licenses/pylint-dev__pylint.txt) |
| [pytest-dev/pytest](https://github.com/pytest-dev/pytest) | [58e6a09db49f](https://github.com/pytest-dev/pytest/blob/58e6a09db49f34886ff13f3b7520dd0bcd7063cd/LICENSE) | [License text](third_party/licenses/pytest-dev__pytest.txt) |
| [scikit-learn/scikit-learn](https://github.com/scikit-learn/scikit-learn) | [3d997697fdd1](https://github.com/scikit-learn/scikit-learn/blob/3d997697fdd166eff428ea9fd35734b6a8ba113e/COPYING) | [License text](third_party/licenses/scikit-learn__scikit-learn.txt) |
| [sphinx-doc/sphinx](https://github.com/sphinx-doc/sphinx) | [f050a7775dfc](https://github.com/sphinx-doc/sphinx/blob/f050a7775dfc9000f55d023d36d925a8d02ccfa8/LICENSE) | [License text](third_party/licenses/sphinx-doc__sphinx.txt) |
| [sympy/sympy](https://github.com/sympy/sympy) | [c50643a49811](https://github.com/sympy/sympy/blob/c50643a49811e9fe2f4851adff4313ad46f7325e/LICENSE) | [License text](third_party/licenses/sympy__sympy.txt) |

The training pool also includes issues from
[marshmallow](https://github.com/marshmallow-code/marshmallow),
[pydicom](https://github.com/pydicom/pydicom),
[astroid](https://github.com/pylint-dev/astroid), and
[SQLFluff](https://github.com/sqlfluff/sqlfluff).
Their corresponding instance records pin the repository and base commit.
Environment recipes download or reuse upstream code separately and do not
change its licensing.

## Dependencies and models

The core installs `python-dotenv`. Optional workflows install the SWE-bench
harness, Hugging Face datasets, or Transformers and use their own licenses.
Tokenization reports refer to Qwen2.5-Coder-7B-Instruct; no model weights are
included. Users must follow their chosen provider's and model's terms.

This project is independently maintained and is not affiliated with or endorsed
by SWE-bench, the upstream repositories, or model providers.
