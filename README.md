*This project has been created as part of the 42 curriculum by zhamza.*

# Call Me Maybe

A Python function-calling project that uses a small language model to translate natural-language requests into structured, type-checked JSON function calls. Developed for the **42 curriculum** (subject version 1.7).

## Description

Given a list of function definitions and user prompts, Call Me Maybe selects an appropriate function and extracts arguments of the required types. It **does not execute** the chosen function or answer the user question directly. Its output is a JSON array of calls, each containing exactly `prompt`, `name`, and `parameters`.

The default model is **Qwen/Qwen3-0.6B**, accessed through the official subject-provided `llm_sdk` package. Function selection is performed using the model, not by matching a prompt to a hard-coded function name.

### Example

Input prompt:

```text
What is the sum of 2 and 3
```

Expected function call:

```json
{
  "prompt": "What is the sum of 2 and 3",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

This is an **expected example**, not a claim that every run or unseen prompt is predicted correctly.

## Project Structure

```text
call-me-maybe/
├── src/
│   ├── __init__.py
│   ├── __main__.py       # Entry point for python -m src
│   ├── main.py           # CLI and function-calling pipeline (if separated)
│   ├── loader.py         # Input file loading and Pydantic validation
│   ├── schema.py         # Function, parameter, and prompt schemas
│   ├── llm.py            # Adapter for the official SDK
│   ├── candidates.py     # Typed candidate extraction
│   ├── constrained_decoder.py
│   ├── output.py         # Save JSON results
│   └── visualizer.py     # Terminal progress output
├── llm_sdk/              # Official provided SDK, beside src/
│   └── __init__.py
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/            # Created on execution
├── Makefile
├── pyproject.toml
└── README.md
```

`src/__main__.py` must call the actual entry point in `src/main.py` if your project separates those files. The exact arrangement of the entry-point files may vary; `uv run python -m src` is the required interface.

## Instructions

### Requirements

- Python **3.10 or newer** (use a version compatible with the SDK and installed PyTorch build).
- [`uv`](https://docs.astral.sh/uv/).
- The **official** `llm_sdk` provided with the subject.
- Enough disk space for dependencies and the Qwen model weights; the first run may require a model download.

The implementation uses **Pydantic** for data validation, and the SDK internally relies on model-inference dependencies such as PyTorch and Transformers. Application code accesses the model using the SDK's public methods.

### Install

From the repository root:

```bash
uv sync
```

Place the provided SDK next to `src/`, ensuring that this import works:

```bash
uv run python -c "from llm_sdk import Small_LLM_Model; print('SDK OK')"
```

Depending on the downloaded SDK archive, its Python package can be nested. The importable package must be available at `llm_sdk/__init__.py`, not only `llm_sdk/llm_sdk/__init__.py`.

### Run

```bash
uv run python -m src
```

With explicit input and output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

If the repository's `Makefile` provides the mandatory targets, equivalent commands include:

```bash
make install
make run
make debug
make clean
make lint
```

The project also supports terminal progress visualization when enabled in its CLI or `make run` configuration. This shows the selected function, extracted arguments, and schema-validation status; it is **not** an accuracy score.

### macOS / 1337 workstations

For an Intel Mac (`uname -m` prints `x86_64`), install a Python and PyTorch combination that has an **Intel macOS wheel**. A PyTorch version locked for a newer Apple Silicon or Linux platform may fail during `uv sync`; the `pyproject.toml` / lockfile must support the actual machine. Do not assume that a CUDA build can run on macOS.

On school machines with a small home partition, store Hugging Face model downloads on a volume with enough space, for example `goinfre`:

```bash
mkdir -p "$HOME/goinfre/huggingface-cache"
export HF_HOME="$HOME/goinfre/huggingface-cache"
export HF_HUB_CACHE="$HF_HOME/hub"
uv run python -m src
```

Verify the target volume first with `df -h "$HOME/goinfre"`; this path depends on the workstation's mount layout. These exports only apply to the current shell session.

## Input and Output Formats

`data/input/functions_definition.json` is a JSON array of function specifications, including function names, descriptions, parameter names and types, and return types. For example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

`data/input/function_calling_tests.json` is a JSON array of objects with a `prompt` field:

```json
[
  {"prompt": "What is the sum of 2 and 3"},
  {"prompt": "Greet shrek"},
  {"prompt": "Reverse the string 'hello'"}
]
```

The result is written to `data/output/function_calls.json` by default. Every output item must have **exactly** these keys:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": {"name": "shrek"}
}
```

A schema-valid call is not necessarily semantically correct. For example, `{"a": 2.0, "b": 2.0}` has the correct JSON types but is wrong for “sum of 2 and 3.” The correction sheet also requires `number` values to be JSON floating-point numbers (such as `2.0`, not `2`) and any declared allowed-value constraints to be respected.

## Algorithm: Constrained Decoding

The application uses a **finite-choice, token-constrained decoder**:

1. Load and validate the function definitions and incoming prompts.
2. Build a function-selection context containing the available definitions and the user request.
3. Tokenize the permitted function names using the public SDK interface.
4. At each generation step, keep only next-token IDs that continue at least one permitted name. Choose the highest-logit token **among allowed tokens**, rather than from the unrestricted vocabulary.
5. After selecting a function, gather candidate values for each typed parameter from the request (for example, numbers, quoted strings, or words).
6. Build an argument-selection context and repeat the same token-prefix-constrained selection over JSON-serialized candidates.
7. Convert `number` parameters to floating-point values, validate the completed call against its function schema, and save the JSON array.

The decoder maintains a prefix of generated token IDs; no next token is accepted unless it matches a candidate continuation. This is different from simply telling the model to “output valid JSON.” The finite candidate set makes selected scalar values JSON-serializable, while Pydantic/schema checks provide an additional validation layer.

**Limitations:** The current decoder makes greedy token decisions, can be sensitive to tokenizer boundaries and prompt wording, and rejects ambiguous token-prefix cases. Candidate extraction may omit values that are spelled out, inferred, or expressed in complex ways. A valid JSON output therefore does not prove correct argument selection. The model must continue to choose the function; candidate extraction must not substitute hard-coded function routing.

## Design Decisions

- **Public SDK only:** The adapter uses `encode`, `decode` where available, and `get_logits_from_input_ids`; no SDK-private model internals are accessed.
- **Separate responsibilities:** Loading, data validation, candidate collection, decoding, saving, and visualization are independent modules.
- **Pydantic models:** Schemas validate incoming definitions/prompts and the resulting call structure.
- **Finite candidates:** Avoids uncontrolled generation of long or malformed number strings and allows token-level restriction.
- **Graceful failure:** Invalid or missing input files, unsupported argument types, missing SDK imports, and unavailable candidate values produce diagnostic errors instead of intentionally fabricating data.

## Testing Strategy

Check the input parser, candidate extraction, argument validation, and actual model-based output separately.

```bash
# Confirm input/output paths and SDK availability
uv run python -c "from llm_sdk import Small_LLM_Model; print('SDK OK')"

# Run the full pipeline
uv run python -m src

# Verify JSON syntax
uv run python -m json.tool data/output/function_calls.json

# Apply project code-quality checks
make lint
```

Recommended edge cases include empty or whitespace-only prompts, malformed JSON, missing files, unknown types, zero and negative numbers, decimal values, repeated numbers (`2 and 2`), quoted strings, multiword names, Unicode, ambiguous requests, and functions with several arguments. Rejecting empty prompts with a clear error is one reasonable project-level policy; the subject specifically requires testing empty strings but does not prescribe that exact response.

For evaluation, use the **official supplied moulinette** and its private exercise set where available. Do not evaluate semantic accuracy solely from successful JSON parsing or the terminal label `VALID`.

## Performance and Reliability

The subject targets **100% parseable, schema-valid JSON**, **at least 90% correct function-and-argument predictions**, and processing within the required time budget. These are **evaluation targets**, not verified results for this implementation.

- **JSON reliability:** Finite choices plus schema validation prevent many malformed outputs, but all supported types and edge cases still require testing.
- **Prediction accuracy:** Depends on function selection, candidate coverage, parameter mapping, and the small model's logits. No measured private-moulinette accuracy is claimed here.
- **Speed:** A small local model reduces inference cost, but selecting each argument requires additional model scoring. Timing varies by CPU/GPU, cache state, and number of prompts. No benchmark is claimed here.

## Challenges Faced

- **Repeated argument values:** Earlier output selected `2.0` for both `a` and `b` in “sum of 2 and 3.” This revealed that independent constrained selections can still be semantically incorrect.
- **String extraction:** Allowing the entire prompt as a candidate caused `"Greet shrek"` to be selected instead of the person's name. Narrower candidates help common cases but can introduce other limitations with multiword values.
- **Tokenization:** Independently encoding complete candidates may not always represent the most natural continuation of the context, especially around whitespace and special tokens.
- **SDK packaging:** A nested SDK directory or a file called `init.py` instead of `__init__.py` can prevent the Python import from working.
- **Intel macOS setup:** Platform-specific PyTorch wheels and limited home-directory space can prevent dependency installation or model downloads. Choose a compatible build and relocate caches to a spacious filesystem where needed.

These cases should remain covered by regression tests; they are not evidence of a complete fix for unseen prompts.

## Resources

- [42 Call Me Maybe subject](https://projects.intra.42.fr/projects/call-me-maybe) — specification and supplied SDK/moulinette attachments.
- [Python documentation](https://docs.python.org/3/) — typing, JSON, and error handling.
- [Pydantic documentation](https://docs.pydantic.dev/) — data validation.
- [uv documentation](https://docs.astral.sh/uv/) — dependency and environment management.
- [Hugging Face Qwen3-0.6B model page](https://huggingface.co/Qwen/Qwen3-0.6B) — default model information.
- [Qwen documentation](https://qwen.readthedocs.io/) — model family background.

### AI Usage Disclosure

AI assistance was used suggest tests for JSON handling, and incorrect arguments.