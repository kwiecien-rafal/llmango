# LLMango responses

This is the dataset behind LLMango: a blog dedicated to showcasing different behaviours of Large Language Models through prompting them in great volumes and data visualization. LLMango is divided into experiments, where each one is different and tries to answer specific questions you might have about LLMs.

One row is one call. It holds the prompt that produced it, the provider's verbatim response, and the usage, cost and timing the provider reported.

- Charts and write-ups: https://llmango.rafalkwiecien.com
- Pipeline that produced this: https://github.com/kwiecien-rafal/llmango

## Usage

A **config** is an experiment. A **split** is one question inside it.

```python
from datasets import load_dataset

baseline = load_dataset("rafalkwiecien/llmango", "e001_fruit", split="001a")
```

Every question shares its experiment's columns, so splits within a config compare directly. A later experiment adds the columns its own analysis needs.

## Experiments

### 001: fruit

Experiment: Can an LLM's response meaningfully change when you prompt it in a different language, or when you present information to the model in a different manner?

**gpt-5.6-luna** at temperature 1.0 is prompted to pick one fruit out of ten, at random. An arm is one combination of prompt language and a response format. There is seventeen arms in total, each sampled 2 000 times, for 34 000 rows. Each split is one question.

| Split | Languages | Response schema | List order |
| --- | --- | --- | --- |
| `001a` | en, pl, ja | English schema | fixed |
| `001b` | en, pl, ja | English schema | a second fixed order |
| `001c` | en, pl, ja | English schema | shuffled per sample |
| `001d` | en, pl, ja | English schema, native schema, no schema | shuffled per sample |

`001a` is the baseline distribution. `001b` re-asks it in all three languages under a different fixed order, which separates a preference for a fruit from a preference for a position. `001c` reshuffles the list for every sample, which removes position as a confound within a language. `001d` asks each language under the English schema, under that language's own schema, and under no schema at all, leaving the response schema as the only variable. English contributes two arms rather than three, because its native schema is the English schema. Every `001d` prompt adds a "one fruit name only" instruction, because the no-schema arm has nothing else constraining it.

A native schema's name is ASCII, because the provider constrains it: the Japanese one is written in romaji, and the Polish one drops its diacritic.

A shuffle draws from a per-sample seed picked at random by the run. Every arm of a question shares that seed for a given sample, so the order shown is controlled rather than varying alongside the language. No seed is replayed across runs; the order each row actually saw is recorded in `prompt_inputs`.

#### Columns

**Identity**

| Column | Type | Meaning |
| --- | --- | --- |
| `question_id` | string | `001a` … `001d`; the only identifier the pipeline takes |
| `lang` | string | Language of the prompt |
| `model` | string | Model as requested |
| `provider` | string | Provider backend that made the call |
| `run_id` | string | `<question_id>__<UTC timestamp>`, one per `llmango run` |
| `sample_idx` | int64 | Sample number within the run; shared across arms |
| `temperature` | float64 | Sampling temperature the question declares |

**Prompt**

| Column | Type | Meaning |
| --- | --- | --- |
| `prompt_sha256` | string | Hash of the exact prompt sent |
| `prompt` | string | The exact prompt sent |
| `prompt_inputs` | string | JSON of the resolved inputs, including the fruit order shown |

**Answer**

| Column | Type | Meaning |
| --- | --- | --- |
| `raw_json` | string | Verbatim model output; free text on the no-schema arm |
| `answer` | string | The answer field read off the parsed response |
| `canonical` | string | Category the answer normalizes to; null when invalid |
| `is_valid` | boolean | Whether the answer named something on the list |
| `chosen_position` | int64 | 1-based place of the pick in the list that row saw |

**Provider response**

| Column | Type | Meaning |
| --- | --- | --- |
| `model_snapshot` | string | Dated snapshot the provider actually served |
| `finish_reason` | string | Why generation stopped |
| `refusal` | string | Provider refusal, null when there was none |
| `error` | string | Error text, null when the call succeeded |
| `response_id` | string | Provider-side id of the completion |
| `service_tier` | string | Service tier the call was served on |
| `provider_created_at` | timestamp[us, UTC] | Creation time the provider reported |
| `response_schema` | string | JSON Schema sent for this arm; null for no-schema |
| `request_envelope` | string | Request body |
| `response_envelope` | string | Verbatim response body |

**Usage and cost**

| Column | Type | Meaning |
| --- | --- | --- |
| `prompt_tokens` | int64 | Input tokens |
| `completion_tokens` | int64 | Output tokens |
| `total_tokens` | int64 | Sum the provider reported |
| `cached_tokens` | int64 | Input tokens served from cache |
| `reasoning_tokens` | int64 | Reasoning tokens, where the model reports them |
| `input_cost_usd` | float64 | Input cost at the pricing version below |
| `output_cost_usd` | float64 | Output cost at the pricing version below |
| `total_cost_usd` | float64 | Sum of the two |
| `pricing_version` | string | Date of the price table used |

**Timing**

| Column | Type | Meaning |
| --- | --- | --- |
| `generation_seconds` | float64 | Wall-clock duration of the call |
| `created_at` | timestamp[us, UTC] | When the call was made |

## Provenance and limitations

- `canonical` on the no-schema arm is assigned by `normalize`, which matches offline first and falls back to an LLM for what it cannot resolve.
- A code run appends to a split, without any replacement of already existing data, nothing is deduplicated.
- No personal data and no human subjects, every row is a model's output to a prompt in `prompts/`.

## Licence

CC BY 4.0. Attribution to Rafał Kwiecień, linking back to this dataset.

The model outputs themselves are the raw material; the collection, prompts, normalization and structure are the contribution being licensed.

## Citation

```bibtex
@misc{kwiecien_llmango,
  author = {Rafał Kwiecień},
  title  = {LLMango: visualizing AI behaviour},
  url    = {https://huggingface.co/datasets/rafalkwiecien/llmango},
  note   = {Charts and write-ups at https://llmango.rafalkwiecien.com}
}
```
