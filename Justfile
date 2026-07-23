# Task shortcuts over the llmango CLI.
#
# Reference an experiment by its number (001) or full id (001_favorite_fruit);
# the CLI resolves either to the same run.
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available recipes.
default:
    @just --list

# Generate raw responses for an experiment: `just run 001 --smoke`.
run exp *args:
    uv run llmango run {{ exp }} {{ args }}

# Submit an experiment's run via the OpenAI Batch API.
batch exp *args:
    uv run llmango run {{ exp }} --batch {{ args }}

# Fetch a previously submitted batch by run id.
batch-fetch run_id:
    uv run llmango batch-fetch {{ run_id }}

# Map raw answers to canonical categories.
normalize exp *args:
    uv run llmango normalize {{ exp }} {{ args }}

# Aggregate normalized answers into the committed JSON the site reads.
analyze exp:
    uv run llmango analyze {{ exp }}

# Run the full pipeline for one experiment: `just all 001 --smoke`.
all exp *args:
    just run {{ exp }} {{ args }}
    just normalize {{ exp }}
    just analyze {{ exp }}

# Format the codebase with ruff.
format:
    uv run ruff format .

# Lint the codebase with ruff.
lint:
    uv run ruff check .

# Run the test suite.
test:
    uv run pytest

# Full quality gate: lint, format check, types, tests.
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright
    uv run pytest
