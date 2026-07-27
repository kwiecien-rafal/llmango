# Task shortcuts over the llmango CLI.
#
# Every recipe takes a question id (001a, 001b, ...). Nothing takes an experiment:
# a question id is the only identifier the pipeline resolves.
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available recipes.
default:
    @just --list

# Generate raw responses for a question: `just run 001a --smoke`.
# Submit through the OpenAI Batch API with `just run 001a --batch`.
run question *args:
    uv run llmango run {{ question }} {{ args }}

# Fetch a previously submitted batch by run id.
batch-fetch run_id:
    uv run llmango batch-fetch {{ run_id }}

# Map raw answers to canonical categories.
normalize question *args:
    uv run llmango normalize {{ question }} {{ args }}

# Aggregate normalized answers into the committed JSON the charts read.
aggregate question:
    uv run llmango aggregate {{ question }}

# Draw the charts the site embeds, under site/public/charts/.
analyze question:
    uv run llmango analyze {{ question }}

# Run a question, then normalize, aggregate and chart it: `just all 001a --smoke`.
all question *args:
    just run {{ question }} {{ args }}
    just normalize {{ question }}
    just aggregate {{ question }}
    just analyze {{ question }}

# Serve the site with hot reload; charts refresh as `just analyze` rewrites them.
site:
    npm --prefix site run dev

# Build the static site into site/dist/.
site-build:
    npm --prefix site run build

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
