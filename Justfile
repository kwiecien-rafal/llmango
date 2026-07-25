# Task shortcuts over the llmango CLI.
#
# `run` takes a question (001a, 001b, ...); `normalize`, `aggregate` and `analyze`
# take an experiment by its number (001) or full id (001_fruit).
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available recipes.
default:
    @just --list

# Generate raw responses for a question: `just run 001a --smoke`.
run question *args:
    uv run llmango run {{ question }} {{ args }}

# Submit a question's run via the OpenAI Batch API.
batch question *args:
    uv run llmango run {{ question }} --batch {{ args }}

# Fetch a previously submitted batch by run id.
batch-fetch run_id:
    uv run llmango batch-fetch {{ run_id }}

# Map raw answers to canonical categories.
normalize exp *args:
    uv run llmango normalize {{ exp }} {{ args }}

# Aggregate normalized answers into the committed JSON the charts read.
aggregate exp:
    uv run llmango aggregate {{ exp }}

# Draw the charts the site embeds, under site/public/charts/<experiment_id>/.
analyze exp:
    uv run llmango analyze {{ exp }}

# Run a question, then normalize, aggregate and chart it: `just all 001a 001 --smoke`.
all question exp *args:
    just run {{ question }} {{ args }}
    just normalize {{ exp }}
    just aggregate {{ exp }}
    just analyze {{ exp }}

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
