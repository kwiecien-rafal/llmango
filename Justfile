# Task shortcuts over the llmango CLI.
# Every pipeline recipe takes a question id (001a, 001b, ...); analyze takes none,
# because a chart is an experiment-level artifact and may read several questions.
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# List available recipes.
default:
    @just --list

# Generate raw responses for a question: `just run 001a -n 5`.
run question *args:
    uv run llmango run {{ question }} {{ args }}

# Map raw answers to canonical categories.
normalize question *args:
    uv run llmango normalize {{ question }} {{ args }}

# Aggregate normalized answers into the committed JSON the charts read.
aggregate question:
    uv run llmango aggregate {{ question }}

# Redraw every experiment's charts the site embeds, under site/public/charts/.
analyze:
    uv run llmango analyze

# Upload every normalized question to HuggingFace; the card is DATASET_CARD.md
# under generated frontmatter. `just publish --dry-run` needs no token.
publish *args:
    uv run llmango publish {{ args }}

# Run a question, then normalize, aggregate and chart it: `just all 001a -n 5`.
all question *args:
    just run {{ question }} {{ args }}
    just normalize {{ question }}
    just aggregate {{ question }}
    just analyze

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
