## SmartTour

SmartTour is a travel-planning PoC that turns natural language preferences into a multi-day itinerary.

### Run

```bash
uv sync
uv run python -m smarttour.db.seed
uv run fastapi dev smarttour/api/app.py
uv run streamlit run smarttour/ui/app.py
```

### Quality Checks

```bash
uv run ruff check smarttour tests scripts
uv run ruff format smarttour tests scripts
uv run mypy smarttour tests scripts
uv run pytest tests
```
