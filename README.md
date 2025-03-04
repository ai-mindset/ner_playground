# Named Entity Recognition Playground 

A comprehensive pipeline for performing Named Entity Recognition (NER) on text documents using spaCy.

## Features

- Extract standard named entities using spaCy's pre-trained models
- Add custom entity patterns for domain-specific NER
- Generate visualisations of recognised entities
- Produce detailed analysis of entity type distribution
- Save results in structured formats

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ai-mindset/ner_playground.git
   cd ner_playground 
   ```

2. Create a virtual environment with UV:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the package and development dependencies:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. Install the spaCy model:
   ```bash
   # Download the model wheel directly
   curl -LO https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

   # Install it directly with UV
   uv pip install en_core_web_sm-3.8.0-py3-none-any.whl
   ```

## Usage

Run the NER analysis on a text file:

```bash
python -m src.ner.main --input texts/sample.txt --output plots/entities.html
```

### API Usage

```python
from src.ner.main import perform_ner_analysis

# Sample text for analysis
text = """spaCy is an open-source software library for advanced natural language processing, written in Python and Cython. The main developers are Matthew Honnibal and Ines Montani."""

# Run the analysis
results = perform_ner_analysis(text)

# Access the entities found
entities_df = results["all_entities"]
print(entities_df)

# The HTML visualisation is available at
html_viz = results["visualization_html"]
```

## Project Structure

```
ner_playground/
├── pyproject.toml         # Project configuration and dependencies
├── src/
│   └── ner/
│       └── main.py        # Main NER pipeline implementation
├── texts/                 # Sample texts for analysis
└── plots/                 # Output directory for visualisations
```

## How It Works

The NER pipeline performs the following steps:

1. Loads the spaCy language model (`en_core_web_sm`)
2. Processes the input text to create a spaCy document
3. Extracts standard named entities (people, organisations, locations, etc.)
4. Applies custom entity patterns for domain-specific terminology
5. Combines all entities and sorts them by position in the text
6. Generates a summary of entity type distribution
7. Creates an HTML visualisation of the entities in context
8. Returns structured results for further analysis

## Customisation

You can customise the NER pipeline by modifying the custom patterns in `src/ner/main.py`. The default implementation includes patterns for programming languages and libraries.

## Troubleshooting

If you encounter errors with spaCy model loading:

1. Verify the model is installed correctly:
   ```bash
   python -c "import spacy; print(spacy.util.get_installed_models())"
   ```

2. If the model is not listed, reinstall using the method in the Installation section.

## Licence

MIT 
