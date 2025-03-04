"""A comprehensive pipeline for performing Named Entity Recognition
on text documents using spaCy. It extracts standard entities, adds custom entity patterns,
combines results, summarizes entity types, and generates visualizations.
"""

# %% [markdown]
# ## Imports

# %%
import pandas as pd
import spacy
from spacy import displacy
from spacy.matcher import Matcher

# %% [markdown]
# ## Main NER pipeline
# TODO: Break down this pipeline into pure functions for testability and maintainability


# %%
def perform_ner_analysis(
    text: str, model_name: str = "en_core_web_sm"
) -> dict[str, displacy.Doc | pd.DataFrame | str]:
    """
    Perform complete Named Entity Recognition analysis on the provided text.

    Args:
        text (str): The input text to analyze

    Returns:
        dict: Results dictionary containing:
            - doc: spaCy Doc object
            - all_entities: DataFrame of all entities
            - type_summary: DataFrame of entity type counts
            - visualization_html: HTML string of entity visualization
    """

    try:
        # 1. Load model
        nlp: displacy.Language = spacy.load(model_name)
    except OSError as e:
        raise ImportError(
            f"Model {model_name} not found. Install it with: python -m spacy download {model_name}"
        )

    # 2. Process document
    doc: displacy.Doc = nlp(text)

    # 3. Extract standard entities
    entities = []
    for ent in doc.ents:
        entities.append(
            {
                "Text": ent.text,
                "Start": ent.start_char,
                "End": ent.end_char,
                "Type": ent.label_,
                "Description": spacy.explain(ent.label_),
            }
        )

    # 4. Add custom entity patterns
    matcher = Matcher(nlp.vocab)
    custom_patterns = [
        # Organizations
        (
            "ORG",
            [
                [{"LOWER": "who"}],
                [{"LOWER": "world"}, {"LOWER": "health"}, {"LOWER": "organization"}],
                [{"LOWER": "ihme"}],
                [{"LOWER": "fao"}],
                [{"LOWER": "unep"}],
                [{"LOWER": "iea"}],
                [{"LOWER": "nasa"}],
                [{"LOWER": "health"}, {"LOWER": "effects"}, {"LOWER": "institute"}],
            ],
        ),
        # Research concepts
        (
            "RESEARCH_CONCEPT",
            [
                [{"LOWER": "energy"}, {"LOWER": "ladder"}],
                [{"LOWER": "energy"}, {"LOWER": "poverty"}],
                [{"LOWER": "indoor"}, {"LOWER": "air"}, {"LOWER": "pollution"}],
                [{"LOWER": "improved"}, {"LOWER": "cook"}, {"LOWER": "stoves"}],
                [{"LOWER": "particulate"}, {"LOWER": "matter"}],
                [{"LOWER": {"IN": ["pm2.5", "pm10"]}}],
            ],
        ),
        # Energy sources
        (
            "ENERGY_SOURCE",
            [
                [{"LOWER": "biomass"}],
                [{"LOWER": "fuelwood"}],
                [{"LOWER": "charcoal"}],
                [{"LOWER": "coal"}],
                [{"LOWER": "liquefied"}, {"LOWER": "petroleum"}, {"LOWER": "gas"}],
                [{"LOWER": "crop"}, {"LOWER": "waste"}],
                [{"LOWER": "dried"}, {"LOWER": "dung"}],
            ],
        ),
        # Health conditions
        (
            "HEALTH_CONDITION",
            [
                [{"LOWER": "pneumonia"}],
                [{"LOWER": "copd"}],
                [
                    {"LOWER": "chronic"},
                    {"LOWER": "obstructive"},
                    {"LOWER": "pulmonary"},
                    {"LOWER": "disease"},
                ],
                [{"LOWER": "lung"}, {"LOWER": "cancer"}],
                [{"LOWER": "cataracts"}],
                [{"LOWER": "burns"}],
                [{"LOWER": "stillbirths"}],
            ],
        ),
        # Geographic regions
        (
            "GEOG",
            [
                [{"LOWER": "sub-saharan"}, {"LOWER": "africa"}],
                [{"LOWER": "africa"}],
                [{"LOWER": "asia"}],
                [{"LOWER": "latin"}, {"LOWER": "america"}],
                [{"LOWER": "kenya"}],
                [{"LOWER": "china"}],
                [{"LOWER": "india"}],
                [{"LOWER": "rome"}],
                [{"LOWER": "delhi"}],
            ],
        ),
        # Measurements
        (
            "MEASUREMENT",
            [
                [
                    {"LOWER": "micrograms"},
                    {"LOWER": "per"},
                    {"LOWER": "cubic"},
                    {"LOWER": "metre"},
                ],
                [{"TEXT": {"REGEX": "\\d+\\s*µg/m3"}}],
                [{"TEXT": {"REGEX": "\\d+\\s*gigatons"}}],
            ],
        ),
    ]

    for name, patterns in custom_patterns:
        matcher.add(name, patterns)

    matches: list[displacy.Span] = matcher(doc)
    custom_entities = []
    for match_id, start, end in matches:
        span = doc[start:end]
        custom_entities.append(
            {
                "Text": span.text,
                "Start": span.start_char,
                "End": span.end_char,
                "Type": nlp.vocab.strings[match_id],
                "Description": "Custom entity",
            }
        )

    # 5. Combine all entities
    all_entities: pd.DataFrame = pd.concat(
        [pd.DataFrame(entities), pd.DataFrame(custom_entities)]
        if custom_entities
        else [pd.DataFrame(entities)],
        ignore_index=True,
    ).sort_values("Start")

    # 6. Summarize by entity type
    type_summary: pd.DataFrame = all_entities["Type"].value_counts().reset_index()
    type_summary.columns = ["Entity Type", "Count"]

    # 7. Generate visualization HTML
    html: str = displacy.render(doc, style="ent", page=True)

    return {
        "doc": doc,
        "all_entities": all_entities,
        "type_summary": type_summary,
        "visualization_html": html,
    }


# %%
if __name__ == "__main__":
    # Example usage with sample text
    sample_text = open("texts/ourworldindata.md").read()
    # Run the analysis
    results = perform_ner_analysis(sample_text)

    # Display results
    print(
        f"Found {len(results['all_entities'])} entities of {len(results['type_summary'])} different types"
    )

    # Show entity type distribution
    print("\nEntity types distribution:")
    print(results["type_summary"])

    # Show all found entities
    print("\nEntities found (sorted by position):")
    print(results["all_entities"][["Text", "Type", "Description"]])

    # Save visualization to HTML file
    output_file = "plots/ner_visualization.html"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(results["visualization_html"])
    print(f"\nEntity visualization saved to {output_file}")
