
import spacy


class PatentNER:

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

        # Add patent-specific entity recognition rules
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = [
            {"label": "MATERIAL", "pattern": "copper"},
            {"label": "MATERIAL", "pattern": "silicon"},
            {"label": "MATERIAL", "pattern": "stainless steel"},
            {"label": "MATERIAL", "pattern": "aluminium"},
            {"label": "MATERIAL", "pattern": "lithium"},

            {"label": "COMPONENT", "pattern": "lithium-ion battery"},
            {"label": "COMPONENT", "pattern": "semiconductor device"},
            {"label": "COMPONENT", "pattern": "transistor"},
            {"label": "COMPONENT", "pattern": "transformer"},
            {"label": "COMPONENT", "pattern": "copper conductor"},
            {"label": "COMPONENT", "pattern": "photovoltaic cell"},

            {"label": "TECHNOLOGY", "pattern": "wireless communication"},
            {"label": "TECHNOLOGY", "pattern": "solar energy"},
            {"label": "TECHNOLOGY", "pattern": "machine learning"},
            {"label": "TECHNOLOGY", "pattern": "artificial intelligence"},
        ]

        ruler.add_patterns(patterns)

    def extract_entities(self, text):
        doc = self.nlp(text)

        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
               "description": {
    "COMPONENT": "Patent component or device",
    "MATERIAL": "Material used in an invention",
    "TECHNOLOGY": "Technology or technical method"
}.get(ent.label_, spacy.explain(ent.label_) or ent.label_)
            })

        return entities


if __name__ == "__main__":

    ner = PatentNER()

    sample_text = """
    A high-voltage electrical system uses copper conductors,
    silicon semiconductor devices and a lithium-ion battery.
    The system was developed by Siemens in Germany.
    """

    for entity in ner.extract_entities(sample_text):
        print(entity)
