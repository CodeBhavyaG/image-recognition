"""Curated metadata used to enrich a classifier prediction into the structured
``ImageUnderstanding`` JSON (attributes + caption).

These are safe, editable defaults for the animal classes. They let the
classifier baseline emit meaningful structured output today; a real vision model
can later substitute richer, image-specific understanding behind the same schema.
"""
# attributes per English class label
CLASS_ATTRIBUTES = {
    "dog": ["domestic", "four-legged", "carnivore"],
    "horse": ["equine", "four-legged", "herbivore"],
    "elephant": ["large", "gray", "herbivore"],
    "butterfly": ["colorful", "insect", "wings"],
    "chicken": ["domestic", "poultry", "feathered"],
    "cat": ["domestic", "carnivore", "four-legged"],
    "cow": ["herbivore", "mammal", "spotted"],
    "sheep": ["herbivore", "woolly", "flock"],
    "squirrel": ["small", "rodent", "bushy tail"],
    "spider": ["arachnid", "eight-legged", "web-weaving"],
}

# typical environment per class, used by the caption template
CLASS_ENVIRONMENT = {
    "dog": "a park",
    "horse": "a meadow",
    "elephant": "a savanna",
    "butterfly": "a garden",
    "chicken": "a farm",
    "cat": "a home",
    "cow": "a pasture",
    "sheep": "a field",
    "squirrel": "a forest",
    "spider": "a woodland",
}

_VOWELS = {"a", "e", "i", "o", "u"}


def indefinite_article(word: str) -> str:
    """Return 'an' for vowel-starting words, else 'a' (simple English rule)."""
    return "an" if word and word[0].lower() in _VOWELS else "a"


def build_caption(subject: str) -> str:
    """Deterministic caption: '<A/An> <subject> in <environment>.'"""
    env = CLASS_ENVIRONMENT.get(subject, "its habitat")
    return f"{indefinite_article(subject).capitalize()} {subject} in {env}."


def class_attributes(subject: str):
    """Return the curated attribute list for a class (empty list if unknown)."""
    return CLASS_ATTRIBUTES.get(subject, [])