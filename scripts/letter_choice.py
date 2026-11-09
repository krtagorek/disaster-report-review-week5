"""Handout's first-step constrained letter baseline, adapted to six categories."""
import json
from baseline_common import LABELS
CHOICES = 'ABCDEF'


def letter_prompt(tokenizer, text):
    system = ('You are a historical disaster report classification assistant. '
              'Classify the report by responding with ONLY a single letter — nothing else:\n'
              + '\n'.join(f'{ch}) {label}' for ch,label in zip(CHOICES, LABELS)))
    return tokenizer.apply_chat_template([{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Disaster Report: '+text}], tokenize=False, add_generation_prompt=True)


def choice_ids(tokenizer):
    ids = []
    forms = []
    for ch in CHOICES:
        spaced = tokenizer.encode(' '+ch, add_special_tokens=False)
        plain = tokenizer.encode(ch, add_special_tokens=False)
        chosen = spaced if len(spaced) == 1 else plain
        assert len(chosen) == 1, (ch, 'Not a single token')
        ids.append(chosen[0])
        forms.append(' '+ch if len(spaced) == 1 else ch)
    assert len(set(ids)) == 6
    return ids, forms
