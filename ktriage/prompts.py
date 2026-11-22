"""Prompt text. SYSTEM, pair() and direct_set() are byte-identical to the 2026-10-22 pilot so its cache is reused."""

SYSTEM = "You are an emergency department triage nurse. Judge only from the triage information given."
SAATY = ("Use Saaty's scale: 1 = equally urgent, 3 = moderately more urgent, "
         "5 = strongly more urgent, 7 = very strongly more urgent, 9 = extremely more urgent "
         "(2, 4, 6, 8 are intermediate).")
PAIR_JSON = 'Answer as JSON: {"more_urgent": "A" or "B" or "equal", "strength": integer 1-9}'
KTAS = ("Korean Triage and Acuity Scale (KTAS): 1 = resuscitation, 2 = emergent, 3 = urgent, "
        "4 = less urgent, 5 = non-urgent.")


def pair(da, db):
    """Arms pair_bt / pair_ahp (and *_swap): winner + Saaty strength, JSON."""
    return (f"Patient A: {da}\n\nPatient B: {db}\n\nWhich patient should be seen first, and how much more "
            f"urgent is that patient? {SAATY}\n{PAIR_JSON}")


def direct_set(descs, labels):
    """Arm direct_set: all patients of a set in one prompt, one KTAS level each (pilot baseline)."""
    lines = "\n".join(f"Patient {l}: {d}" for l, d in zip(labels, descs))
    return (f"{lines}\n\nAssign each patient a Korean Triage and Acuity Scale (KTAS) level, "
            "1 = resuscitation (most urgent) to 5 = non-urgent.\nAnswer as JSON: "
            '{"levels": {"P1": integer 1-5, ...}}')


def direct_one(d):
    """Arm direct_each: one patient, one digit. Single-token answer so logprobs give P(level)."""
    return (f"Patient: {d}\n\nAssign this patient a triage level. {KTAS}\n"
            "Answer with a single digit (1, 2, 3, 4 or 5) and nothing else.")


def option_one(d, level):
    """Arm optionwise: yes/no for one level (option-wise scoring)."""
    names = {1: "resuscitation", 2: "emergent", 3: "urgent", 4: "less urgent", 5: "non-urgent"}
    return (f"Patient: {d}\n\n{KTAS}\nIs KTAS level {level} ({names[level]}) the correct triage level for "
            "this patient? Answer with a single word, Yes or No, and nothing else.")


def pair_letter(da, db):
    """Arm pairprob_*: who first, single letter so logprobs give P(A first)."""
    return (f"Patient A: {da}\n\nPatient B: {db}\n\nWhich patient should be seen first? "
            "Answer with a single letter, A or B, and nothing else.")
