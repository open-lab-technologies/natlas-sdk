import pytest

from natlas.bench import data, scoring


@pytest.mark.parametrize(("reply", "expected"), [
    ("B", "B"),
    ("B. $75,000", "B"),
    ("(C)", "C"),
    ("Answer: D", "D"),
    ("a ti ṣe é. C", "C"),  # lowercase "a" is a Yoruba/Hausa word, not option A
    ("I am not sure", None),
])
def test_parse_choice(reply, expected):
    assert scoring.parse_choice(reply) == expected


@pytest.mark.parametrize(("reply", "expected"), [
    ("contradiction", "contradiction"),
    ("Neutral.", "neutral"),
    ("Entailment, not contradiction", "entailment"),
    ("yes", None),
])
def test_parse_nli(reply, expected):
    assert scoring.parse_nli(reply) == expected


@pytest.mark.parametrize(("reply", "expected"), [
    ("16 - 3 - 4 = 9\n9 * 2 = 18\nAnswer: 18", "18"),
    ("Answer: $1,200.", "1200"),
    ("Final Answer: 13.00", "13.00"),
    ("it costs 5 then 7", "7"),
    ("no idea", None),
])
def test_parse_number(reply, expected):
    assert scoring.parse_number(reply) == expected


def test_numbers_equal_ignores_formatting():
    assert scoring.numbers_equal("13.00", "13")
    assert not scoring.numbers_equal("12", "13")
    assert not scoring.numbers_equal(None, "13")


def test_chrf_rewards_closer_translation():
    refs = ["Olú ìlú Nàìjíríà ni Abuja."]
    assert scoring.chrf(refs, refs) == pytest.approx(100.0)
    assert scoring.chrf(["Olú ìlú ni Abuja."], refs) > scoring.chrf(["Ẹ kú àárọ̀."], refs)


def test_sample_is_seeded_and_ordered():
    items = [data.Item(f"t-yo-{i}", "p", "g") for i in range(50)]
    first = data.sample(items, 10, seed=0)
    assert first == data.sample(items, 10, seed=0)
    assert first != data.sample(items, 10, seed=1)
    ids = [int(i.id.rsplit("-", 1)[1]) for i in first]
    assert ids == sorted(ids)
    assert data.sample(items, None, seed=0) == items
