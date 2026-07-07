"""Do two texts share an author? Same-author probability with the features
that drove the verdict. Needs biberplus[neural]; downloads the PAN 2020
forest from the Hub on first use (or set NEUROBIBER_AV_PATH to a local .pkl).

    python examples/author_comparison.py
"""
from biberplus.neurobiber.authorship import compare_texts

TEXT_A = (
    "I wasn't going to open the door. I want that on the record, because "
    "everything after only happened because Marcy knocked twice, waited, "
    "and knocked again in that particular way she has, like the wood owed "
    "her money. So I opened it. Of course I did."
)
TEXT_B = (
    "The structure was considered unremarkable by the county's surveyors: "
    "a squat tower of granite block, a storehouse with a collapsed roof, "
    "and a walled garden long since surrendered to gorse. An inspection "
    "was commissioned in 1989; its findings were recorded without comment."
)

result = compare_texts(TEXT_A, TEXT_B)
print(f"Same-author probability: {result['same_author_probability']} "
      f"-> {result['verdict']}")
for d in result["drivers"][:6]:
    print(f"  {d['code']:7s} {d['name']:28s} "
          f"{'matches' if d['agree'] else 'differs'}")
print(result["caveat"])
