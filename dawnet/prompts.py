"""Helper prompts to be used with debugging"""

MATH = [
  """What is (234 + 413) * 89?"""  # Reasoning and math
]

QA = [
  """What is the largest city in Europe?"""    # Moscow, Istanbul
]

IOI = [
  """When John and Mary went to the store, John gave the bag to"""
]

Winograd_style = [
  (
    "Robert woke up at 8:00am while Samuel woke up at 6:00am, "
    "so the person woke up later is",
    "Robert woke up at 8:00am while Samuel woke up at 9:00am, "
    "so the person woke up later is"
  )
]

PROMPTS = MATH + QA + IOI
# vim: ts=2 sts=2 sw=2 et
