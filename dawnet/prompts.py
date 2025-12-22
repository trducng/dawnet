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

WORDS_BY_CATEGORIES = {
  "boy_first_names": [
    "James", "John", "Michael", "David", "Daniel",
    "Matthew", "Andrew", "Thomas", "Mark", "Paul",
    "Peter", "Kevin", "Brian"
  ],
  "girl_first_names": [
    "Mary", "Sarah", "Emma", "Emily", "Jessica",
    "Anna", "Laura", "Lisa", "Jennifer", "Karen",
    "Amy", "Rachel", "Susan"
  ],
  "countries": [
    "France", "Spain", "Italy", "Japan", "China",
    "Brazil", "Canada", "Mexico", "India", "Egypt",
    "Greece", "Turkey", "Peru"
  ],
  "capitals": [
    "Paris", "London", "Rome", "Tokyo", "Berlin",
    "Madrid", "Ottawa", "Cairo", "Athens", "Moscow",
    "Beijing", "Delhi", "Lima"
  ],
  "colors": [
    "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "black", "white",
    "gray", "silver", "gold"
  ],
  "months": [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December"
  ],
  "emotions": [
    "happy", "sad", "angry", "excited", "scared",
    "surprised", "confused", "proud", "worried", "calm",
    "nervous", "grateful", "lonely"
  ],
  "action_verbs": [
    "run", "jump", "walk", "swim", "dance",
    "sing", "eat", "sleep", "read", "write",
    "play", "talk", "laugh"
  ],
  "occupations": [
    "teacher", "doctor", "nurse", "chef", "pilot",
    "farmer", "artist", "writer", "lawyer", "engineer",
    "dentist", "police", "firefighter"
  ]
}

def get_words(
  prepend_space=True,
  with_category=True,
  n_per_category=-1
) -> list[str] | dict[str, list]:
  if with_category:
    wd = {}
    for k, v in WORDS_BY_CATEGORIES.items():
      if n_per_category > 0:
        v = v[:n_per_category]
      if prepend_space:
        wd[k] = [f" {i}" for i in v]
      else:
        wd[k] = v
    return wd

  wl = []
  for v in WORDS_BY_CATEGORIES.values():
    if n_per_category > 0:
      v = v[:n_per_category]
    if prepend_space:
      wl += [f" {i}" for i in v]
    else:
      wl += v
  return wl


PROMPTS = MATH + QA + IOI
# vim: ts=2 sts=2 sw=2 et
