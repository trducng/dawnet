"""Helper prompts to be used with debugging"""
import random
import re
import warnings

import yaml


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

WORDS_BY_CATEGORY = {
  "boy": [
    "James", "John", "Michael", "David", "Daniel",
    "Matthew", "Andrew", "Thomas", "Mark", "Paul",
    "Peter", "Kevin", "Brian"
  ],
  "girl": [
    "Mary", "Sarah", "Emma", "Emily", "Jessica",
    "Anna", "Laura", "Lisa", "Jennifer", "Karen",
    "Amy", "Rachel", "Susan"
  ],
  "country": [
    "France", "Spain", "Italy", "Japan", "China",
    "Brazil", "Canada", "Mexico", "India", "Egypt",
    "Greece", "Turkey", "Peru"
  ],
  "capital": [
    "Paris", "London", "Rome", "Tokyo", "Berlin",
    "Madrid", "Ottawa", "Cairo", "Athens", "Moscow",
    "Beijing", "Delhi", "Lima"
  ],
  "color": [
    "red", "blue", "green", "yellow", "orange",
    "purple", "pink", "brown", "black", "white",
    "gray", "silver", "gold"
  ],
  "month": [
    "January", "February", "March", "April", "May",
    "June", "July", "August", "September", "October",
    "November", "December"
  ],
  "emotion": [
    "happy", "sad", "angry", "excited", "scared",
    "surprised", "confused", "proud", "worried", "calm",
    "nervous", "grateful", "lonely"
  ],
  "action": [
    "run", "jump", "walk", "swim", "dance",
    "sing", "eat", "sleep", "read", "write",
    "play", "talk", "laugh"
  ],
  "job": [
    "teacher", "doctor", "nurse", "chef", "pilot",
    "farmer", "artist", "writer", "lawyer", "engineer",
    "dentist", "police", "firefighter"
  ],
  "location": [
    "shop", "church", "school", "hospital", "library",
    "park", "museum", "restaurant", "bank", "hotel",
    "airport", "station", "beach", "market", "gym"
  ],
  
  "object": [
    "computer", "phone", "book", "chair", "table",
    "lamp", "clock", "pen", "cup", "plate",
    "bag", "key", "mirror", "camera", "watch"
  ],

}

def get_words(
  prepend_space=True,
  with_category=True,
  n_per_category=-1
) -> list[str] | dict[str, list]:
  if with_category:
    wd = {}
    for k, v in WORDS_BY_CATEGORY.items():
      if n_per_category > 0:
        v = v[:n_per_category]
      if prepend_space:
        wd[k] = [f" {i}" for i in v]
      else:
        wd[k] = v
    return wd

  wl = []
  for v in WORDS_BY_CATEGORY.values():
    if n_per_category > 0:
      v = v[:n_per_category]
    if prepend_space:
      wl += [f" {i}" for i in v]
    else:
      wl += v
  return wl


PROMPTS = MATH + QA + IOI
# vim: ts=2 sts=2 sw=2 et


class Prompt:
  """Prompt manipulation

  Example:

    ```
    # location has default value as "church"
    prompt = Prompt("When {boy} and {girl} went to the {location=church}, {boy} gave the {obj} to")

    # swap position of variables -> "When {girl} and {boy} went..."
    new_prompt = prompt.swap_position("pos0", "pos1")

    # swap variable -> "When {girl} and {boy} went...
    new_prompt = prompt.swap("boy", "girl")

    # switch variable name to another variable name
    new_prompt = new_prompt.change(pos3="name2")

    # choose each variable value from a list
    new_prompt.add_choice(boy=["name1", "name2"])

    # materialize, can optionally provide direct value
    new_prompt.materialize(boy="abc")

    # print prompt
    print(prompt.prompt)

    # print prompt info (as dict)
    print(prompt.info)

    # persist to disk, maybe as human-readable yaml file
    prompt.save(path)

    # load from disk
    prompt = Prompt.load(path)
    ```
  """

  _DEFAULT_CATEGORY = WORDS_BY_CATEGORY.copy()

  def __init__(self, s: str = None, parts: list = None, choices: dict = None):
    if parts is not None:
      # Internal constructor with pre-built parts
      self._parts = parts
      self._choices = choices or {}
    else:
      # Parse template string
      self._parts = []
      self._choices = {}

      # Pattern to match {varname} or {varname=default}
      pattern = r'\{([^}=]+)(?:=([^}]+))?\}'
      last_end = 0
      var_position = 0

      for match in re.finditer(pattern, s):
        # Add text before this variable
        if match.start() > last_end:
          self._parts.append(s[last_end:match.start()])

        # Add variable info
        var_name = match.group(1)
        var_default = match.group(2)
        self._parts.append({
          'name': var_name,
          'default': var_default,
          'position': var_position
        })
        var_position += 1
        last_end = match.end()

      # Add remaining text
      if last_end < len(s):
        self._parts.append(s[last_end:])

  @property
  def prompt(self) -> str:
    """Reconstruct template string from parts"""
    result = []
    for part in self._parts:
      if isinstance(part, str):
        result.append(part)
      else:
        # Variable
        if part['default']:
          result.append(f"{{{part['name']}={part['default']}}}")
        else:
          result.append(f"{{{part['name']}}}")
    return ''.join(result)

  @property
  def info(self) -> dict:
    """Return dict with variables, defaults, choices, positions"""
    variables = []
    for part in self._parts:
      if isinstance(part, dict):
        variables.append({
          'position': part['position'],
          'name': part['name'],
          'default': part['default'],
          'choices': self._choices.get(part['name'])
        })
    return {
      'template': self.prompt,
      'variables': variables,
      'choices': self._choices
    }

  def swap_position(self, pos1: str, pos2: str):
    """Swap variables at two positions"""
    # Extract position numbers
    p1 = int(pos1.replace('pos', ''))
    p2 = int(pos2.replace('pos', ''))

    # Copy parts
    new_parts = []
    for part in self._parts:
      if isinstance(part, str):
        new_parts.append(part)
      else:
        new_parts.append(part.copy())

    # Find and swap
    var1_idx = None
    var2_idx = None
    for i, part in enumerate(new_parts):
      if isinstance(part, dict):
        if part['position'] == p1:
          var1_idx = i
        if part['position'] == p2:
          var2_idx = i

    if var1_idx is not None and var2_idx is not None:
      new_parts[var1_idx], new_parts[var2_idx] = new_parts[var2_idx], new_parts[var1_idx]

    return Prompt(parts=new_parts, choices=self._choices.copy())

  def swap(self, var1: str, var2: str):
    """Swap all occurrences of var1 with var2"""
    new_parts = []
    for part in self._parts:
      if isinstance(part, str):
        new_parts.append(part)
      else:
        new_part = part.copy()
        if new_part['name'] == var1:
          new_part['name'] = var2
        elif new_part['name'] == var2:
          new_part['name'] = var1
        new_parts.append(new_part)

    return Prompt(parts=new_parts, choices=self._choices.copy())

  def change(self, **kwargs):
    """Change variable name at specific positions"""
    new_parts = []
    for part in self._parts:
      if isinstance(part, str):
        new_parts.append(part)
      else:
        new_part = part.copy()
        pos_key = f"pos{part['position']}"
        if pos_key in kwargs:
          new_part['name'] = kwargs[pos_key]
        new_parts.append(new_part)

    return Prompt(parts=new_parts, choices=self._choices.copy())

  def add_choice(self, **kwargs):
    """Add choices for variables"""
    for var_name, choices in kwargs.items():
      self._choices[var_name] = choices
    return self

  def sample(self, **kwargs):
    """Sample values for all variables

    Returns a dictionary mapping variable names to sampled values.
    Priority: kwargs > choices > default > category
    Variables that cannot be sampled are excluded from the result.

    Args:
      **kwargs: Direct values to use for specific variables

    Returns:
      dict: Variable names mapped to their sampled values
    """
    var_values = {}
    for part in self._parts:
      if isinstance(part, dict):
        var_name = part['name']
        if var_name not in var_values:
          # Priority: kwargs > choices > default > category
          if var_name in kwargs:
            var_values[var_name] = kwargs[var_name]
          elif var_name in self._choices:
            var_values[var_name] = random.choice(self._choices[var_name])
          elif part['default']:
            var_values[var_name] = part['default']
          elif var_name in self._DEFAULT_CATEGORY:
            var_values[var_name] = random.choice(self._DEFAULT_CATEGORY[var_name])
          # Otherwise, skip this variable (don't include in result)

    return var_values

  def materialize(self, **kwargs):
    """Replace variables with actual values"""
    # Check for non-existent parameters
    if kwargs:
      # Get all unique variable names from template
      template_vars = set()
      for part in self._parts:
        if isinstance(part, dict):
          template_vars.add(part['name'])

      # Warn about kwargs that don't exist in template
      for kwarg_name in kwargs:
        if kwarg_name not in template_vars:
          warnings.warn(
            f"Parameter '{kwarg_name}' does not exist in template. "
            f"Available variables: {sorted(template_vars)}",
            UserWarning,
            stacklevel=2
          )

    # Get sampled values for all variables
    var_values = self.sample(**kwargs)

    # Build result
    result = []
    for part in self._parts:
      if isinstance(part, str):
        result.append(part)
      else:
        var_name = part['name']
        # Use sampled value if available, otherwise leave as template variable
        if var_name in var_values:
          result.append(var_values[var_name])
        else:
          result.append(f"{{{var_name}}}")

    return ''.join(result)

  def save(self, path: str):
    """Save to YAML file"""
    data = {
      'template': self.prompt,
      'choices': self._choices
    }
    with open(path, 'w') as f:
      yaml.dump(data, f, default_flow_style=False)

  @classmethod
  def load(cls, path: str):
    """Load from YAML file"""
    with open(path, 'r') as f:
      data = yaml.safe_load(f)

    prompt = cls(data['template'])
    if 'choices' in data:
      prompt._choices = data['choices']
    return prompt
