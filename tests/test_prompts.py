"""Tests for Prompt class"""
from dawnet.prompts import Prompt


def test_basic_parsing():
  """Test basic template parsing"""
  prompt = Prompt("Hello {name}, welcome to {place}!")
  assert prompt.prompt == "Hello {name}, welcome to {place}!"


def test_parsing_with_defaults():
  """Test parsing variables with default values"""
  prompt = Prompt("When {boy} and {girl} went to the {location=church}, {boy} gave the {obj} to")
  assert prompt.prompt == "When {boy} and {girl} went to the {location=church}, {boy} gave the {obj} to"


def test_info_property():
  """Test info property returns correct metadata"""
  prompt = Prompt("{name} lives in {city=Paris}")
  info = prompt.info

  assert info['template'] == "{name} lives in {city=Paris}"
  assert len(info['variables']) == 2
  assert info['variables'][0]['name'] == 'name'
  assert info['variables'][0]['position'] == 0
  assert info['variables'][0]['default'] is None
  assert info['variables'][1]['name'] == 'city'
  assert info['variables'][1]['position'] == 1
  assert info['variables'][1]['default'] == 'Paris'


def test_swap_position():
  """Test swapping variables by position"""
  prompt = Prompt("When {boy} and {girl} went to the {location=church}")
  new_prompt = prompt.swap_position("pos0", "pos1")

  # Should swap boy and girl positions
  assert new_prompt.prompt == "When {girl} and {boy} went to the {location=church}"


def test_swap_variable_names():
  """Test swapping variable names"""
  prompt = Prompt("When {boy} and {girl} went to the {location}, {boy} gave the {obj} to")
  new_prompt = prompt.swap("boy", "girl")

  # All occurrences of boy become girl and vice versa
  assert new_prompt.prompt == "When {girl} and {boy} went to the {location}, {girl} gave the {obj} to"


def test_change_variable_name():
  """Test changing variable name at specific position"""
  prompt = Prompt("When {boy} and {girl} went to the {location}, {boy} gave the {obj} to")
  new_prompt = prompt.change(pos4="item")

  # pos4 is {obj}, should become {item}
  assert "{item}" in new_prompt.prompt
  assert "{obj}" not in new_prompt.prompt


def test_add_choice():
  """Test adding choices for variables"""
  prompt = Prompt("{name} likes {color}")
  prompt.add_choice(name=["Alice", "Bob"], color=["red", "blue"])

  assert prompt._choices['name'] == ["Alice", "Bob"]
  assert prompt._choices['color'] == ["red", "blue"]


def test_add_choice_chainable():
  """Test that add_choice returns self for chaining"""
  prompt = Prompt("{name} likes {color}")
  result = prompt.add_choice(name=["Alice", "Bob"])
  assert result is prompt


def test_sample_with_kwargs():
  """Test sample method with provided kwargs"""
  prompt = Prompt("Hello {name}, you are {age} years old")
  sampled = prompt.sample(name="John", age="25")

  assert sampled == {"name": "John", "age": "25"}


def test_sample_with_defaults():
  """Test sample method using default values"""
  prompt = Prompt("Welcome to {city=Paris}")
  sampled = prompt.sample()

  assert sampled == {"city": "Paris"}


def test_sample_with_choices():
  """Test sample method with choices"""
  prompt = Prompt("{name} likes {color}")
  prompt.add_choice(name=["Alice"], color=["red"])
  sampled = prompt.sample()

  # With only one choice, it should always be selected
  assert sampled == {"name": "Alice", "color": "red"}


def test_sample_priority():
  """Test sample method respects priority: kwargs > choices > defaults > category"""
  prompt = Prompt("{name=DefaultName}")
  prompt.add_choice(name=["ChoiceName"])

  # kwargs should override choices
  sampled1 = prompt.sample(name="KwargName")
  assert sampled1 == {"name": "KwargName"}

  # choices should override defaults
  sampled2 = prompt.sample()
  assert sampled2 == {"name": "ChoiceName"}


def test_sample_with_category():
  """Test sample method using _DEFAULT_CATEGORY"""
  prompt = Prompt("{boy} and {girl}")
  sampled = prompt.sample()

  # Should use values from WORDS_BY_CATEGORY
  assert "boy" in sampled
  assert "girl" in sampled
  assert isinstance(sampled["boy"], str)
  assert isinstance(sampled["girl"], str)


def test_sample_excludes_unknown():
  """Test that sample excludes variables it cannot sample"""
  prompt = Prompt("{known=default} and {unknown}")
  sampled = prompt.sample()

  # Should only include known variable
  assert "known" in sampled
  assert sampled["known"] == "default"
  assert "unknown" not in sampled


def test_sample_same_variable_multiple_times():
  """Test that sample returns one value for variables that appear multiple times"""
  prompt = Prompt("{name} and {name} are friends with {name}")
  sampled = prompt.sample(name="Alice")

  # Should only have one entry for 'name'
  assert sampled == {"name": "Alice"}


def test_materialize_with_kwargs():
  """Test materialization with provided values"""
  prompt = Prompt("Hello {name}, you are {age} years old")
  result = prompt.materialize(name="John", age="25")

  assert result == "Hello John, you are 25 years old"


def test_materialize_with_defaults():
  """Test materialization using default values"""
  prompt = Prompt("Welcome to {city=Paris}")
  result = prompt.materialize()

  assert result == "Welcome to Paris"


def test_materialize_with_choices():
  """Test materialization with choices (random selection)"""
  prompt = Prompt("{name} likes {color}")
  prompt.add_choice(name=["Alice"], color=["red"])
  result = prompt.materialize()

  # With only one choice, it should always be selected
  assert result == "Alice likes red"


def test_materialize_priority_kwargs_over_choices():
  """Test that kwargs have priority over choices"""
  prompt = Prompt("{name}")
  prompt.add_choice(name=["Alice", "Bob"])
  result = prompt.materialize(name="Charlie")

  assert result == "Charlie"


def test_materialize_priority_choices_over_defaults():
  """Test that choices have priority over defaults"""
  prompt = Prompt("{name=DefaultName}")
  prompt.add_choice(name=["ChoiceName"])
  result = prompt.materialize()

  assert result == "ChoiceName"


def test_materialize_same_variable_multiple_times():
  """Test that same variable gets same value across occurrences"""
  prompt = Prompt("{name} and {name} are friends")
  result = prompt.materialize(name="Alice")

  assert result == "Alice and Alice are friends"


def test_materialize_with_category():
  """Test materialization using _DEFAULT_CATEGORY"""
  prompt = Prompt("{boy} went to the {location}")
  result = prompt.materialize()

  # Should use values from WORDS_BY_CATEGORY
  # boy and location should be materialized (not left as {boy}, {location})
  assert "{boy}" not in result
  assert "{location}" not in result


def test_materialize_unknown_variable():
  """Test that unknown variables without any fallback stay as-is"""
  prompt = Prompt("{unknown_var} is here")
  result = prompt.materialize()

  # Should remain as template variable
  assert result == "{unknown_var} is here"


def test_save_and_load(tmp_path):
  """Test saving and loading prompts"""
  # Create a temporary file path
  filepath = tmp_path / "test_prompt.yaml"

  # Create and configure prompt
  prompt = Prompt("Hello {name}, you live in {city=Paris}")
  prompt.add_choice(name=["Alice", "Bob"], city=["Paris", "London"])

  # Save
  prompt.save(str(filepath))

  # Load
  loaded_prompt = Prompt.load(str(filepath))

  # Verify
  assert loaded_prompt.prompt == prompt.prompt
  assert loaded_prompt._choices == prompt._choices


def test_save_load_preserves_functionality(tmp_path):
  """Test that loaded prompt works correctly"""
  filepath = tmp_path / "test_prompt2.yaml"

  # Create, configure, and save
  prompt = Prompt("{name} likes {color}")
  prompt.add_choice(name=["Charlie"], color=["blue"])
  prompt.save(str(filepath))

  # Load and use
  loaded = Prompt.load(str(filepath))
  result = loaded.materialize()

  assert result == "Charlie likes blue"


def test_complex_workflow():
  """Test the example from docstring"""
  prompt = Prompt("When {boy} and {girl} went to the {location=church}, {boy} gave the {obj} to")

  # Test operations
  new_prompt = prompt.swap_position("pos0", "pos1")
  assert "{girl} and {boy}" in new_prompt.prompt

  new_prompt2 = prompt.swap("boy", "girl")
  assert new_prompt2.prompt == "When {girl} and {boy} went to the {location=church}, {girl} gave the {obj} to"

  new_prompt3 = new_prompt2.change(pos3="item")
  assert "{item}" in new_prompt3.prompt

  new_prompt3.add_choice(girl=["Alice", "Mary"])
  result = new_prompt3.materialize(girl="Alice")

  # girl should be Alice, location should be church (default)
  assert "Alice" in result
  assert "church" in result


def test_empty_template():
  """Test handling of template with no variables"""
  prompt = Prompt("Hello world")
  assert prompt.prompt == "Hello world"
  assert len(prompt.info['variables']) == 0
  assert prompt.materialize() == "Hello world"


def test_consecutive_variables():
  """Test variables without text between them"""
  prompt = Prompt("{first}{second}{third}")
  result = prompt.materialize(first="A", second="B", third="C")
  assert result == "ABC"
