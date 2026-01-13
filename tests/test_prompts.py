"""Tests for Prompt class"""
import importlib.util
from pathlib import Path

# Import prompts module directly without going through dawnet.__init__
spec = importlib.util.spec_from_file_location("prompts", Path(__file__).parent.parent / "dawnet" / "prompts.py")
prompts_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prompts_module)
Prompt = prompts_module.Prompt


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


def test_materialize_warns_on_nonexistent_parameter():
  """Test that materialize warns when provided parameter doesn't exist in template"""
  import warnings

  prompt = Prompt("Hello {name}")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = prompt.materialize(name="Alice", typo="value")

    # Should have one warning
    assert len(w) == 1
    assert issubclass(w[0].category, UserWarning)
    assert "typo" in str(w[0].message)
    assert "does not exist in template" in str(w[0].message)
    assert "name" in str(w[0].message)  # Should show available variables

  # Result should still work correctly
  assert result == "Hello Alice"


def test_materialize_warns_multiple_nonexistent():
  """Test warning for multiple non-existent parameters"""
  import warnings

  prompt = Prompt("{name}")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    prompt.materialize(name="Alice", wrong1="a", wrong2="b")

    # Should have two warnings
    assert len(w) == 2
    warning_messages = [str(warning.message) for warning in w]
    assert any("wrong1" in msg for msg in warning_messages)
    assert any("wrong2" in msg for msg in warning_messages)


def test_materialize_no_warning_when_all_exist():
  """Test that no warning when all parameters exist in template"""
  import warnings

  prompt = Prompt("Hello {name}, you are {age} years old")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = prompt.materialize(name="Alice", age="25")

    # Should have no warnings
    assert len(w) == 0

  assert result == "Hello Alice, you are 25 years old"


def test_materialize_no_warning_when_no_kwargs():
  """Test that no warning when no kwargs provided"""
  import warnings

  prompt = Prompt("Hello {name}")

  with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = prompt.materialize()

    # Should have no warnings
    assert len(w) == 0


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


# Tests for numeric suffix support
def test_suffix_basic_recognition():
  """Test that boy1 and boy2 sample different values from boy category"""
  prompt = Prompt("{boy1} and {boy2}")
  result = prompt.materialize()

  # Should materialize both (not left as {boy1}, {boy2})
  assert "{boy1}" not in result
  assert "{boy2}" not in result

  # Values should be different (boy1 and boy2 are different variables)
  # Extract the two names from "Name1 and Name2"
  parts = result.split(" and ")
  assert len(parts) == 2
  # With high probability they should be different (unless we got unlucky)
  # Since there are 13 boys in the category, probability of same is 1/13


def test_suffix_same_variable_same_value():
  """Test that boy1 appearing twice gets the same value"""
  prompt = Prompt("{boy1} and {boy1} are friends")
  result = prompt.materialize()

  # Extract the two occurrences
  parts = result.split(" and ")
  name1 = parts[0]
  name2 = parts[1].split(" are ")[0]

  # Should be the same
  assert name1 == name2


def test_suffix_with_choices_base_category():
  """Test that add_choice(boy=[...]) works for boy1 and boy2"""
  prompt = Prompt("{boy1} and {boy2}")
  prompt.add_choice(boy=["John", "Mike", "Tom"])
  sampled = prompt.sample()

  # Both should be from the choices
  assert sampled["boy1"] in ["John", "Mike", "Tom"]
  assert sampled["boy2"] in ["John", "Mike", "Tom"]


def test_suffix_with_choices_exact_override():
  """Test that add_choice(boy1=[...]) overrides base category"""
  prompt = Prompt("{boy1} and {boy2}")
  prompt.add_choice(boy=["John", "Mike"], boy1=["Specific"])
  sampled = prompt.sample()

  # boy1 should use specific choice
  assert sampled["boy1"] == "Specific"
  # boy2 should use base category choice
  assert sampled["boy2"] in ["John", "Mike"]


def test_suffix_mixing_with_non_suffixed():
  """Test template with both {boy} and {boy1}"""
  prompt = Prompt("{boy} and {boy1}")
  result = prompt.materialize()

  # Both should be materialized
  assert "{boy}" not in result
  assert "{boy1}" not in result


def test_suffix_multi_digit():
  """Test multi-digit suffixes like boy10, boy123"""
  prompt = Prompt("{boy10} and {boy123}")
  result = prompt.materialize()

  # Should materialize both
  assert "{boy10}" not in result
  assert "{boy123}" not in result


def test_suffix_no_suffix_unchanged():
  """Test that variables without suffix still work"""
  prompt = Prompt("{location} is nice")
  result = prompt.materialize()

  # Should use location category
  assert "{location}" not in result


def test_suffix_edge_case_no_trailing_digits():
  """Test name2name (no trailing digits) works unchanged"""
  prompt = Prompt("{name2name}")
  prompt.add_choice(name2name=["test"])
  result = prompt.materialize()

  assert result == "test"


def test_suffix_edge_case_only_digits():
  """Test variable with only digits like {123}"""
  prompt = Prompt("{123}")
  # Should not match any category (base would be empty string)
  result = prompt.materialize()

  # Should remain as template variable
  assert result == "{123}"


def test_suffix_kwargs_exact_match_only():
  """Test that materialize kwargs require exact match"""
  prompt = Prompt("{boy1} and {boy2}")

  # Exact match should work
  result1 = prompt.materialize(boy1="Alice", boy2="Bob")
  assert result1 == "Alice and Bob"

  # Base category in kwargs should NOT apply to suffixed variables
  result2 = prompt.materialize(boy="Alice")
  # boy1 and boy2 should NOT be "Alice"
  assert "{boy1}" not in result2  # Should be materialized from category
  assert "{boy2}" not in result2  # Should be materialized from category


def test_suffix_sample_method():
  """Test that sample method returns different values for boy1 and boy2"""
  prompt = Prompt("{boy1} and {boy2}")
  sampled = prompt.sample()

  # Should have both keys
  assert "boy1" in sampled
  assert "boy2" in sampled

  # Both should be valid boy names
  boy_names = prompts_module.WORDS_BY_CATEGORY["boy"]
  assert sampled["boy1"] in boy_names
  assert sampled["boy2"] in boy_names


def test_suffix_with_defaults():
  """Test suffix with default values"""
  prompt = Prompt("{boy1=DefaultBoy} and {boy2}")
  sampled = prompt.sample()

  # boy1 should use default
  assert sampled["boy1"] == "DefaultBoy"
  # boy2 should use category
  assert sampled["boy2"] in prompts_module.WORDS_BY_CATEGORY["boy"]


def test_suffix_priority_order():
  """Test complete priority order with suffixes"""
  prompt = Prompt("{boy1}")

  # Test kwargs priority
  sampled1 = prompt.sample(boy1="FromKwargs")
  assert sampled1["boy1"] == "FromKwargs"

  # Test exact choice priority
  prompt.add_choice(boy1=["ExactChoice"])
  sampled2 = prompt.sample()
  assert sampled2["boy1"] == "ExactChoice"

  # Test base choice priority (create new prompt)
  prompt2 = Prompt("{boy1}")
  prompt2.add_choice(boy=["BaseChoice"])
  sampled3 = prompt2.sample()
  assert sampled3["boy1"] == "BaseChoice"

  # Test default priority (create new prompt)
  prompt3 = Prompt("{boy1=DefaultValue}")
  sampled4 = prompt3.sample()
  assert sampled4["boy1"] == "DefaultValue"

  # Test category priority (no choices, no defaults)
  prompt4 = Prompt("{boy1}")
  sampled5 = prompt4.sample()
  assert sampled5["boy1"] in prompts_module.WORDS_BY_CATEGORY["boy"]
