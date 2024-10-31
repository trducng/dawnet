from dawnet.inspector import Inspector

from sample import GPT2

model = GPT2()


def test_inspector_share_model_parameter():
    inspector = Inspector(model)
    assert id(inspector._original_model) == id(model)
    assert id(inspector._model) != id(model)
    assert id(inspector._model.attention.query.weight) == id(
        model.attention.query.weight
    )


def test_inspector_share_model_buffer():
    inspector = Inspector(model)
    assert id(inspector._original_model) == id(model)
    assert id(inspector._model) != id(model)
    assert id(inspector._model.batch_norm.running_mean) == id(
        model.batch_norm.running_mean
    )


def test_copy_inspector():
    inspector1 = Inspector(model)
    inspector2 = inspector1.copy()

    assert id(inspector1._original_model) == id(inspector2._original_model)
    assert id(inspector1._model) != id(inspector2._model)
    assert id(inspector1._model.attention.query.weight) == id(
        inspector2._model.attention.query.weight
    )
    assert id(inspector1._model.batch_norm.running_mean) == id(inspector2._model.batch_norm.running_mean)


def test_disable_op():
    pass


def test_reenable_op():
    pass


def test_remove_op():
    pass
