from dawnet.inspector import Inspector, Handler

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

    handlers1 = inspector1.get_submodule("_model.attention.query")._forward_hooks
    handlers2 = inspector2.get_submodule("_model.attention.query")._forward_hooks
    assert len(handlers1) == 1
    assert len(handlers2) == 1

    handler1 = list(handlers1.values())[0]
    handler2 = list(handlers2.values())[0]
    assert isinstance(handler1, Handler)
    assert isinstance(handler2, Handler)
    assert id(handler1._inspector) == id(inspector1)
    assert id(handler2._inspector) == id(inspector2)

def test_disable_op():
    pass


def test_reenable_op():
    pass


def test_remove_op():
    pass
