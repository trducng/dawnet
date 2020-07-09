import numpy as np
import torch


def get_flipped_predictions(model_1, model_2, dataloader, cuda):
    """Get inconsistent predictions

    # Args
        model_1 [Model]: the model 1
        model_2 [Model]: the model 2
        dataloader [Dataloader]: the dataloader, should not be shuffled dataloader
        cuda [bool]: whether to load data to cuda

    # Returns
        [2D-tensor of shape N x 3]: ground truth, model_1 pred, model_2 pred
    """
    model_1.eval()
    model_2.eval()

    result = []
    with torch.no_grad():
        for inputs, outputs in dataloader:
            inputs, outputs = inputs.cuda(), outputs.cuda()
            preds_1 = model_1(inputs)
            preds_2 = model_2(inputs)

            _, preds_1 = torch.max(preds_1, 1)
            _, preds_2 = torch.max(preds_2, 1)

            result.append(
                np.vstack([
                    outputs.cpu().data.numpy(),
                    preds_1.cpu().data.numpy(),
                    preds_2.cpu().data.numpy(),
                ]).T
            )

    return np.concatenate(result, axis=0)


def analyze_flipped_prediction(preds):
    """Analyze the flipped prediction

    # Args
        preds [2D-tensor]: shape [N x 3] where col 1, 2, 3 are gt, model1 and model 2
            predictions

    # Returns
        <[int]>: list of indices that both model are correct
        <[int]>: list of indices that model 1 is correct but model 2 is not
        <[int]>: list of indices that model 2 is correct and model 1 is not
        <[int]>: list of indices that no model is correct
        <[int]>: list of flipped labels
    """
    result = (preds[:,1:] == np.expand_dims(preds[:,0], axis=1)).astype(np.uint8)

    # indices where all are correct
    models = np.all(result, axis=1)
    models = list(models.nonzero()[0])

    # indices where only model 1 are correct
    model_1 = (result[:,0] - result[:,1]) == 1
    model_1 = list(model_1.nonzero()[0])

    # indices where only model 2 are correct
    model_2 = (result[:,1] - result[:,0]) == 1
    model_2 = list(model_2.nonzero()[0])

    # indices where no model is correct
    no_model = np.all(1 - result, axis=1)
    no_model = list(no_model.nonzero()[0])

    # indices where labels are flipped
    flipped = preds[:,1] != preds[:,2]
    flipped = list(flipped.nonzero()[0])

    return models, model_1, model_2, no_model, flipped
