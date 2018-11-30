# Result confidence calculator
# @author: _john
# =============================================================================
import math
import numpy as np


def extend_with_blanks(sequence, blank_char):
    """Extends a sequence with blank token

    This method add blank character at the beginning and end of sequence, as
    well as in between each element in sequence.

    # Arguments
        sequence [str]: sequence of characters
        blank_char [str]: blank tokenizer
    
    # Returns
        [list of str]: list of elements expaned by blank character
    """
    result = [blank_char]
    for each_element in sequence:
        result.append(each_element)
        result.append(blank_char)
    return result


def label_to_index(sequence, classes):
    """Map a label to corresponding index

    # Arguments
        sequence [list of strs]: a sequence of labels
        classes [dict]: a mapping dictionary
    
    # Returns
        [list of ints]: list of corresponding indices
    """
    return [classes.index(each_element) for each_element in sequence]


def get_label_probability(timestep, char_idx, pred_logit, blanked_seq,
                          blank_idx, cache):
    """Compute the confidence of a sequence decoded from output logits

    This function recursively computes probability of labeling, save the results
    of sub-problems in cache to avoid re-calculation later.

    # Arguments
        timestep [int]: the timestep index to look for in `pred_logit`
        char_idx [int]: the current character index to look for in `blanked_seq`
        pred_logit [np array]: the logit output (shape timestep x chars)
        blanked_seq [list of ints]: the sequence in format of output indices
        blank_idx [int]: the index value of blank charachter
        cache [list of list of floats]: to cache to save partial calculations

    # Returns
        [float]: the probability of `blanked_seq` calculated from `pred_logit`
    """
    # check index of labeling
    if char_idx < 0:
        return 0.0

    # sub-problem already computed
    if cache[timestep][char_idx] != None:
        return cache[timestep][char_idx]

    # initial values
    if timestep == 0:
        if char_idx == 0:
            res = pred_logit[0, blank_idx]
        elif char_idx == 1:
            res = pred_logit[0, blanked_seq[1]]
        else:
            res = 0.0

        cache[timestep][char_idx] = res
        return res

    # recursve on s and t
    res = (
        get_label_probability(timestep-1, char_idx, pred_logit, blanked_seq,
            blank_idx, cache)
      + get_label_probability(timestep-1, char_idx-1, pred_logit, blanked_seq,
            blank_idx, cache)) * pred_logit[timestep, blanked_seq[blank_idx]]

    # in case of a blank or a repeated label, we only consider character
    # index and character index - 1 at timestep - 1, so we're done
    if (blanked_seq[char_idx] == blank_idx
        or (char_idx >= 2 and blanked_seq[char_idx-2] == blanked_seq[char_idx])):
        cache[timestep][char_idx] = res
        return res

    # otherwise, in case of a non-blank and non-repeated label, we additionally
    # add char_idx - 2 at timestep - 1
    res += get_label_probability(
        timestep-1, char_idx-2, pred_logit, blanked_seq,
        blank_idx, cache) * pred_logit[timestep, blanked_seq[char_idx]]
    cache[timestep][char_idx] = res

    return res


def empty_cache(timesteps, sequence):
    """Empty cache

    # Arguments
        timesteps [int]: the number of timesteps
        sequence [int]: the maximum number of characters in the sequence
    
    # Returns
        [list of list of None]: matrix of size timesteps x len(sequence)
    """
    return [[None for _ in range(len(sequence))] for _ in range(timesteps)]


def get_ctc_label_probability(pred_logit, pred_string, char_list):
    """Calculate label probability with CTC tokenization P(string|logit)

    # Arguments
        pred_logit [np array]: the logit output (shape timestep x chars)
        pred_string [str]: the string from logit
        char_list [str]: the string containing all output characters from model
            should correspond to the last dimension of `pred_logit`
    
    # Returns
        [float]: the confidence of `pred_string` calculated from `pred_logit`
    """
    n_timesteps, _ = pred_logit.shape                   # size of input matrix
    blank_idx = 0                                       # index of blank label
    blanked_seq = extend_with_blanks(
        label_to_index(pred_string, char_list),
        blank_idx)

    # cache subresults to avoid recalculating subproblems over and over again
    cache = empty_cache(n_timesteps, blanked_seq)
    
    # both sequences that include and exclude last token are valid
    return (
        get_label_probability(
            n_timesteps-1, len(blanked_seq)-1, pred_logit, blanked_seq,
            blank_idx, cache)
      + get_label_probability(
            n_timesteps-1, len(blanked_seq)-2, pred_logit, blanked_seq,
            blank_idx, cache))


def get_ctc_loss(pred_logit, label, char_list):
    """Calculate CTC loss

    # Arguments
        pred_logit [np array]: the logit output (shape timestep x chars)
        label [str]: the string from logit
        char_list [str]: the string containing all output characters from model
            should correspond to the last dimension of `pred_logit`
    
    # Returns
        [float]: the ctc loss between output and label
    """
    return -math.log(get_ctc_label_probability(pred_logit, label, char_list))


def get_char_positions(pred_logit, pred_string, char_list):
    """Return the indices of each character in `pred_string` in `pred_logit`

    # Arguments
        pred_logit [np array]: the logit output (shape timestep x chars)
        pred_string [str]: the string from logit
        char_list [str]: the string containing all output characters from model
            should correspond to the last dimension of `pred_logit`
    
    # Returns
        [list of tuples]: each tuple is the range of a character
    """    
    max_val = np.argmax(pred_logit, axis=1)
    greedy_string = ''
    for idx in range(len(max_val)):
        greedy_string += char_list[max_val[idx]]
    
    ranges = []
    start_range = 0
    stop_range = 1
    for each_char in pred_string:
        try:
            matched_idx = greedy_string.index(each_char)
            if (matched_idx != 0 and
                greedy_string[matched_idx-1] == char_list[0]):
                start_range = matched_idx - 1
            else:
                start_range = matched_idx
            
            stop_range = matched_idx + 1
            while True:
                if (stop_range < len(greedy_string) and
                    greedy_string[stop_range] == each_char):
                    stop_range += 1
                else:
                    break
            stop_range += 1

        except ValueError:
            start_range = stop_range
            stop_range = start_range + 1

        ranges.append((start_range, stop_range))
        greedy_string = char_list[0] * stop_range + greedy_string[stop_range:]
    
    return ranges


def get_char_confidence(pred_logit, pred_string, char_list):
    """Calculate character confidence

    # Arguments
        pred_logit [np array]: the logit output (shape timestep x chars)
        pred_string [str]: the string from logit
        char_list [str]: the string containing all output characters from model
            should correspond to the last dimension of `pred_logit`
    
    # Returns
        [list of floats]: each float is the confidence for each character in
            `pred_string`
    """
    positions = get_char_positions(pred_logit, pred_string, char_list)
    confidences = []
    for each_char, each_position in zip(pred_string, positions):
        start, stop = each_position
        confidence = get_ctc_label_probability(
            pred_logit[start:stop,:],
            each_char,
            char_list)
        confidences.append(confidence)
    
    return confidences
