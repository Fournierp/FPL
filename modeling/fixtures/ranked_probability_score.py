import numpy as np


def ranked_probability_score(pdf_prediction, outcome):
    """ Compute the accuracy metric of the model againts fixture results

    Args:
        pdf_prediction (array): Inferred probabilities of match results
        outcome (array): Realized outcome of match results

    Returns:
        (int): Ranked Probability Score for the sample predicted
    """
    pdf_outcome = np.zeros_like(pdf_prediction)
    pdf_outcome[outcome] = 1

    cdf_outcome = np.cumsum(pdf_outcome)
    cdf_prediction = np.cumsum(pdf_prediction)

    rps = 0
    for i in range(len(cdf_prediction)):
        rps += (cdf_prediction[i] - cdf_outcome[i]) ** 2

    return rps / (len(cdf_prediction) - 1)


def match_outcome(df):
    """ Find the winner of the game

    Args:
        df (pd.DataFrame): Fixtures

    Returns:
        (pd.DataFrame): Column with the winner
    """
    return np.select(
        [
            df["score1"] > df["score2"],
            df["score1"] == df["score2"],
            df["score1"] < df["score2"],
        ],
        [
            0,
            1,
            2,
        ],
        default=1,
    )
