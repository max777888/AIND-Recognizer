import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    ''' Iterate over sequences to get X and length of X '''
    for index, sequence in test_set.get_all_Xlengths().items():
        X_test, Xlength_test = test_set.get_item_Xlengths(index)
        ''' Dictionary to store word and score '''
        words_score = dict()
        for word, model in models.items():
            try: 
                ''' Get the new score '''
                words_score[word] = model.score(X_test, Xlength_test)
            except:
                words_score[word]= float('-inf')
                       
        probabilities.append(words_score)
    
    for probs in probabilities:
        guesses.append(max(probs, key=probs.get))

    return probabilities,guesses