import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        ''' At the begining best_score and best_model '''
        best_score= float("inf")
        best_model= None

        ''' Get number of features '''
        n_features= len(self.X[0])
        ''' Total num of points = summation of all the sequences '''
        N =  np.sum(self.lengths)
        logN= np.log(N)

        for compo_num in range(self.min_n_components, self.max_n_components + 1):
            try:
                ''' Calculating the GaussianHMM model and loglikelihood.'''
                new_model= self.base_model(compo_num) 
                logL= new_model.score(self.X, self.lengths)

                ''' Calculate parameters and BIC'''
                p= (compo_num**2) + 2 * n_features * compo_num - 1
                BIC= -2 * logL + p * logN

                '''Updating the score and model according to generated values '''    
                if BIC < best_score:
                    best_score = BIC
                    best_model = new_model

            except:
                pass

        return best_model
                

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        ''' At the begining best_score and best_state '''
        best_score = float("-inf")
        best_model = None    
        
        for compo_num in range(self.min_n_components, self.max_n_components + 1):
            words_left_scores= []

            try:
                ''' Calculating log(P(X(i))) '''
                new_model= self.base_model(compo_num)
                logL= new_model.score(self.X, self.lengths) 
                                       
                for word in self.words:
                    if word !=self.this_word:
                        '''hwords is a dict with values of X and length for each key (word)'''
                        X_new, lengths_new= self.hwords[word] 
                        try:
                            ''' log(P(X(all but i)'''
                            words_left_scores.append(new_model.score(X_new, lengths_new)) 
                        except:
                            pass
                        
                '''Calculating the average of all other words'''
                if words_left_scores:
                    words_left_average= np.mean(words_left_scores)
                else:
                    words_left_average=0

                ''' Discriminative Information Criterion, 
                    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i)) '''
                DIC= logL - words_left_average
                if DIC > best_score:
                    best_score= DIC
                    best_model= new_model
            except:
                pass
        return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''
    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        
        ''' At the begining best_score and best_state '''
        best_score = float("-inf")
        best_state = None
        
        '''Number of folds. Must be at least 2. If only 1 sequence provided, then
           that full dataset is used for both train and test'''
        n_splits=min(len(self.sequences),3)
        if n_splits>1:
            split_method = KFold(n_splits=n_splits)
        
        
        ''' Loop over number of hidden states '''
        for compo_num in range(self.min_n_components, self.max_n_components + 1):
            if len(self.sequences) == 1:
                try :
                    hmm_model = GaussianHMM(n_components=compo_num, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    new_score = hmm_model.score(self.X, self.lengths)
                except:
                    continue
            else:
                cv_score = []
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    trained_model = None
                    try:
                        '''Combining the train set with the sequence '''
                        X_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)
                        '''Create the HMM model using training data set '''
                        trained_model = GaussianHMM(n_components=compo_num, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        '''Combining the test set with the sequence '''
                        X_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                        '''Calculate the score for the new model'''
                        state_score = trained_model.score(X_test, lengths_test)
                        cv_score.append(state_score)
                    except:
                        continue
                
                if cv_score:
                    new_score = np.mean(cv_score)
                else:
                    new_score=0
                    
                
            if new_score > best_score:
                best_score = new_score
                best_state = compo_num
                
        if best_state == 0:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_state) 
                

