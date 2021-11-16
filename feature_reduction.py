import pandas as pd
import statsmodels.api as sm

class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05):

        initial = data.columns.tolist()
        forward_list = []
        while len(initial) > 0:
            res = list(set(initial) - set(forward_list))
            new = pd.Series(index=res)
            for col in res: 
                model = sm.OLS(target, sm.add_constant(data[forward_list + [col]])).fit()
                new[col] = model.pvalues[col]
            min_val = new.min()
            if (min_val < significance_level): 
                forward_list.append(new.idxmin())
            else: 
                break
        return forward_list

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): 
        backward_list = data.columns.tolist()
        while len(backward_list) > 0: 
            feats = sm.add_constant(data[backward_list])
            p_val = sm.OLS(target, feats).fit().pvalues[1:]
            max = p_val.max()
            if (max >= significance_level): 
                res = p_val.idxmax()
                backward_list.remove(res)
            else: 
                break
        return backward_list
