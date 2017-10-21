
import numpy as np

def gaussian(x, mu, sigsq):
    return (((1/(np.power((2*(np.pi)*sigsq), 0.5)))) * (np.exp(-np.power((x - mu), 2) / (2 * (sigsq)))))


def normalmapping(tfidf, colnames):

    mews  = {}
    sigmas = {}

    for count in range(len(colnames)):
        col = tfidf[:, count]
        mews.update({count:np.mean(col)})
        sigmas.update({count:(np.var(col, ddof = 1))})

    temp = np.zeros(tfidf.shape)

    for count in range(len(colnames)):
        col = tfidf[:, count]
        for count2 in range(len(col)):
            temp[count2,count] = gaussian(col[count2], mews[count], sigmas[count])

    return temp
