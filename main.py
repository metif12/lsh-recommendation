import numpy as np
import pandas as pd
import re
from datasketch import MinHash, MinHashLSHForest

def preprocess(text):
    text = re.sub(r'[^\w\s]','',str(text))
    tokens = text.lower()
    tokens = tokens.split()
    return tokens


def get_forest(db, perms):    
    minhash = []
    
    for text in db:
        tokens = preprocess(text)
        m = MinHash(num_perm=perms)
        for s in tokens:
            m.update(s.encode('utf8'))
        minhash.append(m)
        
    forest = MinHashLSHForest(num_perm=perms)
    
    for i,m in enumerate(minhash):
        forest.add(i,m)
        
    forest.index()
        
    return forest

def predict(text, database, perms, num_results, forest):    
    tokens = preprocess(text)
    m = MinHash(num_perm=perms)
    for s in tokens:
        m.update(s.encode('utf8'))
        
    idx_array = np.array(forest.query(m, num_results))
    if len(idx_array) == 0:
        return None # if your query is empty, return none
    
    return database
        
if __name__ == '__main__':
    #Number of Permutations
    permutations = 128

    #Number of Recommendations to return
    num_recommendations = 1
        
    db = pd.read_csv('movies_metadata.csv', dtype=str)
    db = db['overview'].unique()
    query = 'Neil McCauley leads a top-notch crew on various insane heists throughout Los Angeles'
    
    forest = get_forest(db, permutations)
    
    result = predict(query, db, permutations, num_recommendations, forest)
    print('\n Query is \n', query)
    print('\n Top Recommendation(s) is(are) \n', result)