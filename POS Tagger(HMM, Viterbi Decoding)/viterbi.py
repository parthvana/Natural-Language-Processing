import numpy as np

def run_viterbi(emission_scores, trans_scores, start_scores, end_scores):
    """Run the Viterbi algorithm.

    N - number of tokens (length of sentence)
    L - number of labels

    As an input, you are given:
    - Emission scores, as an NxL array
    - Transition scores (Yp -> Yc), as an LxL array
    - Start transition scores (S -> Y), as an Lx1 array
    - End transition scores (Y -> E), as an Lx1 array

    You have to return a tuple (s,y), where:
    - s is the score of the best sequence
    - y is the size N array/seq of integers representing the best sequence.
    """
    L = start_scores.shape[0]
    assert end_scores.shape[0] == L
    assert trans_scores.shape[0] == L
    assert trans_scores.shape[1] == L
    assert emission_scores.shape[1] == L
    N = emission_scores.shape[0]

   
    list_sequence = []
    ops=[0]*L
    sequence=np.zeros((L,N+1))
    delta=np.zeros((L,N+1))
    for i in range(L):
        for j in range(N+1):
            sequence[i][j]=-100000
            delta[i][j]=-100000

    emission_scores = emission_scores.T
    
    for k in range(L):  # 1st Column
        delta[k][0] = start_scores[k] + emission_scores[k][0]
        
#    for k in range(L):
#        print(start_scores[k])

    for m in range(1, N):  # From column 1 to N
        for n in range(L):
            temp_state = - 100000
            temp_max = -100000
            for o in range(L):
                temp = trans_scores[o][n] + delta[o][m - 1]
                if temp > temp_max:                    
                    temp_state = o
                    temp_max = temp
            sequence[n][m] = temp_state
            delta[n][m] = emission_scores[n][m] + temp_max

    #print(delta[n][m])
    #print(sequence[n][m])
    
    
    for i in range(L):  #(N+1)th column
        sequence[i][N]=i
        delta[i][N] = end_scores[i] + delta[i][N - 1]
        
#        emission_scores = emission_scores(0.0, emission_var, (N,L))
#        trans_scores = trans_scores(0.0, trans_var, (L,L))
#        start_scores = start_scores(0.0, trans_var, L)
#        end_scores = end_scores(0.0, trans_var, L)
             
  
    max_row = int(np.argmax(delta,axis=0)[-1]) 
    max_score = delta[max_row][N]    
    
    #print(max_score)
    #print(max_row)
    
    # Have to convert seqence values to int otherwise will create problem in subsequent use
    for p in range(N, 0, -1): # Reverse traversal to find best sequence
        list_sequence.append(int(sequence[max_row][p]))
        max_row = int(sequence[max_row][p])

    list_sequence.reverse()
   
    return (max_score, list_sequence)
