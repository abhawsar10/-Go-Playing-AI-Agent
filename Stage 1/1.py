

def find_chunks(self,go):
    
    count = 0
    for i in range(5):
        for j in range(5):
            n_neibors = len(go.find_ally_neighbors(i,j))
            if n_neibors>1:
                count+=1
                
    return count