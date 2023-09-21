
from datetime import datetime
startTime = datetime.now()

from copy import *
import json
import numpy as np
from copy import deepcopy
import argparse
import sys

class IO:
   
    def readip(self,n):
        
        with open("input.txt","r") as f:
            lines=f.readlines()
            
            p_type = int(lines[0])
            previous_board = [[int(x) for x in line.rstrip('\n')] for line in lines[1:n+1]]
            board = [[int(x) for x in line.rstrip('\n')] for line in lines[n+1: 2*n+1]]
    
        return p_type, previous_board, board
    
    
    def writeop(self,result):
        
        res = ""+str(result[0]) + ',' + str(result[1])
        with open("output.txt", 'w') as f:
            f.write(res)
    
    def writePass(self):
    	with open("output.txt", 'w') as f:
    		f.write("PASS")

class GO_BOARD:
    
    def __init__(self,n):
        
        self.board_size = n
        self.died_pieces = []
        self.komi = n/2
        
        
    def set_go_board(self, piece_type, previous_board, board):
        
        self.piece_type = piece_type
        self.previous_board = previous_board
        self.board = board
    
    def boards_equal(self, board1, board2):
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board1[i][j]!=board2[i][j]:
                    return False
        return True
    
    def find_all_neighbors(self, i, j):
        
        board = self.board
        nei = []
        if i > 0: 
            nei.append((i-1, j))
        if i < self.board_size - 1: 
            nei.append((i+1, j))
        if j > 0: 
            nei.append((i, j-1))
        if j < self.board_size - 1: 
            nei.append((i, j+1))
        return nei
    
    def find_ally_neighbors(self, i, j):
        
        board = self.board
        all_nei = self.find_all_neighbors(i, j)  
        ally_nei = []
        
        for piece in all_nei:
            
            if board[piece[0]][piece[1]] == board[i][j]:
                ally_nei.append(piece)
        return ally_nei
    
    def find_allies(self, i, j):
        
        stack = [(i, j)]
        allies = [] 
        
        while stack:
            piece = stack.pop()
            allies.append(piece)
            
            ally_nei = self.find_ally_neighbors(piece[0], piece[1])
            for ally in ally_nei:
                if ally not in stack and ally not in allies:
                    stack.append(ally)
        return allies
    

    def has_liberty(self, i, j):
        
        board = self.board
        allies = self.find_allies(i, j)
        
        for member in allies:
            #iterate over ally neighbors of i,j
            neighbors = self.find_all_neighbors(member[0], member[1])
            for piece in neighbors:
                #iterate over all neighbors of neighbor
                if board[piece[0]][piece[1]] == 0:
                    return True
        
        return False
    
    def find_in_danger_pieces(self, p_type):
        
        board = self.board
        dying_pieces = []

        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == p_type:
                    if not self.has_liberty(i, j):
                        dying_pieces.append((i,j))
        return dying_pieces

    def kill_pieces(self, p_type):
        
        dying_pieces = self.find_in_danger_pieces(p_type)
        
        if dying_pieces: 
            
            board = self.board
            for piece in dying_pieces:
                board[piece[0]][piece[1]] = 0
            self.board=board
    
        return dying_pieces

    def score(self, p_type):
        
        board = self.board
        score = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == p_type:
                    score += 1
        return score          

    def judge_winner(self):
        
        sc_1 = self.score(1)
        sc_2 = self.score(2)
        if sc_1 > sc_2 + self.komi: 
            return 1
        elif sc_1 < sc_2 + self.komi: 
            return 2
        else: 
            return 0


    def place_n_check(self, i, j,prints= False):
       
       
        piece_type = self.piece_type
        board = self.board
        
        

        if not (i >= 0 and i < self.board_size):
            if prints:
                print("Invalid Row Number")
            return False
        if not (j >= 0 and j < self.board_size):
            if prints:
                print("Invalid Row Number")
            return False
        
        if board[i][j] != 0:
            if prints:
                print("Position is full")
            return False
        
        test_go = deepcopy(self)
        test_board = test_go.board

        test_board[i][j] = piece_type
        test_go.board = test_board
        if test_go.has_liberty(i, j):
            return True

        
        test_go.kill_pieces(3 - piece_type)
        
        if not test_go.has_liberty(i, j):
            if prints:
                print('Invalid, No liberty at this position.')
            return False

        else:
            if self.died_pieces and self.boards_equal(self.previous_board, test_go.board):
                if prints:
                    print('Invalid placement by KO Rule')
                return False
        return True




    def check_two_passes(self, action="MOVE"):
                
        # return true if 2 PASS turns are going to happen
        if self.boards_equal(self.previous_board, self.board) and action == "PASS":
            return True
        return False

   

xxx=0






class myQlearner:
    
    def __init__(self, alpha=0.7, gamma=0.9, init_value=0.5, side= None):
        
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.alpha=alpha
        self.gamma=gamma
        self.init_value=init_value
        self.side=side
        self.Qtable= {}
        self.Utable= {}
        self.history_states=[]
        
        
        with open("utility_table.json","r") as f:
            self.Utable=json.load(f)
         
        with open("utility_opponent_table.json","r") as f:
            self.UOtable=json.load(f)    
    
    
    
    def decode_board(self, str):
        
        a=np.zeros((5,5),dtype =np.int)
        i,j = 0,0
        for c in str:
            if j>4: 
                j=0
                i+=1
            a[i][j]=int(c) 
            j+=1
        return a.tolist()

    
    
    def encode_board(self, mat):
        return ''.join([str(mat[i][j]) for i in range(5) for j in range(5)])

    def get_input(self, my_go_board):
         
        move=my_go_board.score(1)+my_go_board.score(2)
        print(move)
        """
        action = self.alpha_beta(my_go_board,1)
        """
        if move <=3:
            action = self.alpha_beta(my_go_board,3)
        elif move>3 and move<=10:
            action = self.alpha_beta(my_go_board,3)
        elif move>10 and move<=14:
            action = self.alpha_beta(my_go_board,5)
        elif move == 15 or move == 16:
            action = self.alpha_beta(my_go_board,5)
        elif move == 17:
            action = self.alpha_beta(my_go_board,7)
        elif move == 18 or move == 19:
            action = self.alpha_beta(my_go_board,5)
        elif move == 20 or move == 21:
            action = self.alpha_beta(my_go_board,3)
        elif move == 22 or move == 23:
            action = self.alpha_beta(my_go_board,1)
        elif move == 24:
            return -1,-1
        
        print(action)
        return action[0],action[1]
       
    
    
    def alpha_beta(self, go, lookahead ):
        
        ply=lookahead
        max_len=0
        v,k,l = self.max_value(go, -np.inf, np.inf, 0, 0, max_len, ply)
        self.write_U_table()
        
        return v,k,l
        
    def max_value(self, go, a, b, m, n, max_len, ply):
        
        if max_len>=ply:
            return m,n,self.state_utility(go.board, m, n, for_opponent= True )
        
        v = -np.inf
        
        safe_copy = deepcopy(go.board)
        go_max = GO_BOARD(5)
        go_max.board = copy(go.board)
        go_max.piece_type = self.side
        
        x,y=-1,-1
        max_len+=1
        for i in range(5):
            for j in range(5):
                if go_max.place_n_check(i, j):
                    
                    go_max.board[i][j] = copy(go_max.piece_type)
                    go_max.kill_pieces(3 - go_max.piece_type)
                    #print(i,j,"\n",np.array(go_max.board))
                    
                    
                    temp = max(   v  ,  self.min_value(go_max, a, b, i, j, max_len,ply)[2]  )                   
                    
                    #print("x",temp)
                    if temp > v:
                        v=temp
                        x,y=i,j
                    
                    go_max.board = deepcopy(safe_copy)
                
                    if v >= b:
                        return i,j,v
                    
                    a = max( a , v )
                    
        del go_max          
        return x,y,v
         
    def min_value(self, go, a, b, i, j, max_len,ply):
        
        if max_len>=ply:
            return i,j,self.state_utility(go.board, i, j,show_eval=True)
        
        v = +np.inf
        
        safe_copy = deepcopy(go.board)
        go_min = GO_BOARD(5)
        go_min.board = copy(go.board)
        go_min.piece_type = 3- self.side
        
        x,y=-1,-1
        max_len+=1
        for m in range(5):
            for n in range(5):
                if go_min.place_n_check(m, n):
                    
                    go_min.board[m][n] = copy(go_min.piece_type)
                    go_min.kill_pieces(3 - go_min.piece_type)
                    #print(m,n,"\n",np.array(go_min.board))
                    temp = min(   v  ,   self.max_value(go_min, a, b, m, n, max_len,ply)[2]   )
                    
                    if temp < v:
                        v=temp
                        x,y=m,n
                        #print("[2,",v,"],")
                    
                    go_min.board = deepcopy(safe_copy)
                    
                    if v <= a:
                        return m,n,v
                    
                    b = min( b , v )
        
        del go_min        
        return x,y,v
             
    def search_tables(self, board, foropp=False):
        
        diff_boards = []
        
        diff_boards.append(self.encode_board(board))
        
        diff_boards.append(self.encode_board(np.rot90(board).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(board,2).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(board,3).tolist()))
        
        diff_boards.append(self.encode_board(np.flipud(board).tolist()))
        
        diff_boards.append(self.encode_board(np.fliplr(board).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(np.fliplr(board))))
        
        diff_boards.append(self.encode_board(np.rot90(np.fliplr(board),3)))
        
        
        if not foropp:
            
            for i  in range(len(diff_boards)):
                
                i_board=diff_boards[i]
                
                if i_board in self.Utable:
                    return 1,self.Utable[i_board]
           
            return 0,board
        
        else:
            
            for i  in range(len(diff_boards)):
                
                i_board=diff_boards[i]
                
                if i_board in self.UOtable:
                    return 1,self.UOtable[i_board]
           
            return 0,board
            
     
    def state_utility(self, state, i, j, for_opponent = False, show_eval = False):
        
        #find utility of state where last stone placed  at i,j
        
        if not for_opponent:
            found,state_trans = self.search_tables(state)
            if found == 1:
                return state_trans
        else:
            found,state_trans = self.search_tables(state, foropp=True)
            if found == 1:
                return state_trans    
        
        
        go_trial = GO_BOARD(5)
        go_trial.board = copy(state)
        go_trial.piece_type = copy(self.side)
        if for_opponent:
             go_trial.piece_type = copy(3- self.side)
             
       
        
        e = self.find_eyes(go_trial)
        L = self.total_liberties(go_trial)
        x = self.no_of_my_stones(go_trial)
        y = self.no_of_opo_stones(go_trial)
        d = self.find_chunks(go_trial)
        val1,val2 = self.stone_pos_reward(go_trial)
        val=(2*val1+val2)
        d1,d2,d34,d5,d6 = 3,1,np.maximum(np.minimum((x+y)-10,10),1),3,np.maximum((10-(x+y))/2,0)
        
        
        if self.side == 1:
            y=y+2.5
        else:
            x=x+2.5
        
        eval_func = d1*e + d2*L + d34*(x - y) + d5*d + d6*val 
        if for_opponent:
            eval_func = -eval_func
                
               
        if show_eval:
            print(np.array(go_trial.board))
            print()
            del go_trial
            print("Newly Placed Point = "+str(i)+","+str(j))
            print()
            print("="*40) 
            print("Total Eyes         = "+str(e)+"  x"+str(d1)+"    = "+str(d1*e))
            print()
            print("Total Liberties    = "+str(L)+"  x"+str(d2)+"     = "+str(d2*L))
            print()
            print("My Stones          = "+str(x)+"  x"+str(d34)+"   = "+str(d34*x))
            print()
            print("Opponents Stones   = "+str(y)+"  x"+str(d34)+"   = "+str(-d34*y))
            print()
            print("Close Chunks       = "+str(d)+"  x"+str(d5)+"   = "+str(d5*d))
            print()
            print("Reward of Position = "+str(val)+"  x"+str(d6)+"    = "+str(d6*val))
            print("="*40)      
            print("Eval = "+str(eval_func))
            print("="*40) 
            
        state_str = self.encode_board(state)
        if for_opponent:
            self.UOtable[state_str]=eval_func
            return self.UOtable[state_str]
        else:
            self.Utable[state_str]=eval_func
            return self.Utable[state_str]
                
          
        
        
         
    def write_U_table(self):
        with open("utility_table.json","w") as f:
            json.dump(self.Utable,f)  
        with open("utility_opponent_table.json","w") as f:
            json.dump(self.UOtable,f)
            
            
    
   #Heuristic Functions:
    
          
    def no_of_my_stones(self, go):
         
         piece = go.piece_type
         boardarr = np.array(go.board) 
         return np.count_nonzero(boardarr == piece) 
     
    def no_of_opo_stones(self, go):
         
         piece = 3- go.piece_type
         boardarr = np.array(go.board) 
         return np.count_nonzero(boardarr == piece)    
    
    def total_liberties(self, go):
        
        
        side = go.piece_type
        liberties = []
        
        for i in range(5):
            for j in range(5):
                if go.board[i][j] == side:
                    neibs = go.find_all_neighbors(i,j)
                    for n in neibs:
                        if go.board[n[0]][n[1]] == 0:
                            liberties.append((n[0],n[1]))
                            
        libs = list(dict.fromkeys(liberties))
        return len(libs)
    
    
    def find_chunks(self,go):
    
        count = 0
        for i in range(5):
            for j in range(5):
                x=0
                if go.board[i][j] == go.piece_type:
                    x=go.find_ally_neighbors(i,j)
                    print("hello",x)
                    n_neibors = len(x)
                    if n_neibors>1:
                        print("bye")
                        count = count+1
                    
        return count    

    
    def stone_pos_reward(self, go):
        
        cent_points = [(2,2)]
        points = [(1,1),(1,2),(1,3),(2,1),(2,3),(3,1),(3,2),(3,3)]   
        
        n1= self.allies_in_range(go,cent_points)
        
        n2= self.allies_in_range(go,points)
        
        return n1,n2
        
        
    def allies_in_range(self, go, points):
        
        c=0
        side = go.piece_type
        for point in points:
            if go.board[point[0]][point[1]] == side:
                c+=1
        return c
    
    
    def manhattan_points(self, x, y, dist):
        
        points=[]
        for i in range(5):
            for j in range(5):
                diff = abs(i-x) + abs(j-y) 
                if diff <= dist and diff!=0:
                    points.append((i,j))
        
        return points
                    
    def find_eyes(self, go):

        
        c_eyes=0
        for i in range(go.board_size):
            for j in range(go.board_size):            
                if go.board[i][j] == 0:
                    found = True
                    neibs=go.find_all_neighbors(i,j)
                    
                    for n in neibs:
                        if go.board[n[0]][n[1]] == 0:
                            neibs.remove(n)
                            found = False
                            break
                        
                    if found == False:
                        continue
                    
                    piece = go.board[neibs[0][0]][neibs[0][1]]
                    found
                    for n in neibs:
                        if piece != go.board[n[0]][n[1]]:
                            found = False
                            break
                    if found == True:
                        c_eyes+=1
        
        return c_eyes
    
    
    
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-learn", type=int, help="Learn values", default=0)
    parser.add_argument("-win", type=int, help="Did you win", default=0)
    args=parser.parse_args()
    
    LEARN_NOW = args.learn
    winner = args.win
    
    N = 5
    io=IO()
    piece_type, previous_board, board = io.readip(N)
    my_go_board = GO_BOARD(N)
    my_go_board.set_go_board(piece_type, previous_board, board)
    player = myQlearner(side=piece_type)
	    
    if LEARN_NOW:
            player.learn(winner)
    
    else:
        
	    action = player.get_input(my_go_board)
	    io.writeop(action) 	    
	    if action[0] == -1 and action[1]==-1:
	    	io.writePass()
    
    
    print(datetime.now() - startTime)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
