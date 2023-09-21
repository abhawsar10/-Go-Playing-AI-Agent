import json
import numpy as np
from copy import deepcopy


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

   












REWARD_WIN = 1
REWARD_LOSE = 0

class myQlearner:
    
    def __init__(self, alpha=0.7, gamma=0.9, init_value=0.5, side= None):
        
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.alpha=alpha
        self.gamma=gamma
        self.init_value=init_value
        self.side=side
        self.Qtable= {}
        self.history_states=[]
        
        with open("Qtable.json","r") as f:
            self.Qtable=json.load(f)
            
        
        with open("history_states.json","r") as f:
            self.history_states=json.load(f)
            
        
    def Qvals(self, state):
        
        if state not in self.Qtable:
            Qs=np.zeros((5,5))
            Qs.fill(self.init_value)            
            self.Qtable[state]=Qs.tolist()
        return self.Qtable[state]
      
                
      
    def maxQ(self, q_values):
        
        max = -np.inf
        row = 0
        col = 0
        
        for i in range(0,5):
            for j in range(0,5):
                if q_values[i][j]>max:
                    max=q_values[i][j]
                    row=i
                    col=j
        
        return row,col
        
       
                
    def encode_board(self, mat):
        return ''.join([str(mat[i][j]) for i in range(5) for j in range(5)])

    
    def get_input(self, go):
        
        
        state_string = self.encode_board(go.board)
        q_values_state=np.zeros((5,5))
        q_values_state = self.Qvals(state_string)
        
        i=0
        
        for x in range(0,24):
            i,j=self.maxQ(q_values_state)
            
            if go.place_n_check(i, j):
                self.history_states.append([state_string,[i,j]])
                self.write_history_table()                
                return i,j
            else:
                q_values_state[i][j] = -1.0
                
                self.Qtable[state_string] = q_values_state
                
        return -1,-1
        
        
    def write_Q_table(self):
        with open("Qtable.json","w") as f:
            json.dump(self.Qtable,f)
            
            
    def write_history_table(self):
        with open("history_states.json","w") as f:
            json.dump(self.history_states,f)   
        
        
    def learn(self, reward):
        
        
        self.history_states.reverse()
        
        max_q_value = -1.0
        
        for hist in self.history_states:
            state, move = hist
            q = self.Qvals(state)
            
            
            
            if max_q_value < 0:
                q[move[0]][move[1]] = reward
            else:
                q[move[0]][move[1]] = q[move[0]][move[1]] * (1 - self.alpha) + self.alpha * self.gamma * max_q_value
            
            self.Qtable[state]=q
            max_q_value = np.max(q)
        
        player.write_Q_table()
        player.history_states = []
        player.write_history_table()
        
    def beginlearn(self,go):
        result = go.judge_winner()
        
        if self.side==result:
            reward = REWARD_WIN
        else :
            reward = REWARD_LOSE
        
        print("learning...")
        self.learn(reward)
        
        
        
if __name__ == "__main__":
    
    LEARN_NOW = False
    N = 5
    io=IO()
    piece_type, previous_board, board = io.readip(N)
    
    my_go_board = GO_BOARD(N)
    my_go_board.set_go_board(piece_type, previous_board, board)
    player = myQlearner(side=piece_type)
    
    action = player.get_input(my_go_board)
    player.write_Q_table()
    
    action_type="MOVE"
    if action[0] == -1 and action[1]==-1:
        action_type="PASS"
    
    if my_go_board.check_two_passes(action_type):
        LEARN_NOW = True
    
    #if my_go_board.game_end(piece_type, action_type): 
     #   LEARN_NOW = True
        
    if LEARN_NOW == True:
        player.beginlearn(my_go_board)
            
    if(action[0] == -1 and action[1]==-1):
        io.writePass()
    io.writeop(action)   
    
             

            
            
            
            
            
            
            
            
            
            
            
            