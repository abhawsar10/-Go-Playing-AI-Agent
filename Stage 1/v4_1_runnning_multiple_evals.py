
from datetime import datetime
startTime = datetime.now()

from copy import *
import json
import numpy as np
from copy import deepcopy
import argparse
import sys
from random_player_training_setup import *

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
        self.X_move = True
        self.n_move = 0
        self.max_move = n * n - 1
        self.piece_type = None
        
    def init_board(self, n):
        
        board = [[0 for x in range(n)] for y in range(n)]  
        
        self.board = board
        self.previous_board = deepcopy(board)    
        
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


    def place_n_check(self, i, j,piece_type, prints= False):
       
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


    def game_end(self, action="MOVE"):
        
        if self.n_move >= self.max_move:
            return True
        if self.boards_equal(self.previous_board, self.board) and action == "PASS":
            return True
        return False
     
    def visualize_board(self):
        '''
        Visualize the board.

        :return: None
        '''
        board = self.board

        print('-' * len(board) * 2)
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 0:
                    print(' ', end=' ')
                elif board[i][j] == 1:
                    print('X', end=' ')
                else:
                    print('O', end=' ')
            print()
        print('-' * len(board) * 2)
    
    def place_chess(self, i, j, piece_type):
        
        board = self.board

        self.previous_board = deepcopy(board)
        board[i][j] = piece_type
        self.board=board
        
        return True
        
    def play(self, player1, player2, verbose):
        
        self.init_board(self.board_size)
        
        # Game starts!
        while 1:
            player_turn = 1 if self.X_move else 2

            # Judge if the game should end
            if self.game_end():       
                result = self.judge_winner()
                if verbose:
                    print('Game ended.')
                    if result == 0: 
                        print('The game is a tie.')
                    else: 
                        print('The winner is {}'.format('X' if result == 1 else 'O'))
                return result

            if verbose:
                player = "X" if player_turn == 1 else "O"
                print(player + " makes move...")

            # Game continues
            if player_turn == 1: 
            	action = player1.get_input(self, player1.side)
            else: 
            	action = player2.get_input(self, player2.side)

            if verbose:
                player = "X" if player_turn == 1 else "O"
                print(action)

            if action != "PASS":
                # If invalid input, continue the loop. Else it places a chess on the board.
                if not self.place_chess(action[0], action[1], player_turn):
                    if verbose:
                        self.visualize_board() 
                    continue

                self.died_pieces = self.kill_pieces(3 - player_turn) # Remove the dead pieces of opponent
            else:
                self.previous_board = deepcopy(self.board)

            if verbose:
                self.visualize_board() # Visualize the board again
                print()

            self.n_move += 1
            self.X_move = not self.X_move # Players take turn











class myQlearner:
    
    def __init__(self, alpha=0.7, gamma=0.9, init_value=0.5, side= None):
        
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")
        self.alpha=alpha
        self.gamma=gamma
        self.init_value=init_value
        self.side=side
        self.Qtable= {}
        self.type='qlearner'
        self.history_states=[]
        
        with open("utility_table.json","r") as f:
            self.Utable=json.load(f)
               
    
    def Qvals(self, state):
        
        if state not in self.Qtable:
            Qs=np.zeros((5,5))
            Qs.fill(self.init_value)            
            self.Qtable[state]=Qs.tolist()
        return self.Qtable[state]
      
    def find_q_table_state(self, curr_state):
        
        #make state such that player is always 1 and opponent is always 2
        #to reduce q table size
        
        if self.side ==2:
            curr_state=curr_state.replace("1","3")
            curr_state=curr_state.replace("2","1")
            curr_state=curr_state.replace("3","2")
        return curr_state
    
           
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

    def search_q_table(self, go):
        
        diff_boards = []
        
        diff_boards.append(self.encode_board(go.board))
        
        diff_boards.append(self.encode_board(np.rot90(go.board).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(go.board,2).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(go.board,3).tolist()))
        
        diff_boards.append(self.encode_board(np.flipud(go.board).tolist()))
        
        diff_boards.append(self.encode_board(np.fliplr(go.board).tolist()))
        
        diff_boards.append(self.encode_board(np.rot90(np.fliplr(go.board))))
        
        diff_boards.append(self.encode_board(np.rot90(np.fliplr(go.board),3)))
        
        for i  in range(len(diff_boards)):
            
            i_board=diff_boards[i]
            q_board = self.find_q_table_state(i_board)
            if q_board in self.Qtable:
                return i,q_board
       
        return 0,self.find_q_table_state(diff_boards[0])
            
    
    def reverse_transform(self, i, j, transformation):
                
        if transformation == 1:
            return j,4-i
        if transformation == 2:
            return 4-i,4-j
        if transformation == 3:
            return 4-j,i
        if transformation == 4:
            return 4-i,j
        if transformation == 5:
            return i,4-j
        if transformation == 6:
            return j,i
        if transformation == 7:
            return 4-j,4-i
    
    
    
    def state_eval(self, state, i, j, for_opponent = False, show_eval = False):
        
        #find utility of state if point is placed at i,j
        
        go_trial = GO_BOARD(5)
        go_trial.board = copy(state)
        go_trial.piece_type = copy(self.side)
        if for_opponent:
             go_trial = copy(3- self.side)
        
        
        if go_trial.place_n_check(i, j, go_trial.piece_type):
        
            go_trial.board[i][j] = copy(self.side)
            if for_opponent:
                go_trial.board[i][j] = copy(3- self.side)
            
            go_trial.kill_pieces(3 - go_trial.piece_type)
            
            e = self.find_eyes(go_trial)
            
            L = self.total_liberties(go_trial)
            
            x = self.no_of_my_stones(go_trial)
            y = self.no_of_opo_stones(go_trial)
            
            val = self.stone_pos_reward(i,j)
            
            d = self.allies_in_range(go_trial,self.manhattan_points(i,j,dist=1))
            
            
            d1,d2,d3,d4,d5,d6 = 10,1,1.5,0.7,0.5,np.maximum(10-(x+y),0)
            if self.side == 1:
                d3,d4 = 0.7,1.5
            
            eval_func = d1*e + d2*L + d3*x - d4*y + d5*d + d6*val 
            if for_opponent:
                eval_func = -d1*e - d2*L - d3*x + d4*y - d5*d - d6*val 
                
               
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
                print("My Stones          = "+str(x)+"  x"+str(d3)+"   = "+str(d3*x))
                print()
                print("Opponents Stones   = "+str(y)+"  x"+str(d4)+"   = "+str(-d4*y))
                print()
                print("Close Allies       = "+str(d)+"  x"+str(d5)+"   = "+str(d5*y))
                print()
                print("Reward of Position = "+str(val)+"  x"+str(d6)+"    = "+str(d6*val))
                print("="*40)      
                print("Eval = "+str(eval_func))
                print("="*40) 
            
            
            return eval_func
        else:
            del go_trial
            return False
    
    
    
    def Uvals(self, state_list):
        
        state_str = self.encode_board(state_list)
        state_failsafe = deepcopy(state_list)
        
        if state_str not in self.Utable:
            Umat=np.zeros((5,5))           
            for i in range(5):
                for j in range(5):
                    if state_list[i][j]==0:
                        
                        x = self.state_eval(state_list, i, j)
                        state_list=deepcopy(state_failsafe)
                        if x:
                            Umat[i][j] = x
            
            self.Utable[state_str] = Umat.tolist()
            if np.count_nonzero(Umat)==0:
                self.Utable[state_str] = False
            
        return self.Utable[state_str]
            
    
    
    
    def get_input(self, go, piece_type):
        
        state = deepcopy(go.board)
        mat = np.array(self.Uvals(state))
        
        if mat.any():
            
            i = np.where(mat == np.max(mat))[0][0]
            j = np.where(mat == np.max(mat))[1][0]
        
            return i,j
        else:
            return "PASS"
        
        
        
    def write_Q_table(self):
        with open("Qtable.json","w") as f:
            json.dump(self.Qtable,f)
            
            
    def write_history_table(self):
        with open("history_states.json","w") as f:
            json.dump(self.history_states,f)   
    
    def write_U_table(self):
        with open("utility_table.json","w") as f:
            json.dump(self.Utable,f)  
        
        
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
        
        self.history_states = []
   
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
    
    def stone_pos_reward(self, i, j):
        
        if i==0 or i==4 or j==0 or j==4:
            return 0.33
        if i==1 or i==3 or j==1 or j==3:
            return 0.66
        if i==2 or j==2:
            return 1
    
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
    
    
    
        

        
   
    
   
    
   
    
   
	
def run_testing(qplayer,rplayer,n,showturns=False,showboard=False):

    winloss = np.array([[0,0],[0,0]])
    N = 5
    	
    for i in range(n):
    
        my_go_board= GO_BOARD(N)
        qplayer.side=1
        rplayer.side=2 
        result=my_go_board.play(qplayer,rplayer,verbose=showboard)
            
        if result==1:
            if showturns:
                print("Your Agent Won playing as black")
            winloss[0][0]+=1
        else:
            if showturns:
                print("Your Agent lost as black")
            winloss[0][1]+=1
 
        del my_go_board
        
        my_go_board= GO_BOARD(N) 
        qplayer.side=2 
        rplayer.side=1   
        result=my_go_board.play(rplayer,qplayer,verbose=showboard)
        
        if result==2:
            if showturns:
                print("Your Agent Won playing as white")  
            winloss[1][0]+=1 
        else:
            if showturns:
                print("Your Agent lost as white")  
            winloss[1][1]+=1
        
        del my_go_board
        
    return winloss		    	
        
def display_winloss(winloss):
	
	winb = winloss[0][0]
	winw = winloss[1][0]
	winpb = (winloss[0][0]*100/(winloss[0][0]+winloss[0][1]))
	winpw = (winloss[1][0]*100/(winloss[1][0]+winloss[1][1]))
	winp = ( (winloss[1][0]+winloss[0][0])*100 / (winloss[1][0]+winloss[1][1]+winloss[0][0]+winloss[0][1]))
	
	print("="*10+str(np.sum(winloss))+" Iterations"+"="*10);
	print("# of Wins as Black= "+str(winb))
	print("# of Wins as White= "+str(winw))
	print("="*40)
	print("Win % as Black= "+str("{:.2f}".format(winpb))+"% ")
	print("Win % as White= "+str("{:.2f}".format(winpw))+"% ")
	print("Total Win %= "+str("{:.2f}".format(winp))+"% ")
	print("="*40)
   
   
        
if __name__ == "__main__":
    
	parser = argparse.ArgumentParser()
	parser.add_argument("-learn", type=int, help="Learn values", default=0)
	parser.add_argument("-win", type=int, help="Did you win", default=0)
	parser.add_argument("-iter", type=int, help="iterations of training", default=50000)
	parser.add_argument("-sturns", type=int, help="show turns", default=0)
	parser.add_argument("-sboard", type=int, help="show board", default=0)
	args=parser.parse_args()
		
	LEARN_NOW = args.learn
	winner = args.win
    
	
	showturns = (args.sturns == 1)
	showboard = (args.sboard == 1)
		
	iter=args.iter
	
	qplayer = myQlearner()
	#qplayer2 = myQlearner()
	rplayer = RandomPlayer()
	
	winloss=run_testing(qplayer,rplayer,int(iter/2),showturns,showboard)
    
	display_winloss(winloss)
	print(datetime.now() - startTime)
	qplayer.write_U_table()
	print(datetime.now() - startTime)
            
            
            
            
            
            
            
