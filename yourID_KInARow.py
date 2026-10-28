'''
<yourID>_KInARow.py
Authors: Nicole Ham, Rena Yin

An agent for playing "K-in-a-Row with Forbidden Squares" and related games.

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

from agent_base import KAgent
from game_types import State, Game_Type
from winTesterForK import winTesterForK

AUTHORS = 'Nicole Ham and Rena Yin' 
STUDENT_IDS = ['n1coleh', '[ID]'] # The first ID here should
# match the one in the file name, e.g., jsmith99_KInARow.py.

import time # You'll probably need this to avoid losing a
 # game due to exceeding a time limit.

# Create your own type of agent by subclassing KAgent:

class OurAgent(KAgent):  # Keep the class name "OurAgent" so a game master
    # knows how to instantiate your agent class.

    def __init__(self, twin=False):
        self.twin=twin
        self.nickname = 'Nic'
        if twin: self.nickname += '2'
        self.long_name = 'Templatus Skeletus'
        if twin: self.long_name += ' II'
        self.persona = 'bland'
        self.voice_info = {'Chrome': 10, 'Firefox': 2, 'other': 0}
        self.playing = "don't know yet" # e.g., "X" or "O".
        self.alpha_beta_cutoffs_this_turn = -1
        self.num_static_evals_this_turn = -1
        self.zobrist_table_num_entries_this_turn = -1
        self.zobrist_table_num_hits_this_turn = -1
        self.current_game_type = None
        self.playing_mode = KAgent.DEMO
        self.special_static_eval_fn = None
        self.deadline = None         # time.time() value at which search must stop
        self.search_root = None      # root state of the current search
        self.preferred_move = None   # best root move from the previous depth

    def introduce(self):
        intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
        if self.twin: intro += "By the way, I'm the TWIN.\n"
        return intro

    # Receive and acknowledge information about the game from
    # the game master:
    def prepare(
        self,
        game_type,
        what_side_to_play,
        opponent_nickname,
        expected_time_per_move = 0.1, # Time limits can be
                                      # changed mid-game by the game master.

        utterances_matter=True):      # If False, just return 'OK' for each utterance,
                                      # or something simple and quick to compute
                                      # and do not import any LLM or special APIs.
                                      # During the tournament, this will be False..
        # Save the setup on this agent so other methods can use it later.
        # Optionally, import your LLM API here.
        # Then you can use it to help create utterances.
        self.current_game_type = game_type
        self.playing = what_side_to_play
        self.opponent_nickname = opponent_nickname
        self.expected_time_per_move = expected_time_per_move
        self.utterances_matter = utterances_matter

        # The autograder checks for this exact string before running any tests.
        # The caller controls playing_mode; preserve its selected mode.
        return "OK"
   
    # The core of your agent's ability should be implemented here:             
    def make_move(self, current_state, current_remark, time_limit=1000,
                  use_alpha_beta=True,
                  use_zobrist_hashing=False, max_ply=3,
                  special_static_eval_fn=None):
        # --- HOW TO IMPLEMENT make_move ---
        # 1. Generate candidate moves: iterate over rows and columns of
        #    current_state.board; any square equal to ' ' is a legal move.
        # 2. For each candidate position (i, j), build a successor state:
        #      new_s = State(old=current_state)       # deep copy
        #      new_s.board[i][j] = current_state.whose_move
        #      new_s.change_turn()                    # switch X <-> O
        # 3. Count blank squares before searching to detect a full board (draw)
        #    and to limit recursion depth appropriately.
        # 4. Call self.minimax(...) on the current state to get the best move.
        # 5. Compose an utterance string appropriate to your agent's persona.
        # 6. Return format:
        #      DEMO / COMPETITION: [[move, new_state], utterance]
        #      AUTOGRADER:         [[move, new_state, stats], utterance]
        #    where stats is a dict with keys defined in agent_base.py, e.g.:
        #      stats['alpha_beta_cutoffs_this_turn']        = self.alpha_beta_cutoffs_this_turn
        #      stats['num_static_evals_this_turn']          = self.num_static_evals_this_turn
        #      stats['zobrist_table_num_entries_this_turn'] = self.zobrist_table_num_entries_this_turn
        #      stats['zobrist_table_num_hits_this_turn']    = self.zobrist_table_num_hits_this_turn

        start = time.time()

        # Reset per-turn statistics.
        self.alpha_beta_cutoffs_this_turn = 0
        self.num_static_evals_this_turn = 0
        self.special_static_eval_fn = special_static_eval_fn

        # Leave a safety margin so we return before the game master's limit.
        if time_limit is None:
            self.deadline = None
        else:
            self.deadline = start + TIME_SAFETY_FRACTION * time_limit

        # Count blank squares to detect a full board and limit depth.
        num_blanks = count_blanks(current_state)
        best_move, best_state = None, None
        self.search_root = current_state
        self.preferred_move = None

        if self.playing_mode == KAgent.AUTOGRADER:
            # Exactly one search at max_ply, in plain row-major order,
            # so the leaf counts match what the autograder expects.
            depths = [min(max_ply, num_blanks)]
        else:
            # Iterative deepening: search depth 1, 2, 3, ... until time runs
            # out, keeping the result of the deepest search that finished.
            depths = range(1, num_blanks + 1)

        for depth in depths:
            try:
                result = self.minimax(current_state, depth, pruning=use_alpha_beta)
            except SearchTimeout:
                break
            if result[1] is not None:
                best_move, best_state = result[1], result[2]
                self.preferred_move = best_move  # search it first next depth

        if best_move is None:
            # Full board, or time ran out before depth 1 finished:
            # take the first legal move if one exists.
            successors = list(successors_and_moves(current_state))
            if successors:
                best_move, best_state = successors[0]
            else:
                best_move, best_state = None, State(old=current_state)
                best_state.change_turn()

        new_remark = "I'll play at " + str(best_move) + "."

        if self.playing_mode != KAgent.AUTOGRADER:
            return [[best_move, best_state], new_remark]

        stats = {
            'alpha_beta_cutoffs_this_turn': self.alpha_beta_cutoffs_this_turn,
            'num_static_evals_this_turn': self.num_static_evals_this_turn,
            'zobrist_table_num_entries_this_turn': self.zobrist_table_num_entries_this_turn,
            'zobrist_table_num_hits_this_turn': self.zobrist_table_num_hits_this_turn,
            'zobrist_num_useful_hits_for_minimax': -1,
            'zobrist_num_useful_hits_for_ordering': -1
        }
        return [[best_move, best_state, stats], new_remark]

    # The main adversarial search function:
    def minimax(self,
            state,
            depth_remaining,
            pruning=False,
            alpha=None,
            beta=None):
        # --- HOW TO IMPLEMENT minimax ---
        # Base cases — call static_eval and return its value:
        #   - depth_remaining == 0  (depth limit reached)
        #   - no blank squares remain on the board (draw)
        #   - terminal win/loss state (do not generate moves after a win)
        # Recursive case:
        #   is_max = (state.whose_move == 'X')   # X maximizes, O minimizes
        #   best_val = -inf if is_max else +inf
        #   For each successor state generated by your move generator:
        #     result = self.minimax(successor, depth_remaining - 1, pruning, alpha, beta)
        #     Update best_val and best_move.
        #     If pruning is True: check alpha/beta cutoff and break early.
        #     Update alpha (if is_max) or beta (if not is_max) after each child.
        # Return at minimum [score, best_move, best_state].
        # Additional values after index 0 can carry info back to make_move.

        # Give up on this search if we are out of time; make_move falls back
        # to the result of the last depth that finished.
        if self.deadline is not None and time.time() > self.deadline:
            raise SearchTimeout()

        # Base cases: game finished, depth limit reached, or no blank squares remain.
        # Stop at the depth limit or a draw, even without a finished flag.
        if state.finished or depth_remaining == 0 or count_blanks(state) == 0:
            return [self.evaluate(state), None, None]

        # Hypothetical successors do not have finished set by the game master.
        # Detect wins from the board before allowing any further moves.
        if has_winner(state, self.current_game_type.k):
            return [self.evaluate(state), None, None]
        # None means "no bound".
        if alpha is None: alpha = float('-inf')
        if beta is None: beta = float('inf')

        is_max = (state.whose_move == 'X')  # X maximizes, O minimizes
        best_val = float('-inf') if is_max else float('inf')
        best_move = None
        best_state = None

        children = successors_and_moves(state)
        if state is self.search_root and self.preferred_move is not None \
                and self.playing_mode != KAgent.AUTOGRADER:
            # Try the previous depth's best move first for more cutoffs.
            children = sorted(children, key=lambda c: c[0] != self.preferred_move)

        game_type = self.current_game_type
        for move, successor in children:
            # A move that completes k in a row ends the game, so the
            # successor is a leaf no matter how much depth remains.
            if game_type is not None and \
                    winTesterForK(successor, move, game_type.k) != 'No win':
                successor.finished = True
            result = self.minimax(successor, depth_remaining - 1, pruning, alpha, beta)
            val = result[0]

            if (is_max and val > best_val) or (not is_max and val < best_val):
                best_val = val
                best_move = move
                best_state = successor

            if pruning:
                if is_max:
                    if best_val >= beta:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
                    alpha = max(alpha, best_val)
                else:
                    if best_val <= alpha:
                        self.alpha_beta_cutoffs_this_turn += 1
                        break
                    beta = min(beta, best_val)

        return [best_val, best_move, best_state]

    # Uses the special static evaluator when one was given (AUTOGRADER mode),
    # otherwise our own static_eval.  Counts every evaluation.
    def evaluate(self, state):
        # Return a numeric score: positive = good for X, negative = good for O.
        # Use game_type.k (the win length) to evaluate partial lines.
        # IMPORTANT: k is NOT stored in the state — you must get it from game_type.
        # In DEMO/COMPETITION mode game_type will be None; use self.current_game_type
        # (which you should have saved during prepare).
        self.num_static_evals_this_turn += 1
        if self.special_static_eval_fn is not None:
            return self.special_static_eval_fn(state)
        return self.static_eval(state, self.current_game_type)
 
    def static_eval(self, state, game_type=None):
        """Score potential winning segments: positive favors X, negative O.

        Generalizes the unblocked-line heuristic from Adversarial Search I,
        slides 9–11, to variable k and boards with forbidden squares.
        """
        if game_type is None:
            game_type = self.current_game_type
        if game_type is None:
            raise ValueError("static_eval needs a game_type or a prior prepare call.")

        board = state.board
        rows, columns = len(board), len(board[0])
        k = game_type.k
        score = 0
        # At most four segments start at each cell. This finite win score
        # exceeds the sum of all possible nonwinning segment contributions.
        win_score = 4 * rows * columns * 10**k + 1
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for row in range(rows):
            for column in range(columns):
                for dr, dc in directions:
                    end_row = row + (k - 1) * dr
                    end_column = column + (k - 1) * dc
                    if not (0 <= end_row < rows and 0 <= end_column < columns):
                        continue

                    segment = [board[row + step * dr][column + step * dc]
                               for step in range(k)]
                    if '-' in segment:
                        continue
                    x_count = segment.count('X')
                    o_count = segment.count('O')

                    # Completed win dominates ordinary positional scores
                    if x_count == k:
                        return win_score
                    if o_count == k:
                        return -win_score

                    # Neither player can complete a segment occupied by both
                    if x_count and o_count:
                        continue
                    if x_count:
                        score += 10**(x_count - 1)
                    elif o_count:
                        score -= 10**(o_count - 1)

        if count_blanks(state) == 0:
            return 0
        return score
 
def has_winner(state, k):
    """Detect a winning line without changing the state or evaluating it."""
    board = state.board
    rows, columns = len(board), len(board[0])
    for row in range(rows):
        for column in range(columns):
            token = board[row][column]
            if token not in ('X', 'O'):
                continue
            for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                end_row = row + (k - 1) * dr
                end_column = column + (k - 1) * dc
                if not (0 <= end_row < rows and 0 <= end_column < columns):
                    continue
                if all(board[row + step * dr][column + step * dc] == token
                       for step in range(1, k)):
                    return True
    return False

# Fraction of the time limit we allow ourselves to spend searching.
TIME_SAFETY_FRACTION = 0.8

# Raised inside minimax when the deadline for this move has passed.
class SearchTimeout(Exception):
    pass

# Count the blank (playable) squares on the board.
def count_blanks(state):
    return sum(row.count(' ') for row in state.board)

# Generate [move, new_state] pairs in row-major order
# (row by row, left to right), as required in AUTOGRADER mode.
def successors_and_moves(state):
    b = state.board
    for i in range(len(b)):
        for j in range(len(b[0])):
            if b[i][j] != ' ': continue
            new_s = State(old=state)       # deep copy
            new_s.board[i][j] = state.whose_move
            new_s.change_turn()            # switch X <-> O
            yield [(i, j), new_s]

# OPTIONAL THINGS TO KEEP TRACK OF:

#  WHO_MY_OPPONENT_PLAYS = other(WHO_I_PLAY)
#  MY_PAST_UTTERANCES = []
#  OPPONENT_PAST_UTTERANCES = []
#  UTTERANCE_COUNT = 0
#  REPEAT_COUNT = 0 or a table of these if you are reusing different utterances

