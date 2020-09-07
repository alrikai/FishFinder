from collections import deque

class GuideHistory():
    def __init__(self, num_history, num_track):
        self.num_history = num_history
        self.num_htrack = num_track
        self.reset_history()

    def add(self, history_data):
        assert(len(history_data) == self.num_htrack)

        #if history is empty (i.e. first call to add), duplicate the inputs
        #to fill out the history
        if len(self.history[0]) == 0:
            #populate the frame histories
            for history_idx in range(self.num_htrack):
                for _ in range(self.num_history):
                    self.history[history_idx].append(history_data[history_idx])
        else:
            #rotate the mask and flow history, s.t. the oldest one is at the end.
            #then we pop the oldest off and append the newest
            for history_idx in range(self.num_htrack):
                self.history[history_idx].rotate(-1)
                self.history[history_idx].pop()
                self.history[history_idx].append(history_data[history_idx])

    def get_history(self):
        return self.history

    def reset_history(self):
        self.history = []
        for _ in range(self.num_htrack):
            self.history.append(deque([], maxlen=self.num_history))
