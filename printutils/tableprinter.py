import tabulate
import numpy as np

class TablePrinter:
    def __init__(self, frame, rows_between_header=10):

        self.frame = frame
        self.rows_between_header = rows_between_header
        self.rows = self.frame.shape[0]
        self.headers = list(self.frame.columns)
        self.rows_added_since_last_header = self.rows
        self.rows_added_since_last_print = self.rows

        if self.rows_added_since_last_header >= self.rows_between_header:
            self.print_header_next = True
        else:
            self.print_header_next = False

    def tab_all(self):
        return tabulate.tabulate(self.frame, headers=self.headers)

    def tab_last_n(self, n):
        return self.tab_all().split('\n')[-n:]

    def tab_header(self):
        return '\n'.join(self.tab_all().split('\n')[:2])

    def new_rows(self, frame):
        self.frame = self.frame.append(frame)
        self.rows = self.frame.shape[0]
        self.rows_added_since_last_header += frame.shape[0]
        self.rows_added_since_last_print += frame.shape[0]

        if self.rows_added_since_last_header >= self.rows_between_header:
            self.print_header_next = True

    def tab_out(self):
        if self.rows_added_since_last_print == 0:
            return

        rows_out = self.tab_last_n(n=self.rows_added_since_last_print)
        self.rows_added_since_last_print = 0

        if self.print_header_next:
            self.print_header_next = False
            self.rows_added_since_last_header = 0
            return '\n'.join([self.tab_header()] + rows_out)
        else:
            return '\n'.join(rows_out)

    def print_width(self):
        return len(self.tab_all().split('\n')[2])

    def break_out(self, divider='-'):
        return divider * self.print_width()

    def msg_out(self, msg, divider='-'):
        print_width = len(self.tab_all().split('\n')[2])
        msg_width = len(msg)
        divider_width_left = int(np.floor((print_width - msg_width - 2) / 2))
        divider_width_right = print_width - 2 - divider_width_left - msg_width

        return "{dl} {m} {dr}".format(dl=divider * divider_width_left,
                                      m=msg,
                                      dr=divider * divider_width_right)
