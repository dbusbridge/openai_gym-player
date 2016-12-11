import tabulate
import numpy as np
import pandas as pd


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

df = pd.DataFrame({'a': np.random.normal(size=10),
                   'b': np.random.normal(size=10)})

tp = TablePrinter(frame=df)

for i in range(100):
    new_df = pd.DataFrame({'a': np.random.normal(size=1),
                           'b': np.random.normal(size=1)},
                          index=pd.Series([i]))

    tp.new_rows(frame=new_df)
    print(tp.tab_out())


tp.tab_last_n(n=tp.rows_added_since_last_print)