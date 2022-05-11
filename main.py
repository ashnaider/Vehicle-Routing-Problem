import os
import numpy as np
from numpy import genfromtxt

from PyQt5 import QtCore, QtWidgets, uic, QtTest
from PyQt5.QtWidgets import QMessageBox, QFileDialog, \
    QDialog, QPushButton, QToolBar, QAction, \
    QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pylab as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib import cm
matplotlib.rcParams['mathtext.fontset'] = 'cm'

from algo import *
from utils import *
from dataframe_model import *
from table_model import *

main_window = os.path.join('UI', 'main_window.ui')
form, base = uic.loadUiType(uifile=main_window)


class MainWindow(base, form):
    def __init__(self):
        super(base, self).__init__()
        self.setupUi(self)

        self.data = None

        self.subplot_rows = 1
        self.subplot_cols = 2

        self.legend_font_size = 12
        self.legend_cols = 2

        self.connect_buttons()
        self.connect_toolbar_actions()
        self.connect_canvas()

        # self.run_debug()

    def connect_buttons(self):
        self.calculate_btn.clicked.connect(self.run_algo)


    def connect_toolbar_actions(self):
        self.open_file_action.triggered.connect(self.upload_data_from_file)

    def connect_canvas(self):
        self.figure = plt.figure()

        # self.cid = self.figure.canvas.mpl_connect('button_press_event', self.process_canvas_click)

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lay = QtWidgets.QVBoxLayout(self.plot_widget)  
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)


    ########## DEBUG CODE ################
    def run_debug(self):
        self.data = self.read_data_from_csv('./samples/var_1.csv')
        self.run_algo()


    ########### DRAW VISUALIZATIONS #########

    def plot_points(self, ax, x, y, center):
        ax.scatter(x, y, marker='D')
        ax.scatter([center[0]], [center[1]], marker='8', s=70)

        ax.annotate("center", (center[0]+0.3, center[1]), fontsize=14)
        for i in range(len(x)):
            ax.annotate(str(i+1), (x[i]+0.5, y[i]), fontsize=14)

    def plot_radial_routes(self, x, y, center):
        self.ax_radial = plt.subplot(self.subplot_rows, self.subplot_cols, 1)
        # self.ax = plt.subplot()
        self.ax_radial.clear()

        self.plot_points(self.ax_radial, x, y, center)

        colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta']
        for i in range(len(x)):
            self.ax_radial.plot([center[0], x[i]], [center[1], y[i]], color='red')

        self.ax_radial.grid()
        self.figure.tight_layout()
        self.canvas.draw()

    def plot_circular_routes(self, x, y, center, routes):
        self.ax_circular = plt.subplot(self.subplot_rows, self.subplot_cols, 2)
        # self.ax = plt.subplot()
        self.ax_circular.clear()

        self.plot_points(self.ax_circular, x, y, center)
        
        self.ax_circular.scatter(x, y, marker='D')
        self.ax_circular.scatter([center[0]], [center[1]], marker='8', s=70)

        self.ax_circular.annotate("center", (center[0]+0.3, center[1]), fontsize=14)
        for i in range(len(x)):
            self.ax_circular.annotate(str(i+1), (x[i]+0.5, y[i]), fontsize=14)

        colors = ['red', 'blue', 'green', 'orange', 'cyan', 'magenta']
        
        for j, route in enumerate(routes):
            if len(route) > 1:
                for i in range(1, len(route)):
                    a = route[i] - 1
                    b = route[i-1] - 1
                    self.ax_circular.plot([x[a], x[b]], [y[a], y[b]], color=colors[j])
            
            start = route[0] - 1
            end = route[-1] - 1
            self.ax_circular.plot([center[0], x[start]], [center[1], y[start]], color=colors[j])
            self.ax_circular.plot([center[0], x[end]], [center[1], y[end]], color=colors[j])

        self.ax_circular.grid()
        self.figure.tight_layout()
        self.canvas.draw()


    ########### RUN ALGORITHM ##############

    def run_algo(self):
        if self.data is None:
            self.show_efrror_box("Load data from the file first")
            return

        x = self.data[:, 0]
        y = self.data[:, 1]
        orderings = self.data[:, 2]
        capacity = self.data[:, 3][0]

        print("x: ", x)
        print("y: ", y)
        print("orderings: ", orderings)
        print("capacity: ", capacity)

        res = ClarkeRight(x, y, orderings=orderings, capacity=capacity)
        routes, NKB, total_vol, total_milage, res, info_df = res

        dist_m = calc_dist_matrix(x, y)
        win_m = calc_win_matrix(x, y, dist_m=dist_m)

        self.set_iterations_table(info_df)

        self.set_dist_matrix_table(dist_m)
        self.set_win_dist_matrix_table(win_m)
        self.set_result_matrix_table(res)

        self.plot_radial_routes(x[1:], y[1:], (x[0], y[0]))
        self.plot_circular_routes(x[1:], y[1:], (x[0], y[0]), routes=routes)

        
        radial_str = "L = 2*("
        for val in dist_m[0]:
            radial_str += f"{val:.1f} + "
        before = 2 * np.sum(dist_m[0])
        radial_str += f") = {before:.1f}\n"

        self.results_textEdit.setText('')

        self.results_textEdit.append(radial_str)
        self.results_textEdit.append(f"Routes len before: {before:.2f}\n")

        self.results_textEdit.append(f"Routes: {routes}\n")
        
        NKB_str = "TOTAL WIN = "
        for i in NKB:
            NKB_str += f"{i:.2f} + "

        NKB_str += f"= {sum(NKB):.2f}\n"

        self.results_textEdit.append(f"Total win: {NKB_str}")
        
        self.results_textEdit.append(f"Routes len after: {total_milage:.2f}\n")
        # self.results_textEdit.append(f"Routes len after 2: {before - sum(NKB):.2f}\n")

        self.results_textEdit.append(f"Total volume: {total_vol:.2f}")

        print("AFTER PLOT POINTS!")


    ############ TABLES DISPLAY ##############

    def set_iterations_table(self, iter_log):
        # pd.set_option('precision', 3)

        iter_log_model = DataFrameModel(iter_log)

        self.iterations_log_tableView.setModel(iter_log_model)

        self.iterations_log_tableView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        
        self.iterations_log_tableView.resizeColumnsToContents()

    def set_dist_matrix_table(self, dist_m):
        dist_m_model = TableModel(dist_m)
        self.dist_matrix_tableView.setModel(dist_m_model)
        self.dist_matrix_tableView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.dist_matrix_tableView.resizeColumnsToContents()

    def set_win_dist_matrix_table(self, win_dist_m):
        win_dist_m_model = TableModel(win_dist_m)
        self.win_dist_matrix_tableView.setModel(win_dist_m_model)
        self.win_dist_matrix_tableView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.win_dist_matrix_tableView.resizeColumnsToContents()

    def set_result_matrix_table(self, res_m):
        win_dist_m_model = DataFrameModel(res_m)
        self.results_tableView.setModel(win_dist_m_model)
        self.results_tableView.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)
        self.results_tableView.resizeColumnsToContents()
        
    def clear_tables(self):
        self.got_tables = False
        empty_df1 = pd.DataFrame()
        empty_model1 = DataFrameModel(empty_df1)

        self.h1_tableView.setModel(empty_model1)
        self.h2_tableView.setModel(empty_model1)
        self.h1_result_tableView.setModel(empty_model1)
        self.h2_result_tableView.setModel(empty_model1)


    ############ BOX MESSAGES ################

    def show_error_box(self, str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setText("Error")
        msg.setInformativeText(str)
        msg.setWindowTitle("Error")
        msg.exec_()

    def show_ok_box(self, str):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Ok")
        msg.setInformativeText(str)
        msg.setWindowTitle("Ok")
        msg.exec_()


    ########### FILE READING PROCEDURE ##################
            
    def read_data_from_csv(self, filename):
        raw_data = genfromtxt(filename, delimiter=';', skip_header=True)
        print("raw data: ", raw_data)
        print("raw_data shape: ", raw_data.shape)

        return raw_data

    def upload_data_from_file(self):
        filename = getFileNameToOpen(self)
        if not filename:
            return 
        try:
           self.data = self.read_data_from_csv(filename)
        except Exception as e:
            self.show_error_box("Unable to read file: " + str(e))
            return 


def main():
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()