import tkinter
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from Utility.Utility import Utility
from ensemble import Ensemble
import threading
from tkinter import ttk


class Interface(object):

    def select_train_data(self):
        traindatapath = filedialog.askopenfilename(title='选择数据源', filetypes=[('文本文档', '*.txt'),
                                                                        ('excel文件', '*.xlsx;*.xls'), ('所有文件', '*')])
        if not traindatapath:
            self.util.filename = False
        else:
            self.textvar1.set(traindatapath)
            self.util.filename = traindatapath

    def select_test_data(self):
        testdatapath = filedialog.askopenfilename(title='选择测试数据', filetypes=[('文本文档', '*.txt'),
                                                                         ('excel文件', '*.xlsx;*.xls'), ('所有文件', '*')])
        if not testdatapath:
            self.util.filename = False
        else:
            self.textvar4.set(testdatapath)
            self.util.filename = testdatapath

    def select_model(self):
        modelpath = filedialog.askdirectory(title='选择模型文件夹')
        if not modelpath:
            self.util.modelpath = False
        else:
            self.textvar3.set(modelpath)
            self.util.modelpath = modelpath

    def save_model(self):
        savemodelpath = filedialog.askdirectory(title='选择模型存放位置')
        if not savemodelpath:
            self.util.savemodelpath = False
        else:
            self.textvar2.set(savemodelpath)
            self.util.savemodelpath = savemodelpath

    def thread_assistant_func1(self):
        self.util.data_for_train()
        self.ensemble.ensemble_train()
        # self.thrain_done = True

    def begin_train(self):
        try:
            if self.util.filename and self.util.savemodelpath:
                # tkinter.Label(, text='开始训练任务，请耐心等待...').pack()
                # messagebox.showinfo('提示', '开始训练任务，请耐心等待...')
                win = tkinter.Toplevel()
                win.title('提示')
                win.geometry('320x160+960+480')
                tkinter.Label(win, text='训练中，请耐心等待...').grid(row=1, column=1)
                p = ttk.Progressbar(win, length=200, mode='indeterminate', orient=tkinter.HORIZONTAL)
                p.grid(row=2, column=1)
                p.start()
                win.protocol('WM_DELETE_WINDOW', lambda _: None)

                # self.train_done = False
                thread = threading.Thread(target=self.thread_assistant_func1, args=())
                thread.setDaemon(True)
                thread.start()
                while not self.train_done:
                    pass
                win.destroy()
                messagebox.showinfo('提示', '训练任务已完成!')
                # self.util.data_for_train()
                # self.ensemble.ensemble_train()
            else:
                messagebox.showwarning('警告', '请输入正确的文件路径!')
                return
        except Exception as e:
            print(e)
            messagebox.showwarning('警告', '请输入正确的文件路径!')
            return

    def thread_assistant_func2(self, win):
        self.util.data_for_train()
        self.ensemble.ensemble_test()
        # self.event.set()
        # self.test_done = True
        win.destroy()
        messagebox.showinfo('提示', '测试任务已完成!')

    def begin_test(self):
        try:
            if self.util.filename and self.util.modelpath:
                win = tkinter.Toplevel()
                win.title('提示')
                win.geometry('320x160+960+480')
                tkinter.Label(win, text='测试中，请耐心等待...').pack(pady=10)
                p = ttk.Progressbar(win, length=200, mode='indeterminate', orient=tkinter.HORIZONTAL)
                p.pack(pady=10)
                p.start()
                win.protocol('WM_DELETE_WINDOW', lambda _: None)

                # self.test_done = False
                # self.event = threading.Event()
                thread = threading.Thread(target=self.thread_assistant_func2, args=(win,))
                thread.start()
                # thread.join()
                # self.event.wait()
                # messagebox.showinfo('提示', '开始测试任务，请耐心等待...')
                # self.util.data_for_train()
                # self.ensemble.ensemble_test()
                # tkinter.Label(, text='开始测试任务，请耐心等待...').pack()
            else:
                messagebox.showwarning('警告', '请输入正确的文件路径!')
                return
        except Exception as e:
            print(e)
            messagebox.showwarning('警告', '请输入正确的文件路径!')
            return

    def select_model_panel(self):

        win1 = tkinter.Toplevel()
        win1.title('选择模型文件')
        win1.geometry('640x320+800+400')
        # win1.wm_attributes('-topmost', 1)
        win1.resizable(0, 0)

        frame = tkinter.Frame(win1)
        frame.pack(pady=34)

        textvar3 = tkinter.StringVar()
        textvar4 = tkinter.StringVar()
        textvar3.set('选择已训练的模型')
        textvar4.set('选择待预测的数据')
        self.textvar3 = textvar3
        self.textvar4 = textvar4
        row1 = tkinter.Frame(frame)
        label = tkinter.Label(row1, textvariable=self.textvar3, font='arial', bg="white", width=50, anchor='w')
        label.pack(side=tkinter.LEFT)
        tkinter.Button(row1, text='选择', font='arial', command=self.select_model).pack(
            side=tkinter.LEFT, padx=10)
        row1.pack(side=tkinter.TOP)

        row2 = tkinter.Frame(frame)
        tkinter.Label(row2, textvariable=self.textvar4, font='arial', bg="white", width=50, anchor='w').pack(
            side=tkinter.LEFT)
        tkinter.Button(row2, text='选择', font='arial', command=self.select_test_data).pack(
            side=tkinter.LEFT, padx=10)
        row2.pack(side=tkinter.TOP, pady=10)

        row3 = tkinter.Frame(frame)
        tkinter.Button(row3, text='开始预测', font='arial', command=self.begin_test).pack()
        row3.pack(side=tkinter.BOTTOM, pady=10)

    def __init__(self):

        util = Utility()
        self.ensemble = Ensemble(util)
        self.util = util

        root = tkinter.Tk()
        root.title('选择训练文件')
        root.geometry('640x320+800+400')
        root.resizable(0, 0)
        self.root = root

        frame = tkinter.Frame(root)
        frame.pack(pady=34)

        textvar1 = tkinter.StringVar()
        textvar2 = tkinter.StringVar()
        textvar1.set('选择待训练的数据源')
        textvar2.set('选择存放模型的位置')
        self.textvar1 = textvar1
        self.textvar2 = textvar2
        row1 = tkinter.Frame(frame)
        tkinter.Label(row1, textvariable=self.textvar1, font='arial', bg="white", width=50, anchor='w').pack(
            side=tkinter.LEFT)
        tkinter.Button(row1, text='选择', font='arial', command=self.select_train_data).pack(
            side=tkinter.LEFT, padx=10)
        row1.pack(side=tkinter.TOP)

        row2 = tkinter.Frame(frame)
        tkinter.Label(row2, textvariable=self.textvar2, font='arial', bg="white", width=50, anchor='w').pack(
            side=tkinter.LEFT)
        tkinter.Button(row2, text='选择', font='arial', command=self.save_model).pack(
            side=tkinter.LEFT, padx=10)
        row2.pack(side=tkinter.TOP, pady=10)

        row3 = tkinter.Frame(frame)
        tkinter.Button(row3, text='开始训练', font='arial', command=self.begin_train).pack(
            side=tkinter.LEFT, padx=20)
        tkinter.Button(row3, text='读取模型', font='arial', command=self.select_model_panel).pack(
            side=tkinter.LEFT, padx=20)
        row3.pack(side=tkinter.BOTTOM, pady=10)

