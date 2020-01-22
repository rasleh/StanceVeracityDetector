from tkinter import Tk, Label, Button, Entry, IntVar, END, W, E, Frame
from src import data_loader

LARGE_FONT= ("Verdana", 12)


class AnnotationGUI(Tk):
    def __init__(self, *args, **kwargs):
        Tk.__init__(self, *args, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        main_container = Frame(self)
        main_container.grid(column=0, row=0, sticky='nesw')
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for frame_class in (StartPage, ClaimPage, StancePage):
            frame = frame_class(main_container, self)

            self.frames[frame_class] = frame

            frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.show_frame(StartPage)

    def show_frame(self, frame_class):
        frame = self.frames[frame_class]
        frame.tkraise()


class StartPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        label = Label(self, text="Perform annotation for claim identification or stance prediction?", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=10, columnspan=2)

        claim_button = Button(self, text="Claim", command=lambda: controller.show_frame(ClaimPage))
        claim_button.grid(row=1, column=0)

        stance_button = Button(self, text="Stance", command=lambda: controller.show_frame(StancePage))
        stance_button.grid(row=1, column=1)


class ClaimPage(Frame):
    def __init__(self, parent, controller):
        Frame.__init__(self, parent)
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.controller = controller
        self.columns = 3
        self.raw_data = []
        label = Label(self, text="Claim annotator", font=LARGE_FONT)
        label.grid(row=0, column=0, pady=10, columnspan=self.columns)

        self.data_button = Button(self, text="Load data", command=lambda: self.load_data())
        self.data_button.grid(row=1, column=0, columnspan=self.columns)

    def load_data(self):
        w = 600
        h = 600
        self.controller.geometry('{}x{}'.format(w, h))
        self.raw_data = data_loader.load_annotation_data('claim')
        self.data_button.destroy()
        label = Label(self, text="Data loaded successfully")
        label.grid(row=1, column=0, columnspan=self.columns, pady=10, padx=10)

        # Set up annotation
        print(self.raw_data)


class StancePage(Frame):
    def __init__(self, parent, controller):
        self.raw_data = []
        Frame.__init__(self, parent)
        title = Label(self, text="Stance annotator", font=LARGE_FONT)
        title.place(x=self.winfo_width()/2, y=25, anchor='center')

        self.data_button = Button(self, text="Load data", command=lambda: self.load_data())
        self.data_button.grid(pady=10, padx=10)

    def load_data(self):
        self.raw_data = data_loader.load_annotation_data('stance')
        self.data_button.destroy()

        label = Label(self, text="Data loaded successfully")
        label.grid(pady=10, padx=10)
        print(len(self.raw_data))
        for i in range(len(self.raw_data)):

            for j in range(len(self.raw_data[0])):
                datapoint = Entry(self, text=self.raw_data[i][j]['full_text'])
                datapoint.grid()
        # Set up annotation
        print(self.raw_data)

my_gui = AnnotationGUI()
my_gui.mainloop()
