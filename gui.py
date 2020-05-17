from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import video_processing


root = Tk()
root.title('Application')
# root.iconbitmap('images/icon.ico')
root.geometry("500x500")

# Center the frame
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2) - 150
positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2) - 150
root.geometry("+{}+{}".format(positionRight, positionDown))

# ------------------------ CHOOSE DIRECTORY OR FILE ------------------------
frame1 = LabelFrame(root, text="  Choose between a directory or a file  ")
frame1.pack(fill="x", padx=8, pady=8)

r = IntVar()

Radiobutton(frame1, text="Directory", variable=r, value=1).grid(row=0, column=0)
Radiobutton(frame1, text="File", variable=r, value=2).grid(row=0, column=1)


# -------------------------- SELECT PATH DIR/FILE --------------------------
def open_file():
    frame2.filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("mp4 files", "*.mp4"), ("MOV files", "*.MOV"), ("avi files", "*.avi"), ("All files", "*.*")))
    e.delete(0, END)
    e.insert(0, frame2.filename)
    # Label(frame2, text=frame2.filename).pack()


frame2 = LabelFrame(root, text="  Select the file/directory  ")
frame2.pack(fill="x", padx=8, pady=8)
e = Entry(frame2, width=71, fg="#696969")
e.pack(padx=5, pady=5, side=LEFT)
e.insert(END, "Enter the path")  # suggestion text
Button(frame2, text=" ... ", command=open_file).pack(padx=5, pady=5, side=RIGHT)


# ----------------------------- CHOOSE OPTIONS -----------------------------
frame3 = LabelFrame(root, text="  Select the configuration  ")
frame3.pack(fill="x", padx=8, pady=8)


def var_states():
    Label(root, text="Perform for path %s with config %d-%d-%d-%d-%d" % (e.get(), pnt_detection.get(), pnt_rectification.get(), pnt_retrieval.get(), ppl_detection.get(), ppl_localization.get())).pack()
    video_processing.process_video(e.get(), 0)


pnt_detection = IntVar()
pnt_rectification = IntVar()
pnt_retrieval = IntVar()
ppl_detection = IntVar()
ppl_localization = IntVar()
Checkbutton(frame3, text="Painting Detection", variable=pnt_detection).grid(row=0, sticky=W)
Checkbutton(frame3, text="Painting Rectification", variable=pnt_rectification).grid(row=1, sticky=W)
Checkbutton(frame3, text="Painting Retrieval", variable=pnt_retrieval).grid(row=2, sticky=W)
Checkbutton(frame3, text="People Detection", variable=ppl_detection).grid(row=3, sticky=W)
Checkbutton(frame3, text="People Localization", variable=ppl_localization).grid(row=4, sticky=W)

# ------------------------------ ENTER OR QUIT ------------------------------
Button(root, text='Start', command=var_states).pack(pady=3)
Button(root, text='Quit', command=root.quit).pack(pady=3)


root.mainloop()
