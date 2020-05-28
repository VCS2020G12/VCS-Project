from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import video_processing

# HEIGHT = 1000
# WIDTH = 900

root = Tk()
root.title('Painting Detection and Recognition')
root.geometry("500x680")
root.resizable(0,0)

# Center the frame
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2) - 150
positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2) - 150
root.geometry("+{}+{}".format(positionRight, positionDown))

# canvas = Canvas(root, height=HEIGHT, width=WIDTH)
# canvas.pack()

background_image = PhotoImage(file='images/background.png')
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)


# -------------------------- SELECT PATH DIR/FILE --------------------------
def open_file():
    frame_path.filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("mp4 files", "*.mp4"), ("MOV files", "*.MOV"), ("avi files", "*.avi"), ("All files", "*.*")))
    e.delete(0, END)
    e.insert(0, frame_path.filename)
    # Label(path, text=frame_path.filename).pack()


frame_path = Frame(root, bg='#ffffff', bd=5)
frame_path.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.15, anchor='n')
Label(frame_path, width=34, fg="#696969", bg='#ffffff', font=5, text='Type or select the path of the video:', anchor='w').grid(row=0, padx=10, pady=10)
e = Entry(frame_path, width=34, fg="#696969", bg='#ffffff', font=9)
e.grid(row=1, column=0, padx=10, pady=10)
e.insert(END, "Enter the path")
Button(frame_path, text=" ... ", command=open_file, bg='#ffffff').grid(row=1, column=1, padx=10, pady=10)

# ----------------------------- SELECT OPTIONS -----------------------------
frame_options = Frame(root, bg='#ffffff', bd=5)
frame_options.place(relx=0.5, rely=0.29, relwidth=0.9, relheight=0.4, anchor='n')
Label(frame_options, width=34, fg="#696969", bg='#ffffff', font=5, text='Optional features').grid(row=0)


def var_states():
    video_processing.process_video(e.get(), jump.get(), max_fps.get(), bool(output_video.get()))


Label(frame_options, width=50, fg="#696969", bg='#ffffff', text='How many frames to jump during processing?', anchor='w').grid(row=1, padx=10)
jump = Scale(frame_options, from_=0, to=100, orient=HORIZONTAL, bd=0, bg='#ffffff', length=300)
jump.grid(row=2, padx=10, pady=10)
Label(frame_options, width=50, fg="#696969", bg='#ffffff', text='Select the maximum number of fps to process', anchor='w').grid(row=3, padx=10)
max_fps = Scale(frame_options, from_=0, to=100, orient=HORIZONTAL, bg='#ffffff', length=300)
max_fps.grid(row=4, padx=10, pady=10)
Label(frame_options, width=50, fg="#696969", bg='#ffffff', text='Do you want a video in output?', anchor='w').grid(row=5, padx=10)
output_video = IntVar()
Checkbutton(frame_options, text="Output Video", variable=output_video, bg='#ffffff', fg='#696969').grid(row=6, sticky=W, pady=10)

# ------------------------------ ENTER OR QUIT ------------------------------

frame_buttons = Frame(root, bg='#ffffff', bd=5)
frame_buttons.place(relx=0.5, rely=0.73, relwidth=0.9, relheight=0.07, anchor='n')
Button(frame_buttons, text='Start', command=var_states, bg='#ffffff', font=11, fg='#696969', anchor=W).grid(row=0, column=0, padx=3, pady=3)
Button(frame_buttons, text='Quit', command=root.quit, bg='#ff7f7f', font=11, fg='#ffffff', anchor=W).grid(row=0, column=1, padx=3, pady=3)

#frame_return = Frame(root, bg='#ffffff', bd=5)
#frame_return.place(relx=0.5, rely=0.84, relwidth=0.9, relheight=0.07, anchor='n')


root.mainloop()

