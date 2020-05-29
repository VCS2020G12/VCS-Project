from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import video_processing

root = Tk()
root.title('Painting Detection and Recognition')
root.geometry("550x680")
#root.resizable(0, 0)

# Center the frame
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2) - 150
positionDown = int(root.winfo_screenheight() / 2 - windowHeight / 2) - 150
root.geometry("+{}+{}".format(positionRight, positionDown))

background_image = PhotoImage(file='images/background.png')
background_label = Label(root, image=background_image)
background_label.place(relwidth=1, relheight=1)


# -------------------------- SELECT PATH DIR/FILE --------------------------
def open_file():
    frame_path.filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("mp4 files", "*.mp4"), ("MOV files", "*.MOV"), ("avi files", "*.avi"), ("All files", "*.*")))
    e.delete(0, END)
    e.insert(0, frame_path.filename)


frame_path = Frame(root, bg='#ffffff', bd=5)
frame_path.place(relx=0.5, rely=0.1, relwidth=0.9, relheight=0.15, anchor='n')

Label(frame_path, width=34, fg="#696969", bg='#ffffff', font=5, text='Type or select the path of the video:', anchor='w').place(relx=0.02, rely=0.15, relwidth=0.96, relheight=0.3)
e = Entry(frame_path, fg="#696969", bg='#ffffff', font=9)
e.place(relx=0.02, rely=0.55, relwidth=0.84, relheight=0.30)
e.insert(END, "Enter the path")
Button(frame_path, text=" ... ", command=open_file, bg='#ffffff').place(relx=0.88, rely=0.55, relwidth=0.1, relheight=0.30)

# ----------------------------- SELECT OPTIONS -----------------------------
frame_options = Frame(root, bg='#ffffff', bd=5)
frame_options.place(relx=0.5, rely=0.29, relwidth=0.9, relheight=0.47, anchor='n')
Label(frame_options, fg="#696969", bg='#ffffff', font=5, text='Optional features').place(relx=0.02, rely=0.03, relwidth=0.96, relheight=0.1)


def var_states():
    video_processing.process_video(e.get(), jump.get(), max_fps.get(), bool(output_video.get()))


Label(frame_options, fg="#696969", bg='#ffffff', text='How many frames to jump during processing?', anchor='w').place(relx=0.02, rely=0.16, relwidth=0.96, relheight=0.1)
jump = Scale(frame_options, from_=0, to=100, orient=HORIZONTAL, bd=0, bg='#ffffff')
jump.place(relx=0.02, rely=0.26, relwidth=0.96, relheight=0.12)
Label(frame_options, width=50, fg="#696969", bg='#ffffff', text='Select the maximum number of fps to process', anchor='w').place(relx=0.02, rely=0.41, relwidth=0.96, relheight=0.1)
max_fps = Scale(frame_options, from_=0, to=100, orient=HORIZONTAL, bg='#ffffff')
max_fps.place(relx=0.02, rely=0.51, relwidth=0.96, relheight=0.12)
Label(frame_options, width=50, fg="#696969", bg='#ffffff', text='Do you want a video in output?', anchor='w').place(relx=0.02, rely=0.66, relwidth=0.96, relheight=0.1)
output_video = IntVar()
Checkbutton(frame_options, text="Output Video", variable=output_video, bg='#ffffff', fg='#696969', anchor='w').place(relx=0.02, rely=0.76, relwidth=0.96, relheight=0.1)

# ------------------------------ ENTER OR QUIT ------------------------------

frame_buttons = Frame(root, bg='#ffffff', bd=5)
frame_buttons.place(relx=0.5, rely=0.8, relwidth=0.4, relheight=0.1, anchor='n')
Button(frame_buttons, text='Start', command=var_states, bg='#ffffff', font=11, fg='#696969').place(relx=0.05, rely=0.2, relwidth=0.4, relheight=0.6)
Button(frame_buttons, text='Quit', command=root.quit, bg='#ff7f7f', font=11, fg='#ffffff').place(relx=0.55, rely=0.2, relwidth=0.4, relheight=0.6)

root.mainloop()

