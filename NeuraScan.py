import io
import time
from pathlib import Path
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label
from PIL import Image, ImageTk
from model_test import predict_single_image
from keras.models import load_model
from lime_integration import explain_image
import matplotlib.pyplot as plt
import numpy as np

# OUTPUT_PATH = Path(__file__).parent
# ASSETS_PATH = OUTPUT_PATH / Path(r"frame0")
#
# def relative_to_assets(path: str) -> Path:
#     return ASSETS_PATH / Path(path)

def relative_to_assets(img_file):
    return f'frame0/{img_file}'


window = Tk()

window.geometry("1920x1080")
window.configure(bg = "#2A2A2A")

# Setting icon photo, taskbar photo and title
window.title('NeuraScan')

ico = Image.open('frame0/Artificial Intelligence.png')
icon_photo = ImageTk.PhotoImage(ico)
window.wm_iconphoto(False, icon_photo)

canvas = Canvas(
    window,
    bg = "#2A2A2A",
    height = 1080,
    width = 1920,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)

# GREY BACKGROUND SQUARE 1
image_image_14 = PhotoImage(
    file=relative_to_assets("GreyDecorSq1.png"))
image_14 = canvas.create_image(
    1550.0,
    750.0,
    image=image_image_14
)

# GREY BACKGROUND SQUARE 2
image_image_15 = PhotoImage(
    file=relative_to_assets("GreyDecorSq2.png"))
image_15 = canvas.create_image(
    1580.0,
    820.0,
    image=image_image_15
)

# WHITE DESCRIPTION BOX
canvas.create_rectangle(
    0.0,
    108.0,
    272.0,
    1123.0,
    fill="#FFFFFF",
    outline="")

# WHAT IS NEURASCAN TEXT
image_image_2 = PhotoImage(
    file=relative_to_assets("What is NeuraScan_.png"))
image_2 = canvas.create_image(
    120.0,
    180.0,
    image=image_image_2
)

# NEURASCAN DESCRIPTION
image_image_1 = PhotoImage(
    file=relative_to_assets("Description.png")
)
image_1 = canvas.create_image(
    130.0,
    600.0,
    image=image_image_1
)

# BLUE HEADER RECTANGLE
canvas.create_rectangle(
    0.0,
    0.0,
    1920.0,
    127.0,
    fill="#00A3FF",
    outline="")

# NEURASCAN LOGO
image_image_4 = PhotoImage(
    file=relative_to_assets("NeuraScan.png"))
image_4 = canvas.create_image(
    350.0,
    65.0,
    image=image_image_4
)

# NEURASCAN SYMBOL
image_image_5 = PhotoImage(
    file=relative_to_assets("Artificial Intelligence.png"))
image_5 = canvas.create_image(
    130.0,
    65.0,
    image=image_image_5
)

# LEFT OUTPUT RECTANGLE
image_image_6 = PhotoImage(
    file=relative_to_assets("OutputRight.png"))
image_6 = canvas.create_image(
    670.0,
    700.0,
    image=image_image_6
)

# RIGHT OUTPUT RECTANGLE
image_image_7 = PhotoImage(
    file=relative_to_assets("OutputRight.png"))
image_7 = canvas.create_image(
    1450.0,
    700.0,
    image=image_image_7
)

# PREDICTION IS TEXT
image_image_3 = PhotoImage(
    file=relative_to_assets("Prediction is_.png"))
image_3 = canvas.create_image(
    510.0,
    375.0,
    image=image_image_3
)

# HEADER DECORATIONS
# BRAIN
image_image_10 = PhotoImage(
    file=relative_to_assets("Brain.png"))
image_10 = canvas.create_image(
    1480.0,
    70.0,
    image=image_image_10
)

# SECOND BRAIN
image_image_11 = PhotoImage(
    file=relative_to_assets("Critical Thinking.png"))
image_11 = canvas.create_image(
    1580.0,
    55.0,
    image=image_image_11
)

# ROBOT
image_image_12 = PhotoImage(
    file=relative_to_assets("Bot.png"))
image_12 = canvas.create_image(
    1680.0,
    65.0,
    image=image_image_12
)

# MACHINE LEARNING
image_image_13 = PhotoImage(
    file=relative_to_assets("Machine Learning.png"))
image_13 = canvas.create_image(
    1800.0,
    70.0,
    image=image_image_13
)
# UPLOAD SCAN BUTTON
image_image_8 = PhotoImage(
    file=relative_to_assets("ScanButton.png"))
image_8 = canvas.create_image(
    1010.0,
    240.0,
    image=image_image_8
)

# SCAN BUTTON HOVER
image_image_16 = PhotoImage(
    file=relative_to_assets("ScanButtonHover.png"))
image_16 = canvas.create_image(
    1010.0,
    240.0,
    image=image_image_16
)


# UPLOAD SCAN TEXT
image_image_9 = PhotoImage(
    file=relative_to_assets("Upload Scan.png"))
image_9  = canvas.create_image(
    1010.0,
    240.0,
    image=image_image_9
)

# ------------FUNCTIONALITY------------#

model = load_model('classificationModel.keras')

def matplotlib_to_tk_photo(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf))

def please_wait_text():
    # If there is already a prediction label on screen, delete it
    prediction_labels = canvas.find_withtag("prediction_label")
    for label in prediction_labels:
        canvas.delete(label)

    canvas.create_text(650.0, 377.0, text="This may take a moment...",
                       font=("Arial", 28), fill="white", tags="prediction_label", anchor="w")
    window.update()
    time.sleep(0.5)

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        please_wait_text()
        print("Uploaded file path:", file_path)

        prediction, confidence, img = predict_single_image(model, file_path)

        if prediction == "MildDemented":
            prediction = "Mild Demented"
        elif prediction == "NonDemented":
            prediction = "Non-Demented"
        elif prediction == "VeryMildDemented":
            prediction = "Very Mild Demented"
        else:
            prediction = "Moderate Demented"

        print(f"Prediction is: {prediction}\nConfidence is: {round(confidence*100, 4)}%")

        prediction_labels = canvas.find_withtag("prediction_label")
        for label in prediction_labels:
            canvas.delete(label)

        canvas.create_text(650.0, 377.0, text=f"{prediction} {round(confidence*100, 4)}%",
                           font=("Arial", 28), fill="white", tags="prediction_label", anchor="w")

        heatmap, explained_image = explain_image(img, prediction)
        # Create the heatmap figure
        fig_heatmap, ax_heatmap = plt.subplots(figsize=(5, 5))
        heatmap_plot = ax_heatmap.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        ax_heatmap.set_title('Heatmap')
        fig_heatmap.colorbar(heatmap_plot, ax=ax_heatmap)
        ax_heatmap.axis('off')

        # Create the explained image figure
        fig_explained, ax_explained = plt.subplots(figsize=(5, 5))
        explained_plot = ax_explained.imshow(explained_image)
        ax_explained.set_title('Explained Image')
        ax_explained.axis('off')

        # Convert the matplotlib figures to Tkinter-compatible format
        fig_photo_heatmap = matplotlib_to_tk_photo(fig_heatmap)
        fig_photo_explained = matplotlib_to_tk_photo(fig_explained)

        # Create labels to display the figures
        fig_label_heatmap = Label(window, image=fig_photo_heatmap)
        fig_label_heatmap.image = fig_photo_heatmap
        fig_label_heatmap.place(relx=0.345, rely=0.645, anchor="center")

        fig_label_explained = Label(window, image=fig_photo_explained)
        fig_label_explained.image = fig_photo_explained
        fig_label_explained.place(relx=0.755, rely=0.645, anchor="center")

        # Ensure the figures are closed to prevent memory leaks
        plt.close(fig_heatmap)
        plt.close(fig_explained)


button_pressed_yet = False
if button_pressed_yet is False:
    canvas.itemconfig(image_8, state='normal')
    canvas.itemconfig(image_16, state='hidden')

def button_pressed(event):
    canvas.itemconfig(image_8, state='hidden')
    canvas.itemconfig(image_16, state='normal')

def button_released(event):
    canvas.itemconfig(image_8, state='normal')
    canvas.itemconfig(image_16, state='hidden')
    upload_file()

invisible_button = Button(window, image=image_image_9, bg="#00A3FF", activebackground="#49BEFF",
                          borderwidth=0, width=710, height=70)

invisible_button.bind("<ButtonPress-1>", button_pressed)
invisible_button.bind("<ButtonRelease-1>", button_released)

# Place the button on the canvas
canvas.create_window(1010.0, 240.0, window=invisible_button)


window.resizable(False, False)
window.mainloop()
