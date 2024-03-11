from customtkinter import *
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import io
from keras.models import load_model
from model_test import predict_single_image
from lime_integration import explain_image

app = CTk()
app.geometry("920x1080")
app.resizable(False, False)
set_appearance_mode("light")

model = load_model("classificationModel.keras")

image_size = (300, 300)

def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        print("Uploaded file path:", file_path)

        image = Image.open(file_path)

        image = image.resize(image_size)

        photo = ImageTk.PhotoImage(image)

        uploaded_image_label = CTkLabel(master=app, text=" ")
        uploaded_image_label.place(relx=0.5, rely=0.4, anchor="center")

        uploaded_image_label.configure(image=photo)
        uploaded_image_label.image = photo

        # Preprocessing and making the predicion
        img = Image.open(file_path)
        img = img.resize((128, 128))
        img = np.asarray(img)
        img = img / 255.0

        # CHECK IF IT ALREADY HAS ALL THE DIMS BEFORE DOING THIS
        if img.shape[-1] == 128:
            img = np.expand_dims(img, axis=-1)
        if img.shape[0] == 128:
            img = np.expand_dims(img, axis=0)

        if img.shape[-1] != 3:
            img = np.repeat(img, 3, axis=3)

        prediction, confidence = predict_single_image(model, img)

        if prediction == "MildDemented":
            prediction = "Mild Demented"
        elif prediction == "NonDemented":
            prediction = "Non-Demented"
        elif prediction == "VeryMildDemented":
            prediction = "Very Mild Demented"
        else:
            prediction = "Moderate Demented"

        # Generating the LIME explanation
        heatmap, explained_image = explain_image(img, prediction)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        heatmap_plot = axes[0].imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
        axes[0].set_title('Heatmap')

        regular_plot = axes[1].imshow(explained_image)
        axes[1].set_title('Explained Image')
        fig.colorbar(heatmap_plot, ax=axes[0])

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(f"Prediction: {prediction}\nConfidence: {round(confidence*100, 4)}%")

        # Convert the matplotlib figure to a Tkinter-compatible format
        fig_photo = matplotlib_to_tk_photo(fig)

        # Create a label to display the figure
        fig_label = CTkLabel(app, image=fig_photo)
        fig_label.image = fig_photo
        fig_label.place(relx=0.5, rely=0.75, anchor="center")

        # Ensure the figure is closed to prevent memory leaks
        plt.close(fig)


# Function to convert matplotlib figure to Tkinter PhotoImage
def matplotlib_to_tk_photo(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return ImageTk.PhotoImage(Image.open(buf))

btn = CTkButton(master=app, text="Upload Image", command=upload_file, corner_radius=32)
btn.place(relx=0.5, rely=0.2, anchor="center")

neurascan_symbol = CTkLabel(master=app, text="NeuraScan", font=("Arial", 36))
neurascan_symbol.place(relx=0.5, rely=0.1, anchor="center")

app.mainloop()
