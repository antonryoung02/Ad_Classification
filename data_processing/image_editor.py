import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import dotenv
from PIL import Image, ImageTk, ImageFilter
import tkinter as tk
from data_processing.metadata_dataframe import MetadataDataframe

# Run as a script python image_editor.py <'absolute_path_to_data_directory'>
class ImageEditor:
    def __init__(self, data_path:str, metadata_dataframe:MetadataDataframe):
        self.data_path = data_path
        self.metadata_dataframe = metadata_dataframe
        self.image_index = 0
        self.files = [f for f in os.listdir(self.data_path) if os.path.isfile(os.path.join(self.data_path, f))]

        if not self.files:
            raise ValueError(f"No image files found in the directory: {self.data_path}")

        self.current_image = None
        self.display_image = None
        self.zoom_factor = 1.0
        self.blur_radius = 2

        self.root = tk.Tk()
        self.root.title("Image Editor")

        self.root.bind("<Left>", self.prev_key)
        self.root.bind("<Right>", self.next_key)
        self.root.bind("d", self.delete_key)
        self.root.bind("s", self.save_key)
        self.root.bind("<MouseWheel>", self.zoom)
        self.root.bind("<B1-Motion>", self.apply_blur)

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.prev_button = tk.Button(self.root, text="Prev", command=self.prev)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.root, text="Next", command=self.next)
        self.next_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(self.root, text="Save", command=self.save_changes)
        self.save_button.pack(side=tk.LEFT)

        self.delete_button = tk.Button(self.root, text="Delete", command=self.delete)
        self.delete_button.pack(side=tk.LEFT)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.quit)
        self.quit_button.pack(side=tk.LEFT)

        self.load_image()

        self.root.mainloop()

    def load_image(self):
        if self.files:
            file_path = self.get_file_path()
            self.current_image = Image.open(file_path)
            self.display_image = self.current_image.copy()
            self.show_image()
        else:
            self.image_label.config(image='', text="No images found")
            self.image_label.image = None

    def show_image(self):
        if not self.display_image:
            return
        img = self.display_image.resize(
            (int(self.display_image.width * self.zoom_factor),
            int(self.display_image.height * self.zoom_factor)),
            Image.Resampling.LANCZOS
        )
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def prev(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.load_image()

    def next(self):
        if self.image_index < len(self.files) - 1:
            self.image_index += 1
            self.load_image()

    def prev_key(self, event):
        self.prev()

    def next_key(self, event):
        self.next()

    def delete_key(self, event):
        self.delete()
        
    def save_key(self, event):
        self.save_changes()

    def zoom(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        self.zoom_factor *= 1.1
        self.show_image()

    def zoom_out(self):
        self.zoom_factor /= 1.1
        self.show_image()

    def apply_blur(self, event):
        if not self.current_image:
            return
        x = int(event.x / self.image_label.winfo_width() * self.current_image.width)
        y = int(event.y / self.image_label.winfo_height() * self.current_image.height)

        left = max(x - self.blur_radius, 0)
        right = min(x + self.blur_radius, self.current_image.width)
        top = max(y - self.blur_radius, 0)
        bottom = min(y + self.blur_radius, self.current_image.height)

        blurred_region = self.current_image.crop((left, top, right, bottom)).filter(ImageFilter.GaussianBlur(self.blur_radius))
        self.current_image.paste(blurred_region, (left, top))

        self.display_image = self.current_image.copy()
        self.show_image()

    def get_file_path(self):
        return os.path.join(self.data_path, self.files[self.image_index])

    def delete(self):
        file_path = self.get_file_path()
        if os.path.exists(file_path):
            os.remove(file_path)
            self.metadata_dataframe.delete(file_path=file_path)
            self.metadata_dataframe.save()
            self.files.pop(self.image_index)
            if self.image_index >= len(self.files):
                self.image_index = len(self.files) - 1
            if self.files:
                self.load_image()
            else:
                self.current_image = None
                self.image_label.config(image='', text="No images left")
                self.image_label.image = None
        else:
            print("Path in self.files was not found!")

    def save_changes(self):
        file_path = self.get_file_path()
        self.current_image.save(file_path)
        self.next()

    def quit(self):
        self.root.destroy()

if __name__ == "__main__":
    #get data path and metadata path from env not cl args
    data_path = os.getenv("DATASET_DIRECTORY")
    metadata_path =os.getenv("METADATA_PATH")
    meta_df = MetadataDataframe(metadata_path)
    image_editor = ImageEditor(data_path=data_path, metadata_dataframe=meta_df)
