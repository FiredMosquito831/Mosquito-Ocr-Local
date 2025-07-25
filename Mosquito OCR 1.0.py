import easyocr
import os.path
import cv2
import numpy as np
import torch
from PIL import ImageGrab, Image
import os
import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk

# Global vars

# GUI THEMES
THEMES = {
    "dark": {
        "bg": "#2e2e2e",          # Main window background
        "frame_bg": "#4a4a4a",     # LabelFrame background
        "button_bg": "#555555",    # Standard button background
        "button_fg": "white",      # Standard button text
        "highlight_button_bg": "#808080", # Highlighted button background (hover)
        "text_bg": "#1e1e1e",      # Text widget background
        "text_fg": "white",        # Text widget text
        "listbox_bg": "#3e3e3e",   # Listbox background
        "listbox_fg": "white",     # Listbox text
        "label_fg": "white",       # Label text
        "combobox_bg": "#3e3e3e",  # Combobox background
        "combobox_fg": "white",    # Combobox text
        "combobox_field_bg": "#1e1e1e", # Combobox entry field background
        "combobox_field_fg": "white",   # Combobox entry field text
    },
    "light": {
        "bg": "#ffffff",
        "frame_bg": "#f0f0f0",
        "button_bg": "#d9d9d9",
        "button_fg": "black",
        "highlight_button_bg": "#bdbdbd", # Slightly darker for hover
        "text_bg": "white",
        "text_fg": "black",
        "listbox_bg": "white",
        "listbox_fg": "black",
        "label_fg": "black",
        "combobox_bg": "white",
        "combobox_fg": "black",
        "combobox_field_bg": "white",
        "combobox_field_fg": "black",
    }
}

# language dictionary
EASYOCR_LANGUAGES = {
    "Abaza": "abq",
    "Adyghe": "ady",
    "Afrikaans": "af",
    "Angika": "ang",
    "Arabic": "ar",
    "Assamese": "as",
    "Azerbaijani": "az",
    "Belarusian": "be",
    "Bulgarian": "bg",
    "Bengali": "bn",
    "Bosnian": "bs",
    "Catalan": "ca",
    "Cebuano": "ceb",
    "Czech": "cs",
    "Welsh": "cy",
    "Danish": "da",
    "German": "de",
    "Greek": "el",
    "English": "en",
    "Spanish": "es",
    "Estonian": "et",
    "Persian (Farsi)": "fa",
    "Finnish": "fi",
    "French": "fr",
    "Irish": "ga",
    "Goan Konkani": "gom",
    "Hindi": "hi",
    "Croatian": "hr",
    "Hungarian": "hu",
    "Armenian": "hy",
    "Indonesian": "id",
    "Icelandic": "is",
    "Italian": "it",
    "Japanese": "ja",
    "Javanese": "jv",
    "Georgian": "ka",
    "Kazakh": "kk",
    "Central Khmer": "km",
    "Kannada": "kn",
    "Korean": "ko",
    "Kurdish": "ku",
    "Kyrgyz": "ky",
    "Latin": "la",
    "Lao": "lo",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Macedonian": "mk",
    "Malayalam": "ml",
    "Mongolian": "mn",
    "Marathi": "mr",
    "Malay": "ms",
    "Burmese": "my",
    "Nepali": "ne",
    "Dutch": "nl",
    "Norwegian": "no",
    "Occitan": "oc",
    "Punjabi": "pa",
    "Polish": "pl",
    "Pashto": "ps",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Sanskrit": "sa",
    "Sinhala": "si",
    "Slovak": "sk",
    "Slovenian": "sl",
    "Albanian": "sq",
    "Serbian": "sr",
    "Swedish": "sv",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Thai": "th",
    "Tagalog": "tl",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Urdu": "ur",
    "Uzbek": "uz",
    "Vietnamese": "vi",
    "Chinese (Simplified)": "ch_sim",
    "Chinese (Traditional)": "ch_tra",
    # Add more languages as needed from EasyOCR's supported list
}

# Check for cuda requires cuda to be installed and in the dependencies specific cuda version of torch
# pip install torch  --index-url https://download.pytorch.org/whl/cu128      change based on your version of cuda, although they are backwards compatible
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Computing on {device}. (CUDA)")
    easyocr_gpu_arg = True
# Check for MPS (Apple Silicon GPU) - Requires PyTorch 1.12+ and macOS 12.3+
elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Computing on {device}. (MPS)")
    easyocr_gpu_arg = True
else:
    device = torch.device("cpu")
    print(f"Computing on {device}.")
    easyocr_gpu_arg = False



reader = None
selected_ocr_language_codes = ['en', 'ro'] # Default languages preferences modify to your liking


def extract_text_from_image(image_path):
    try:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

        # Perform OCR on the image
        results = reader.readtext(image)

        # Sort results based on the vertical (Y) position first, then horizontal (X) position
        sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))

        # Prepare a string to hold the extracted text
        formatted_text = ""
        for (_, text, _) in sorted_results:
            formatted_text += text + "\n"

        return formatted_text.strip() if formatted_text else "No text detected."
    except Exception as e:
        return f"Error: {str(e)}"

def OcrLocalImageV3(): # Renamed for clarity
    """
    Extracts text from a locally selected image file using OCR,
    aiming for better accuracy and formatting by reusing logic from OcrClipboardV4.
    """
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tiff;*.webp")]
    )

    if not file_path: # Check if a file was actually selected
        print("No image file selected.")
        # messagebox.showinfo("OCR", "No image file selected.") # Uncomment if using in GUI
        return "No image file selected."

    try:
        # --- Load and Preprocess the Image ---
        # Use PIL to open the image, which handles various formats robustly
        pil_image = Image.open(file_path)
        print(f"Loaded image: {file_path} (Mode: {pil_image.mode})")

        # Ensure the image is in a mode compatible with conversion to NumPy/OpenCV
        if pil_image.mode not in ('RGB', 'RGBA', 'L'):
            print(f"Image mode '{pil_image.mode}' detected. Converting to RGB.")
            pil_image = pil_image.convert('RGB')

        # Convert the PIL image to a NumPy array
        numpy_image = np.array(pil_image)
        print(f"Converted PIL image to NumPy array with shape: {numpy_image.shape} and dtype: {numpy_image.dtype}")

        # Convert the NumPy array (RGB format from PIL) to grayscale for OCR
        # Handle different numbers of channels for grayscale conversion
        if len(numpy_image.shape) == 3: # Color image (RGB or RGBA)
            # Convert RGB/RGBA to BGR first (standard OpenCV format if further processing needed)
            # But for grayscale conversion, we can go directly or via RGB
            # Let's convert to grayscale directly using OpenCV's standard method
            # OpenCV expects BGR, but PIL provides RGB. Let's be explicit.
            if numpy_image.shape[2] == 3: # RGB
                 # Convert PIL RGB to OpenCV BGR if needed for other ops, but for grayscale:
                 # Alternative direct grayscale: gray = np.dot(numpy_image[...,:3], [0.2989, 0.5870, 0.1140])
                 # Or simply use cvtColor correctly
                 rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) # If BGR needed later
                 gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY) # Convert BGR to Grayscale
            elif numpy_image.shape[2] == 4: # RGBA
                # Blend with white background for OCR (similar to OcrClipboardV4)
                r, g, b, a = cv2.split(numpy_image)
                alpha_float = a.astype(np.float32) / 255.0
                white_bg = np.ones_like(r, dtype=np.uint8) * 255
                blended_r = (r * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                blended_g = (g * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                blended_b = (b * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                blended_rgb = cv2.merge([blended_r, blended_g, blended_b])
                rgb_image = blended_rgb # Now RGB without alpha
                gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) # Convert RGB to Grayscale
            else:
                raise ValueError(f"Unexpected number of channels ({numpy_image.shape[2]}) in color image.")

        elif len(numpy_image.shape) == 2: # Grayscale image
            gray = numpy_image # Already grayscale
        else:
            raise ValueError(f"Unexpected NumPy array shape: {numpy_image.shape}")

        # Apply a mild Gaussian blur to reduce noise (optional, often helpful)
        # ocr_image = cv2.GaussianBlur(gray, (3, 3), 0)
        # Use the grayscale image for OCR
        ocr_image = gray

    except Exception as e:
        error_msg = f"Error loading or preprocessing image '{file_path}': {e}"
        print(error_msg)
        # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
        return error_msg

    try:
        # --- Perform OCR ---
        # Perform OCR on the preprocessed grayscale image
        # detail=1 ensures bounding boxes are returned (default)
        results = reader.readtext(ocr_image, detail=1)

        if not results:
            # Update GUI text box
            input_text.delete(1.0, tk.END)
            input_text.insert(tk.END, "No text detected by OCR in the selected image.")
            return "No text detected."

        # --- Improved Text Grouping and Formatting (Reusing logic from OcrClipboardV4) ---

        # 1. Estimate average character width to improve spacing
        avg_char_width = estimate_average_char_width(results)
        if avg_char_width <= 0:
            avg_char_width = 10 # Fallback default if estimation fails
        space_threshold = avg_char_width * 0.7 # Threshold for adding a space (adjustable factor)

        # 2. Group detections into lines based on Y overlap/ proximity
        lines = group_ocr_results_into_lines(results)

        # 3. Sort lines by their average Y-coordinate
        lines.sort(key=lambda line: np.mean([word_info[0][0][1] for word_info in line]) if line else 0)

        # 4. Build the final formatted text
        formatted_text_lines = []
        for line in lines:
            # Sort words in the line by their X-coordinate
            line.sort(key=lambda word_info: word_info[0][0][0]) # Sort by top-left X

            line_text_parts = []
            last_x_end = None

            for (bbox, text, confidence) in line:
                # Get bounding box coordinates
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                current_x_start = top_left[0]
                current_x_end = bottom_right[0]

                if last_x_end is not None:
                    # Calculate the gap between the end of the last word and the start of the current word
                    gap = current_x_start - last_x_end
                    if gap > space_threshold:
                        # Estimate number of spaces based on the gap and average character width
                        num_spaces = max(1, int(round(gap / avg_char_width)))
                        line_text_parts.append(" " * num_spaces)
                # Add the current word's text
                line_text_parts.append(text)
                # Update the end position for the next iteration
                last_x_end = current_x_end

            # Join parts for the line
            formatted_text_lines.append("".join(line_text_parts))

        final_text = "\n".join(formatted_text_lines)

        # Update GUI text box
        input_text.delete(1.0, tk.END)
        input_text.insert(tk.END, final_text)

        return final_text.strip() if final_text else "No text detected after formatting."

    except Exception as e:
        error_msg = f"Error during OCR processing of '{file_path}': {e}"
        print(error_msg)
        # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
        return error_msg



def OcrClipboardV4():
    """
    Extracts text from an image in the clipboard using OCR,
    aiming for better accuracy and formatting.
    Handles cases where clipboard contains files or multiple items.
    """
    # Grab the content from the clipboard
    clipboard_content = ImageGrab.grabclipboard()
    pil_image = None

    # Check the type of content returned
    if clipboard_content is None:
        print("Clipboard is empty or content type not recognized for image grabbing.")
        # messagebox.showinfo("OCR", "Clipboard is empty or content not recognized.") # Uncomment if using in GUI
        return "Clipboard is empty or content not recognized."

    elif isinstance(clipboard_content, list):
        # ImageGrab.grabclipboard() returns a list of filenames if files are copied
        print(f"Clipboard contains a list (likely file paths): {clipboard_content}")
        if clipboard_content: # Check if the list is not empty
            # Try to load the first file path as an image
            first_item = clipboard_content[0]
            if isinstance(first_item, str) and os.path.isfile(first_item):
                try:
                    print(f"Attempting to load image from file path: {first_item}")
                    pil_image = Image.open(first_item)
                    # If it's not a common mode, convert it
                    if pil_image.mode not in ('RGB', 'RGBA', 'L'):
                         print(f"Loaded image mode is '{pil_image.mode}'. Converting to RGB.")
                         pil_image = pil_image.convert('RGB')
                except Exception as e:
                    error_msg = f"Error opening file '{first_item}' as image: {e}"
                    print(error_msg)
                    # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
                    return error_msg
            else:
                print("First item in clipboard list is not a valid file path.")
                # messagebox.showinfo("OCR", "Copied item is not a valid image file path.") # Uncomment if using in GUI
                return "Copied item is not a valid image file path."
        else:
            print("Clipboard list is empty.")
            # messagebox.showinfo("OCR", "Clipboard list is empty.") # Uncomment if using in GUI
            return "Clipboard list is empty."

    elif isinstance(clipboard_content, Image.Image):
        # Standard case: a PIL Image object was grabbed directly
        print("Direct image grabbed from clipboard.")
        pil_image = clipboard_content
        # Ensure the image is in a compatible mode
        if pil_image.mode not in ('RGB', 'RGBA', 'L'):
            print(f"Clipboard image mode is '{pil_image.mode}'. Converting to RGB.")
            try:
                pil_image = pil_image.convert('RGB')
            except Exception as conv_e:
                error_msg = f"Error converting image from mode '{pil_image.mode}' to RGB: {conv_e}"
                print(error_msg)
                # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
                return error_msg
    else:
        # Unexpected type returned
        error_msg = f"Unexpected clipboard content type: {type(clipboard_content)}"
        print(error_msg)
        # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
        return error_msg

    # --- Proceed with OCR if pil_image is successfully loaded ---
    if pil_image is not None:
        try:
            # Convert the PIL image to a NumPy array
            numpy_image = np.array(pil_image)
            print(f"Converted PIL image to NumPy array with shape: {numpy_image.shape} and dtype: {numpy_image.dtype}")

            # Convert the NumPy array (RGB format from PIL) to BGR format for OpenCV (if needed)
            # Or keep as grayscale/RGB based on the number of channels for OCR
            if len(numpy_image.shape) == 3: # Color image (RGB or RGBA)
                if numpy_image.shape[2] == 3: # RGB
                    image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
                elif numpy_image.shape[2] == 4: # RGBA
                    # Blend with white background for OCR
                    r, g, b, a = cv2.split(numpy_image)
                    alpha_float = a.astype(np.float32) / 255.0
                    white_bg = np.ones_like(r, dtype=np.uint8) * 255
                    blended_r = (r * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                    blended_g = (g * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                    blended_b = (b * alpha_float + white_bg * (1 - alpha_float)).astype(np.uint8)
                    blended_rgb = cv2.merge([blended_r, blended_g, blended_b])
                    image = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR) # Convert to BGR if passing to OpenCV functions later
                    # For OCR, we'll convert to grayscale below anyway, so BGR conversion might be skipped
                else:
                    raise ValueError(f"Unexpected number of channels ({numpy_image.shape[2]}) in NumPy array.")
            elif len(numpy_image.shape) == 2: # Grayscale image
                image = numpy_image # Use grayscale directly
            else:
                raise ValueError(f"Unexpected NumPy array shape: {numpy_image.shape}")

        except Exception as e:
            error_msg = f"Error converting PIL image to OpenCV format: {e}"
            print(error_msg)
            # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
            return error_msg

        try:
            # --- Preprocessing for OCR ---
            # Determine if the image is already grayscale or convert it
            if len(image.shape) == 3 and image.shape[2] == 3:
                # It's a BGR image, convert to grayscale for OCR
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                 # Single channel, treat as grayscale
                gray = image.squeeze()
            elif len(image.shape) == 2:
                # Already grayscale
                gray = image
            else:
                # Unexpected format, try converting to grayscale anyway
                print(f"Unexpected image format for grayscale conversion: {image.shape}. Attempting conversion.")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # This assumes BGR if 3 channels, might fail otherwise

            # Apply a mild Gaussian blur to reduce noise (optional, often helpful)
            # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            # Use the grayscale image for OCR
            ocr_image = gray

            # Perform OCR on the preprocessed image
            # detail=1 ensures bounding boxes are returned (default)
            # You might experiment with adding `paragraph=True` if the text has clear paragraph structures,
            # though it might interfere with custom line grouping.
            results = reader.readtext(ocr_image, detail=1)

            if not results:
                 # Update GUI text box
                 input_text.delete(1.0, tk.END)
                 input_text.insert(tk.END, "No text detected by OCR.")
                 return "No text detected."

            # --- Improved Text Grouping and Formatting ---

            # 1. Estimate average character width to improve spacing
            avg_char_width = estimate_average_char_width(results)
            if avg_char_width <= 0:
                avg_char_width = 10 # Fallback default if estimation fails
            space_threshold = avg_char_width * 0.7 # Threshold for adding a space (adjustable factor)

            # 2. Group detections into lines based on Y overlap/ proximity
            lines = group_ocr_results_into_lines(results)

            # 3. Sort lines by their average Y-coordinate
            lines.sort(key=lambda line: np.mean([word_info[0][0][1] for word_info in line]) if line else 0)

            # 4. Build the final formatted text
            formatted_text_lines = []
            for line in lines:
                # Sort words in the line by their X-coordinate
                line.sort(key=lambda word_info: word_info[0][0][0]) # Sort by top-left X

                line_text_parts = []
                last_x_end = None

                for (bbox, text, confidence) in line:
                    # Get bounding box coordinates
                    top_left = tuple(map(int, bbox[0]))
                    bottom_right = tuple(map(int, bbox[2]))
                    current_x_start = top_left[0]
                    current_x_end = bottom_right[0]

                    if last_x_end is not None:
                        # Calculate the gap between the end of the last word and the start of the current word
                        gap = current_x_start - last_x_end
                        if gap > space_threshold:
                            # Estimate number of spaces based on the gap and average character width
                            num_spaces = max(1, int(round(gap / avg_char_width)))
                            line_text_parts.append(" " * num_spaces)
                    # Add the current word's text
                    line_text_parts.append(text)
                    # Update the end position for the next iteration
                    last_x_end = current_x_end

                # Join parts for the line
                formatted_text_lines.append("".join(line_text_parts))

            final_text = "\n".join(formatted_text_lines)

            # Update GUI text box
            input_text.delete(1.0, tk.END)
            input_text.insert(tk.END, final_text)

            return final_text.strip() if final_text else "No text detected after formatting."

        except Exception as e:
            error_msg = f"Error during OCR processing: {e}"
            print(error_msg)
            # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
            return error_msg

    else:
        # This case should ideally be covered by the checks above, but as a fallback:
        error_msg = "Failed to load image from clipboard content."
        print(error_msg)
        # messagebox.showerror("OCR Error", error_msg) # Uncomment if using in GUI
        return error_msg


# --- Helper functions remain the same ---
def estimate_average_char_width(ocr_results):
    """Estimates the average character width from OCR results."""
    total_width = 0
    total_chars = 0
    for (bbox, text, confidence) in ocr_results:
        if text.strip(): # Only consider non-empty text
            # Bounding box: [top_left, top_right, bottom_right, bottom_left]
            top_left = tuple(map(int, bbox[0]))
            top_right = tuple(map(int, bbox[1]))
            # Calculate width of the bounding box
            width = top_right[0] - top_left[0]
            if width > 0: # Avoid division by zero or negative widths
                num_chars = len(text.strip())
                if num_chars > 0:
                    avg_width_for_word = width / num_chars
                    total_width += avg_width_for_word
                    total_chars += 1
    if total_chars > 0:
        return total_width / total_chars
    else:
        return 0 # Return 0 if unable to estimate

def group_ocr_results_into_lines(ocr_results, vertical_threshold_factor=0.5):
    """
    Groups OCR results (words) into lines based on vertical proximity.
    Returns a list of lines, where each line is a list of (bbox, text, confidence) tuples.
    """
    if not ocr_results:
        return []

    # Sort results by top-left Y coordinate primarily, then X
    sorted_results = sorted(ocr_results, key=lambda x: (x[0][0][1], x[0][0][0]))

    lines = []
    current_line = []

    for result in sorted_results:
        bbox, text, confidence = result
        # Calculate the vertical center of the current word's bounding box
        top_left_y = bbox[0][1]
        bottom_left_y = bbox[3][1]
        current_center_y = (top_left_y + bottom_left_y) / 2

        is_new_line = True
        if current_line:
            # Get the last word in the current line being formed
            last_bbox, _, _ = current_line[-1]
            last_top_left_y = last_bbox[0][1]
            last_bottom_left_y = last_bbox[3][1]
            last_center_y = (last_top_left_y + last_bottom_left_y) / 2

            # Estimate line height based on the last word
            last_height = last_bottom_left_y - last_top_left_y
            # Use a factor of the height as the threshold for vertical grouping
            vertical_threshold = last_height * vertical_threshold_factor

            # Check if the current word's center is within the threshold of the last word's center
            if abs(current_center_y - last_center_y) <= vertical_threshold:
                is_new_line = False

        if is_new_line:
            if current_line: # Save the previous line if it exists
                lines.append(current_line)
            current_line = [result] # Start a new line with the current word
        else:
            current_line.append(result) # Add word to the current line

    # Don't forget to add the last line
    if current_line:
        lines.append(current_line)

    return lines


def initialize_ocr_reader(language_codes):
    """Initializes the global EasyOCR reader with the specified language codes and device."""
    global reader, easyocr_gpu_arg
    try:
        print(f"Initializing EasyOCR Reader with language codes: {language_codes} and GPU: {easyocr_gpu_arg}")
        reader = easyocr.Reader(language_codes, gpu=easyocr_gpu_arg)
        print("EasyOCR Reader initialized successfully.")
        # messagebox.showinfo("OCR Reader", f"OCR Reader initialized with languages: {', '.join(language_codes)}")
    except Exception as e:
        error_msg = f"Failed to initialize OCR Reader: {e}"
        print(error_msg)
        messagebox.showerror("OCR Error", error_msg)
        reader = None # Ensure reader is None on failure

def get_full_name_from_code(code):
    """Gets the full language name from the code."""
    for name, c in EASYOCR_LANGUAGES.items():
        if c == code:
            return name
    return code # Return code if name not found

def get_code_from_full_name(name):
    """Gets the language code from the full name."""
    return EASYOCR_LANGUAGES.get(name)

def add_language():
    """Adds a language selected from the full names to the selected list and listbox."""
    # Get the selected item from the OptionMenu (full name)
    selected_full_name = selected_language_var.get()
    if selected_full_name and selected_full_name in EASYOCR_LANGUAGES:
        lang_code = EASYOCR_LANGUAGES[selected_full_name]
        if lang_code not in selected_ocr_language_codes:
            selected_ocr_language_codes.append(lang_code)
            # Insert the full name into the listbox for display
            lang_listbox.insert(tk.END, selected_full_name)
            # Reset the OptionMenu selection
            selected_language_var.set("Select Language")
            print(f"Added language: {selected_full_name} ({lang_code})")
        else:
            messagebox.showwarning("Add Language", f"Language '{selected_full_name}' is already selected.")
    else:
         messagebox.showwarning("Add Language", "Please select a valid language from the dropdown.")

def remove_language():
    """Removes the selected language (displayed by full name) from the listbox and the selected codes list."""
    try:
        # Get the index of the currently selected item in the listbox
        selected_index = lang_listbox.curselection()[0]
        # Get the full name displayed in the listbox
        removed_full_name = lang_listbox.get(selected_index)
        # Find the corresponding code
        removed_code = get_code_from_full_name(removed_full_name)
        if removed_code:
            selected_ocr_language_codes.remove(removed_code)
            lang_listbox.delete(selected_index)
            print(f"Removed language: {removed_full_name} ({removed_code})")
        else:
             # Shouldn't happen if listbox is managed correctly
            raise ValueError(f"Code not found for name: {removed_full_name}")
    except IndexError:
        # No item selected in listbox
        messagebox.showwarning("Remove Language", "Please select a language to remove from the list.")
    except ValueError as e:
        messagebox.showerror("Remove Language Error", str(e))

def update_reader():
    """Re-initializes the OCR reader with the current selected language codes."""
    if not selected_ocr_language_codes:
        messagebox.showerror("Update Reader", "Please select at least one language.")
        return
    initialize_ocr_reader(selected_ocr_language_codes)

def show_easyocr_languages():
    """Displays the list of supported EasyOCR languages and codes."""
    lang_list_text = "Supported EasyOCR Languages:\n"
    # Sort languages alphabetically by full name for display
    sorted_languages = sorted(EASYOCR_LANGUAGES.items())
    for full_name, code in sorted_languages:
        lang_list_text += f"{full_name} - {code}\n"

    # Create a new Toplevel window to display the list
    lang_window = tk.Toplevel() # Use Toplevel for a new window
    lang_window.title("EasyOCR Language Codes")
    lang_window.geometry("500x400")

    # ScrolledText widget to display the list
    lang_display = scrolledtext.ScrolledText(lang_window, wrap=tk.WORD, width=60, height=25)
    lang_display.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    lang_display.insert(tk.END, lang_list_text)
    lang_display.config(state=tk.DISABLED) # Make it read-only

def apply_theme(widget, theme, is_toplevel=False):
    """Recursively applies theme colors to a widget and its children."""
    theme_colors = THEMES[theme]

    # --- Configure widget based on its type ---
    if isinstance(widget, tk.Tk) or isinstance(widget, tk.Toplevel):
        widget.configure(bg=theme_colors["bg"])
    elif isinstance(widget, tk.Frame) or isinstance(widget, tk.LabelFrame):
        widget.configure(bg=theme_colors["frame_bg"])
    elif isinstance(widget, tk.Button):
        widget.configure(
            bg=theme_colors["button_bg"],
            fg=theme_colors["button_fg"],
            activebackground=theme_colors["highlight_button_bg"], # Optional: Change color on click
            activeforeground=theme_colors["button_fg"]
        )
    elif isinstance(widget, tk.Label):
        widget.configure(bg=theme_colors["frame_bg"], fg=theme_colors["label_fg"])
    elif isinstance(widget, scrolledtext.ScrolledText):
        # Note: Changing bg/fg of ScrolledText might not work perfectly on all systems
        # due to the underlying Text and Scrollbar widgets.
        widget.configure(bg=theme_colors["text_bg"], fg=theme_colors["text_fg"])
        # Try to configure the internal Text and Scrollbar if accessible (advanced)
        # This is often not necessary or fully effective.
    elif isinstance(widget, tk.Listbox):
        widget.configure(bg=theme_colors["listbox_bg"], fg=theme_colors["listbox_fg"])
    elif isinstance(widget, ttk.Combobox):
         # Styling ttk widgets is more complex and usually done via styles.
         # Basic color changes might work for some elements.
         # This is a simplified attempt.
         # A more robust way involves creating ttk.Style objects.
         try:
             # This might change the dropdown list colors (not always reliable)
             widget.configure(foreground=theme_colors["combobox_fg"])
             # Background of the entry field
             widget.configure(background=theme_colors["combobox_field_bg"])
             # Text color inside the entry field
             widget.configure(foreground=theme_colors["combobox_field_fg"]) # Might be overridden
             # Select background/foreground when text is selected
             widget.configure(selectbackground=theme_colors["button_bg"])
             widget.configure(selectforeground=theme_colors["button_fg"])
         except tk.TclError:
             # Ignore errors if specific options aren't available
             pass

    # --- Recursively apply theme to children ---
    # Important: Use winfo_children() to get current children
    for child in widget.winfo_children():
        # Toplevel windows need special handling if they are children
        # but created separately (like the language list window).
        # For child widgets within the main window hierarchy, this recursion works.
        if isinstance(child, tk.Toplevel) and child != widget:
             # Don't recurse into separate Toplevels automatically here.
             # They need their theme applied when created (e.g., in show_easyocr_languages)
             # Or we could apply the theme if we track them.
             # For now, skip direct recursion into Toplevels.
             # The theme will be applied when they are created or via a separate mechanism.
             pass
        else:
            apply_theme(child, theme) # Recursive call for child widgets

def toggle_theme():
    """Switches between dark and light mode."""
    global current_theme
    current_theme = "light" if current_theme == "dark" else "dark"
    apply_theme(root, current_theme)
    # Update button text
    theme_button.config(text=f"Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode")
    print(f"Theme switched to {current_theme} mode.")

def create_ocr_gui():
    """Builds and runs the OCR app with theme support."""
    global input_text, lang_listbox, selected_language_var, root, theme_button, current_theme

    # --- Set initial theme ---
    current_theme = "dark" # Default to dark mode

    # Create the main window
    root = tk.Tk()
    root.title("Mosquito OCR")
    root.geometry("750x550")
    # Apply initial theme
    apply_theme(root, current_theme)

    # --- OCR Language Selection Frame ---
    lang_frame = tk.LabelFrame(root, text="OCR Languages", padx=10, pady=10)
    lang_frame.pack(padx=10, pady=5, fill=tk.X)
    apply_theme(lang_frame, current_theme) # Apply theme to this frame

    # Listbox to display selected language FULL NAMES
    lang_listbox = tk.Listbox(lang_frame, height=5, width=30)
    lang_listbox.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
    apply_theme(lang_listbox, current_theme) # Apply theme to listbox

    # Populate listbox with initial language FULL NAMES based on codes
    for code in selected_ocr_language_codes:
        full_name = get_full_name_from_code(code)
        if full_name:
            lang_listbox.insert(tk.END, full_name)
        else:
            lang_listbox.insert(tk.END, code)

    # Scrollbar for the listbox
    lang_scrollbar = tk.Scrollbar(lang_frame, orient=tk.VERTICAL, command=lang_listbox.yview)
    lang_scrollbar.pack(side=tk.LEFT, fill=tk.Y)
    lang_listbox.config(yscrollcommand=lang_scrollbar.set)
    # Scrollbar theming is tricky, often inherits parent or system defaults.

    # Frame for controls
    lang_control_frame = tk.Frame(lang_frame)
    lang_control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    apply_theme(lang_control_frame, current_theme) # Apply theme to control frame

    # --- REPLACED OptionMenu with ttk.Combobox ---
    label_select_lang = tk.Label(lang_control_frame, text="Select/Add Language:")
    label_select_lang.pack(anchor=tk.W)
    apply_theme(label_select_lang, current_theme) # Apply theme to label

    # StringVar to hold the selected/typed value
    selected_language_var = tk.StringVar(value="")

    # Create the list of full names for the Combobox values, sorted
    sorted_full_names = sorted(EASYOCR_LANGUAGES.keys())

    # Create the Combobox
    language_combobox = ttk.Combobox(
        lang_control_frame,
        textvariable=selected_language_var,
        values=sorted_full_names,
        width=35,
        state="normal"
    )
    # Configure the dropdown listbox height
    root.option_add('*TCombobox*Listbox.height', 15)
    language_combobox.pack(fill=tk.X, pady=(0, 5))
    # Basic theme application for Combobox (limited)
    apply_theme(language_combobox, current_theme)

    # --- END Combobox Replacement ---

    # Buttons to add/remove languages
    add_remove_frame = tk.Frame(lang_control_frame)
    add_remove_frame.pack(fill=tk.X)
    apply_theme(add_remove_frame, current_theme) # Apply theme to button frame

    add_lang_button = tk.Button(add_remove_frame, text="Add Language", command=add_language)
    add_lang_button.pack(side=tk.LEFT, padx=(0, 5))
    apply_theme(add_lang_button, current_theme) # Apply theme to button

    remove_lang_button = tk.Button(add_remove_frame, text="Remove Selected", command=remove_language)
    remove_lang_button.pack(side=tk.LEFT, padx=(0, 5))
    apply_theme(remove_lang_button, current_theme) # Apply theme to button

    # Button to update/re-initialize the OCR reader
    update_reader_button = tk.Button(lang_control_frame, text="Update OCR Reader", command=update_reader)
    update_reader_button.pack(pady=(10, 5), anchor=tk.W)
    apply_theme(update_reader_button, current_theme) # Apply theme to button

    # Button to show the list of all supported languages
    show_lang_list_button = tk.Button(lang_control_frame, text="Show All Language Codes", command=show_easyocr_languages)
    show_lang_list_button.pack(pady=(0, 5), anchor=tk.W)
    apply_theme(show_lang_list_button, current_theme) # Apply theme to button

    # --- Theme Toggle Button ---
    theme_button = tk.Button(root, text=f"Switch to {'Light' if current_theme == 'dark' else 'Dark'} Mode", command=toggle_theme)
    theme_button.pack(pady=5)
    apply_theme(theme_button, current_theme) # Apply theme to theme button
    # --- End Theme Toggle ---

    # --- Initialize Reader on Startup ---
    initialize_ocr_reader(selected_ocr_language_codes)

    # --- GUI Elements (Buttons and Text Box) ---
    main_button_frame = tk.Frame(root)
    main_button_frame.pack(pady=10)
    apply_theme(main_button_frame, current_theme) # Apply theme to main button frame

    ocr_clipboard_button = tk.Button(
        main_button_frame,
        text="OCR Clipboard Image",
        command=OcrClipboardV4,
        font=("Arial", 10, "bold"),
        padx=10,
        pady=5
    )
    ocr_clipboard_button.pack(side=tk.LEFT, padx=10)
    apply_theme(ocr_clipboard_button, current_theme) # Apply theme to button

    ocr_local_button = tk.Button(
        main_button_frame,
        text="OCR Local Image",
        command=OcrLocalImageV3,
        font=("Arial", 10, "bold"),
        padx=10,
        pady=5
    )
    ocr_local_button.pack(side=tk.LEFT, padx=10)
    apply_theme(ocr_local_button, current_theme) # Apply theme to button

    input_text = scrolledtext.ScrolledText(
        root,
        wrap=tk.WORD,
        width=110,
        height=32,
        font=("Consolas", 10)
    )
    input_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    apply_theme(input_text, current_theme) # Apply theme to text widget

    # --- Run the GUI ---
    root.mainloop()
if __name__ == "__main__":
     reader = easyocr.Reader(['en', 'ro'])
     create_ocr_gui()