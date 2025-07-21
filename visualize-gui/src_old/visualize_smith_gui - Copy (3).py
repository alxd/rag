import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from upsetplot import UpSet, from_indicators
import numpy as np
import os
import warnings
from PIL import Image, ImageTk, ImageDraw, ImageFont
import threading
from collections import defaultdict
import re
import html

# Common suffixes for substring grouping (except suffixes)
common_suffixes = [
    'ation', 'ment', 'ness', 'sion', 'tion', 'ing', 'ed', 'ly', 'er', 'est', 'ful', 'less', 'able', 'ible', 'ous', 'ive', 'al', 'ic', 'ant', 'ent', 'ism', 'ist', 'ity', 'ty', 'en', 'ize', 'ise', 'ward', 'wise'
]

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class ZoomablePanCanvas:
    def __init__(self, parent):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg='white')
        
        # Scrollbars
        self.h_scrollbar = ttk.Scrollbar(parent, orient="horizontal", command=self.canvas.xview)
        self.v_scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        
        # Zoom and pan variables
        self.zoom_level = 1.0
        self.original_images = []
        self.image_refs = []
        self.image_items = []
        
        # Bind mouse events for panning
        self.canvas.bind("<Button-1>", self.start_pan)
        self.canvas.bind("<B1-Motion>", self.do_pan)
        self.canvas.bind("<MouseWheel>", self.zoom_wheel)
        
        self.last_x = 0
        self.last_y = 0
        
    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self.last_x = event.x
        self.last_y = event.y
        
    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        
    def zoom_wheel(self, event):
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
            
    def set_images(self, image_paths, window_width):
        self.original_images = []
        self.image_refs = []
        self.image_items = []
        self.canvas.delete("all")
        if not image_paths:
            return
        # Load all images and get max height
        loaded_imgs = []
        max_height = 0
        for img_path in image_paths:
            try:
                img = Image.open(img_path)
                loaded_imgs.append(img)
                max_height = max(max_height, img.height)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
        # Calculate total width to fit window
        num_images = len(loaded_imgs)
        padding = 10
        available_width = window_width - (num_images + 1) * padding - 50  # Account for scrollbar
        image_width = max(200, available_width // num_images)  # Minimum 200px per image
        x_offset = padding
        for img in loaded_imgs:
            aspect_ratio = img.width / img.height
            img_resized = img.resize((image_width, int(image_width / aspect_ratio)), Image.Resampling.LANCZOS)
            # Center vertically in max_height
            y_offset = padding + (max_height - img_resized.height) // 2
            self.original_images.append((img_resized, x_offset, y_offset))
            x_offset += image_width + padding
        self.zoom_level = 1.0
        self.update_display()
        
    def update_display(self):
        self.canvas.delete("all")
        self.image_refs = []
        self.image_items = []
        
        for img, x, y in self.original_images:
            # Apply zoom
            new_width = int(img.width * self.zoom_level)
            new_height = int(img.height * self.zoom_level)
            
            if new_width > 0 and new_height > 0:
                zoomed_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(zoomed_img)
                self.image_refs.append(photo)
                
                # Position with zoom
                scaled_x = int(x * self.zoom_level)
                scaled_y = int(y * self.zoom_level)
                
                item = self.canvas.create_image(scaled_x, scaled_y, anchor="nw", image=photo)
                self.image_items.append(item)
        
        # Update scroll region
        self.canvas.update_idletasks()
        bbox = self.canvas.bbox("all")
        if bbox:
            self.canvas.configure(scrollregion=bbox)
            
    def zoom_in(self):
        self.zoom_level = min(self.zoom_level * 1.2, 5.0)
        self.update_display()
        
    def zoom_out(self):
        self.zoom_level = max(self.zoom_level / 1.2, 0.2)
        self.update_display()
        
    def fit_to_window(self):
        self.zoom_level = 1.0
        self.update_display()

class UpSetGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("UpSet Plot Generator")
        self.root.geometry("1400x800")
        
        self.csv_file = None
        self.output_images = []
        self.color_mapping = {}
        self.use_colors = tk.BooleanVar(value=True)
        self.group_by_subletters = tk.BooleanVar(value=False)
        self.group_by_words = tk.BooleanVar(value=True)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)  # Results frame row
        
        # File selection
        ttk.Label(main_frame, text="CSV File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.file_label = ttk.Label(main_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        
        ttk.Button(main_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=(10, 0), pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="5")
        options_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Color checkbox and entry
        color_frame = ttk.Frame(options_frame)
        color_frame.grid(row=0, column=0, sticky=tk.W)
        ttk.Checkbutton(color_frame, text="Color criteria", variable=self.use_colors).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(color_frame, text="Min shared letters:").grid(row=0, column=1, padx=(10,0))
        self.min_letters_var = tk.StringVar(value="5")
        self.min_letters_entry = ttk.Entry(color_frame, textvariable=self.min_letters_var, width=3)
        self.min_letters_entry.grid(row=0, column=2, padx=(2,0))
        ttk.Checkbutton(color_frame, text="Group by subletters (except suffixes)", variable=self.group_by_subletters, command=self.on_group_by_subletters).grid(row=0, column=3, padx=(10,0))
        ttk.Checkbutton(color_frame, text="Group by whole words", variable=self.group_by_words, command=self.on_group_by_words).grid(row=0, column=4, padx=(10,0))
        
        # Process button and progress
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        button_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.process_btn = ttk.Button(button_frame, text="Generate UpSet Plots", 
                                     command=self.start_processing, state='disabled')
        self.process_btn.grid(row=0, column=1)
        
        # Results frame with zoom controls
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="5")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(1, weight=1)
        
        # Zoom controls
        zoom_frame = ttk.Frame(results_frame)
        zoom_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Button(zoom_frame, text="Zoom In", command=self.zoom_in).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(zoom_frame, text="Zoom Out", command=self.zoom_out).grid(row=0, column=1, padx=(0, 5))
        ttk.Button(zoom_frame, text="Fit to Window", command=self.fit_to_window).grid(row=0, column=2, padx=(0, 5))
        
        ttk.Label(zoom_frame, text="Use mouse wheel to zoom, click and drag to pan").grid(row=0, column=3, padx=(20, 0))
        
        # Zoomable canvas
        canvas_frame = ttk.Frame(results_frame)
        canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        self.zoomable_canvas = ZoomablePanCanvas(canvas_frame)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=4, column=0, columnspan=3, pady=5)
        
    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.csv_file = file_path
            self.file_label.config(text=os.path.basename(file_path), foreground="black")
            self.process_btn.config(state='normal')
            
    def zoom_in(self):
        self.zoomable_canvas.zoom_in()
        
    def zoom_out(self):
        self.zoomable_canvas.zoom_out()
        
    def fit_to_window(self):
        self.zoomable_canvas.fit_to_window()
        
    def extract_concepts(self, text):
        if pd.isna(text):
            return []
        lines = text.split('\n')
        concepts = []
        for line in lines:
            line = line.strip()
            if line.startswith(tuple(str(i)+'.' for i in range(1, 21))):
                # Markdown or plain numbered list
                if '**' in line:
                    c = line.split('**')[1].replace(':', '').replace('.', '').strip()
                    if c:
                        concepts.append(c)
                else:
                    c = line.lstrip('0123456789. ').replace(':', '').replace('.', '').strip()
                    if c:
                        concepts.append(c)
        return concepts
    
    def normalize_word(self, word):
        """Normalize word by removing plurals and common variations"""
        word = word.lower().strip()
        
        # Handle common plural forms
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            return word[:-2]
        elif word.endswith('s') and len(word) > 2:
            return word[:-1]
        
        # Handle common suffixes
        suffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        
        return word
    
    def extract_key_words(self, concept):
        """Extract meaningful words from a concept"""
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        
        words = re.findall(r'\b\w+\b', concept.lower())
        meaningful_words = [self.normalize_word(word) for word in words if word not in stop_words and len(word) > 2]
        
        return meaningful_words
    
    def create_color_mapping(self, all_concepts):
        """Assign a color to each global group, by whole word or substring, case-insensitive."""
        self.color_mapping = {}
        min_letters = 5
        try:
            min_letters = int(self.min_letters_var.get())
        except Exception:
            pass
        all_concepts_sorted = sorted(all_concepts, key=lambda x: x.lower())
        groups = []
        assigned = set()
        group_by_words = self.group_by_words.get()
        for i, concept in enumerate(all_concepts_sorted):
            concept_lc = concept.lower()
            if concept_lc in assigned:
                continue
            group = [concept]
            words1 = set(re.findall(r'\b\w+\b', concept_lc))
            for j, other in enumerate(all_concepts_sorted):
                other_lc = other.lower()
                if i == j or other_lc in assigned:
                    continue
                words2 = set(re.findall(r'\b\w+\b', other_lc))
                found = False
                if group_by_words:
                    # Group if any whole word matches
                    if words1 & words2:
                        found = True
                else:
                    # Group if any substring of min_letters matches
                    for w1 in words1:
                        for w2 in words2:
                            for k in range(len(w1) - min_letters + 1):
                                sub = w1[k:k+min_letters]
                                if sub and sub in w2:
                                    # Check if sub is a suffix in either word and is a common suffix
                                    if (w1.endswith(sub) or w2.endswith(sub)) and sub in common_suffixes:
                                        continue  # skip grouping by suffix
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                if found:
                    group.append(other)
            groups.append(group)
            assigned.update([c.lower() for c in group])
        # Assign colors to groups
        colors = [
            '#800000',  # Maroon
            '#FF8C00',  # DarkOrange
            '#228B22',  # ForestGreen
            '#8B008B',  # DarkMagenta
            '#A0522D',  # Sienna
            '#2E8B57',  # SeaGreen
            '#9932CC',  # DarkOrchid
            '#FFD700',  # Gold
            '#556B2F',  # DarkOliveGreen
            '#C71585',  # MediumVioletRed
            '#8B4513',  # SaddleBrown
            '#20B2AA',  # LightSeaGreen
            '#B22222',  # FireBrick
            '#FF4500',  # OrangeRed
            '#6A5ACD',  # SlateBlue
            '#D2691E',  # Chocolate
            '#006400',  # DarkGreen
            '#708090',  # SlateGray
            '#FF6347',  # Tomato
            '#483D8B',  # DarkSlateBlue
            '#000000',  # Black
            '#808000',  # Olive
            '#8B0000',  # DarkRed
            '#FF1493',  # DeepPink
        ]
        color_idx = 0
        for group in groups:
            color = colors[color_idx % len(colors)]
            for concept in group:
                self.color_mapping[concept] = color
            color_idx += 1
        return self.color_mapping
    
    def create_colored_text_image(self, text, color, font_size=12):
        """Create an image with colored text on white background"""
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Get text dimensions
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create image with white background
        img = Image.new('RGB', (text_width + 10, text_height + 6), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw text in color
        draw.text((5, 3), text, fill=color, font=font)
        
        return img
    
    def start_processing(self):
        if not self.csv_file:
            messagebox.showerror("Error", "Please select a CSV file first")
            return
            
        # Start processing in a separate thread
        self.process_btn.config(state='disabled')
        self.progress.start()
        self.status_label.config(text="Processing...")
        
        thread = threading.Thread(target=self.process_data)
        thread.daemon = True
        thread.start()
        
    def process_data(self):
        try:
            # Read the CSV file
            df = pd.read_csv(self.csv_file, encoding='cp1253')
            
            # Extract concepts
            df['Concepts'] = df['Main Answer'].apply(self.extract_concepts)

            all_concepts = set()
            for concepts in df['Concepts']:
                all_concepts.update(concepts)
            self.create_color_mapping(list(all_concepts))  # Always global, not per block
            
            params = ['Temperature', 'Top-p', 'Top-k', 'BM25 Weight']
            param_labels = {'Temperature': 'Temp', 'Top-p': 'Topp', 'Top-k': 'Topk', 'BM25 Weight': 'BM25'}
            
            # Create output directory
            outdir = 'smith_gui_output'
            os.makedirs(outdir, exist_ok=True)
            
            # Block definitions
            blocks = [
                (0, 5, 'Temperature'),
                (5, 10, 'Top-p'),
                (10, 15, 'Top-k'),
                (15, 20, 'BM25 Weight'),
            ]
            
            output_files = []
            
            for start, end, varying_param in blocks:
                subset = df.iloc[start:end].copy()
                if subset.empty:
                    continue
                    
                mlb = MultiLabelBinarizer()
                concept_matrix = pd.DataFrame(mlb.fit_transform(subset['Concepts']), columns=mlb.classes_)
                
                for p in params:
                    concept_matrix[p] = subset[p].values
                
                concept_matrix_reset = concept_matrix.drop(params, axis=1).astype(bool).reset_index(drop=True)
                upset_data = from_indicators(concept_matrix_reset, concept_matrix_reset.columns)
                
                fig = plt.figure(figsize=(12, 8))
                upset = UpSet(upset_data, show_counts=True)
                axes = upset.plot(fig=fig)
                bar_ax = axes['intersections']
                matrix_ax = axes['matrix']
                bars = bar_ax.patches
                upset_index = upset_data.index
                
                # Get min_letters from GUI
                try:
                    min_letters = int(self.min_letters_var.get())
                except Exception:
                    min_letters = 3
                block_concepts = list(concept_matrix_reset.columns)
                block_color_mapping = self.get_block_color_mapping_with_global_palette(block_concepts, min_letters, self.color_mapping)

                # Apply block-local color mapping to y-tick labels
                if self.use_colors.get():
                    # Use color mapping
                    for label in matrix_ax.get_yticklabels():
                        concept = label.get_text()
                        if concept in block_color_mapping:
                            color = block_color_mapping[concept]
                            label.set_color(color)
                            label.set_weight('bold')
                            label.set_fontsize(10)
                else:
                    # All black, not bold
                    for label in matrix_ax.get_yticklabels():
                        label.set_color('black')
                        label.set_weight('normal')
                        label.set_fontsize(10)
                
                # Add parameter value labels
                for i, (bar, intersection) in enumerate(zip(bars, upset_index)):
                    if i == 0:
                        continue
                    if bar.get_height() == 0:
                        continue
                    
                    x = bar.get_x() + bar.get_width() / 2
                    intersection_idx = i - 1
                    actual_intersection = upset_index[intersection_idx]
                    
                    mask = np.ones(len(concept_matrix), dtype=bool)
                    for col, present in zip(concept_matrix.drop(params, axis=1).columns, actual_intersection):
                        if present:
                            mask &= concept_matrix[col] == 1
                        else:
                            mask &= concept_matrix[col] == 0
                    
                    param_vals = concept_matrix.loc[mask, varying_param].unique()
                    label = ','.join(map(str, param_vals)) if len(param_vals) > 0 else ''
                    
                    present_indices = [j for j, present in enumerate(actual_intersection) if present]
                    if not present_indices:
                        continue
                    
                    y_label = len(concept_matrix_reset.columns) - 0.3
                    matrix_ax.text(x, y_label, label, ha='center', va='bottom', 
                                 fontsize=10, color='red', rotation=0, clip_on=False, weight='bold')
                
                # Add title
                other_params = [p for p in params if p != varying_param]
                fixed_vals = {param_labels[p]: subset[p].iloc[0] for p in other_params}
                fixed_str = ' | '.join(f"{p}={v}" for p, v in fixed_vals.items())
                plt.suptitle(f"UpSet Diagram: {param_labels[varying_param]} sweep\nOther Params: {fixed_str}", fontsize=14, y=0.98)
                # Add consistent top padding
                plt.subplots_adjust(top=0.88, bottom=0.12)
                plt.tight_layout(rect=[0, 0.12, 1, 0.88])
                
                # Save plot
                outpath = f"{outdir}/upset_{param_labels[varying_param]}_composed.png"
                plt.savefig(outpath, dpi=150, bbox_inches='tight', pad_inches=0.5)
                plt.close('all')
                
                output_files.append(outpath)
            
            # Update GUI in main thread
            self.root.after(0, self.update_results, output_files)
            
            # Generate HTML table
            self.generate_html_table(blocks, df, self.color_mapping, outdir, param_labels)
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_results(self, output_files):
        # Get current window width
        window_width = self.root.winfo_width()
        
        # Update zoomable canvas with new images
        self.zoomable_canvas.set_images(output_files, window_width)
        
        # Stop progress and update status
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text=f"Processing complete. Generated {len(output_files)} plots.")
    
    def show_error(self, error_msg):
        self.progress.stop()
        self.process_btn.config(state='normal')
        self.status_label.config(text="Error occurred")
        messagebox.showerror("Processing Error", f"An error occurred during processing:\n\n{error_msg}")

    def generate_html_table(self, blocks, df, color_mapping, outdir, param_labels):
        html_lines = [
            "<style>",
            ".concept-box { display: inline-block; padding: 1px 3px; margin: 1px; border-radius: 2px; font-weight: bold; white-space: nowrap; font-size: 10px; color: #fff; }",
            ".parameter-box { font-weight: bold; color: #333; font-size: 11px; }",
            ".varying-parameter { background-color: #ffffcc; }",
            ".concepts-line { white-space: nowrap; overflow-x: auto; }",
            "table { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 12px; }",
            "th, td { border: 1px solid #ddd; padding: 6px; text-align: left; vertical-align: top; }",
            "th { background-color: #f2f2f2; font-size: 11px; }",
            "</style>",
            "<table>",
            "<thead><tr>"
            "<th class='varying-parameter'>Temperature</th>"
            "<th class='varying-parameter'>Top P</th>"
            "<th class='varying-parameter'>Top K</th>"
            "<th class='varying-parameter'>BM25 Weight</th>"
            "<th>Extracted Concepts</th>"
            "</tr></thead><tbody>"
        ]
        for start, end, varying_param in blocks:
            subset = df.iloc[start:end].copy()
            for idx, row in subset.iterrows():
                html_lines.append("<tr>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Temperature','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Top-p','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('Top-k','')}</span></td>")
                html_lines.append(f"<td class='varying-parameter'><span class='parameter-box'>{row.get('BM25 Weight','')}</span></td>")
                html_lines.append("<td><div class='concepts-line'>")
                for concept in row['Concepts']:
                    color = color_mapping.get(concept, '#888')
                    html_lines.append(
                        f"<span class='concept-box' style='background-color: {color};'>{html.escape(concept)}</span>"
                    )
                html_lines.append("</div></td></tr>")
        html_lines.append("</tbody></table>")
        with open(os.path.join(outdir, "smith_compare_auto.htm"), "w", encoding="utf-8") as f:
            f.write("\n".join(html_lines))

    def get_block_color_mapping_with_global_palette(self, concepts, min_letters, global_color_mapping):
        # Group concepts by shared consecutive substring
        concepts_sorted = sorted(concepts)
        groups = []
        assigned = set()
        for i, concept in enumerate(concepts_sorted):
            if concept in assigned:
                continue
            group = [concept]
            words1 = re.findall(r'\b\w+\b', concept.lower())
            for j, other in enumerate(concepts_sorted):
                if i == j or other in assigned:
                    continue
                words2 = re.findall(r'\b\w+\b', other.lower())
                found = False
                for w1 in words1:
                    for w2 in words2:
                        for k in range(len(w1) - min_letters + 1):
                            sub = w1[k:k+min_letters]
                            if sub and sub in w2:
                                # Check if sub is a suffix in either word and is a common suffix
                                if (w1.endswith(sub) or w2.endswith(sub)) and sub in common_suffixes:
                                    continue  # skip grouping by suffix
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    group.append(other)
            groups.append(group)
            assigned.update(group)
        # Assign the color of the first concept in the group (from the global mapping) to all in the group
        color_mapping = {}
        for group in groups:
            color = global_color_mapping[group[0]]
            for concept in group:
                color_mapping[concept] = color
        return color_mapping

    def on_group_by_subletters(self):
        if self.group_by_subletters.get():
            self.group_by_words.set(False)

    def on_group_by_words(self):
        if self.group_by_words.get():
            self.group_by_subletters.set(False)

def main():
    root = tk.Tk()
    app = UpSetGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()