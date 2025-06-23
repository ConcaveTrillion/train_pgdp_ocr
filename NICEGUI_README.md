# NiceGUI OCR Data Labeler

This is a web-based version of the OCR data labeler using NiceGUI instead of Jupyter widgets.

## Files

- `nicegui_labeler.py` - Main NiceGUI implementation of the OCR labeler
- `nicegui_page_editor.py` - NiceGUI page editor for line-by-line editing
- `nicegui_line_editor.py` - NiceGUI line editor for individual word editing
- `run_nicegui_labeler.py` - Example script showing how to use the NiceGUI labeler

## Key Differences from Jupyter Version

### Technology Stack
- **Original**: Uses `ipywidgets` for Jupyter notebook interface
- **NiceGUI Version**: Uses `nicegui` for web-based interface

### Interface
- **Original**: Embedded in Jupyter notebook cells
- **NiceGUI Version**: Standalone web application accessible via browser

### Layout
- **Original**: Uses `VBox`, `HBox`, `Tab` from ipywidgets
- **NiceGUI Version**: Uses NiceGUI's layout components (`ui.row`, `ui.splitter`, `ui.tabs`)

### Image Display
- **Original**: Uses `ipywidgets.Image` with binary data
- **NiceGUI Version**: Uses base64 encoded images for web display

### Event Handling
- **Original**: Widget callbacks with event parameters
- **NiceGUI Version**: Direct function calls and NiceGUI's reactive system

## Features

The NiceGUI version maintains all the core functionality of the original:

### Main Labeler Features
- **Navigation**: Previous/Next page buttons and direct page navigation
- **OCR Processing**: Run OCR on pages, save/load OCR data
- **Image Views**: Multiple tabs showing different visualizations:
  - Original page image
  - Bounding boxes for paragraphs, lines, and words
  - Mismatch highlighting
- **Text Display**: Side-by-side OCR text and PGDP ground truth
- **Export Functions**: Export to training and validation sets
- **Bounding Box Refinement**: Tools to refine OCR bounding boxes

### Page Editor Features
- **Line Filtering**: Show all lines, only mismatches, or only unvalidated mismatches
- **Line Statistics**: View accuracy and validation statistics
- **Batch Operations**: Mark exact matches as validated, copy OCR to ground truth
- **Page-level Actions**: Expand and refine all bounding boxes

### Line Editor Features (Per Line)
- **Individual Line Editing**: Edit OCR text and ground truth for each line
- **Word-level Operations**:
  - **Split Words**: Interactive word splitting with visual feedback
  - **Merge Words**: Merge adjacent words (left/right)
  - **Edit Bounding Boxes**: Precise bounding box editing with margin controls
  - **Crop Operations**: Crop word bounding boxes (top, bottom, or both)
  - **Delete/Copy**: Delete words or copy OCR text to ground truth
- **Validation**: Mark lines as validated
- **Visual Feedback**: Color-coded borders (green=validated, gray=exact match, red=mismatch)

## Usage

### Basic Usage

```python
from data_labeler.nicegui_labeler import create_nicegui_labeler
from pd_book_tools.pgdp.pgdp_results import PGDPExport

# Load your PGDP data
pgdp_export = PGDPExport.from_json_file("path/to/your/pgdp_data.json")

# Run the labeler
create_nicegui_labeler(
    pgdp_export=pgdp_export,
    labeled_ocr_path="path/to/labeled_ocr",
    training_set_output_path="path/to/training_output",
    validation_set_output_path="path/to/validation_output",
    monospace_font_name="monospace",
    host='localhost',
    port=8080
)
```

### Running the Example

1. Adjust the paths in `run_nicegui_labeler.py` to match your data structure
2. Run the script:
   ```bash
   python run_nicegui_labeler.py
   ```
3. Open your browser to `http://localhost:8080`

## Installation

The NiceGUI version requires the `nicegui` package in addition to the existing dependencies:

```bash
pip install nicegui
```

## Interface Overview

### Main Interface
1. **Header**: Navigation controls and action buttons
2. **Left Panel**: Image views with different visualizations
3. **Right Panel**: Three tabs:
   - **Line Matching**: Interactive line-by-line editing
   - **OCR Text**: Read-only OCR text display
   - **PGDP P3 Text**: Read-only ground truth display

### Line Matching Tab
- **Filter Controls**: Radio buttons to show different line types
- **Line Cards**: Each line shown in a bordered card with:
  - Line image (cropped from page)
  - OCR text and ground truth displays
  - Action buttons (copy, delete, validate, crop)
  - Word matching table with individual word controls

### Word-Level Editing
Each word in the line matching interface provides:
- **Word Image**: Cropped word image
- **OCR Text**: Read-only OCR result
- **Ground Truth Input**: Editable text field
- **Action Buttons**:
  - **X**: Delete word
  - **ML/MR**: Merge left/right with adjacent words
  - **EB**: Edit bounding box with margin controls
  - **SP**: Split word with interactive splitting tool
  - **CT/CB/CA**: Crop top/bottom/all edges of bounding box

### Task Interfaces

#### Split Task
When splitting a word:
- Visual split line overlay on word image
- Pixel-level movement controls (<1p, >1p, <5%, >5%, <20%, >20%)
- Character-level text splitting controls
- Execute or cancel buttons

#### Edit Bounding Box Task
When editing bounding boxes:
- Margin adjustment controls for all four edges
- Pixel-level precision controls (±1, ±5, ±10 pixels)
- Refine button for automatic content fitting
- Save or cancel buttons

## Advantages of NiceGUI Version

1. **Standalone Web App**: No need for Jupyter notebook environment
2. **Modern UI**: Clean, responsive web interface with better user experience
3. **Accessibility**: Can be accessed from any device with a web browser
4. **Real-time Updates**: Immediate visual feedback for all operations
5. **Performance**: Generally better performance than Jupyter widgets
6. **Debugging**: Easier to debug with standard web development tools
7. **Scalability**: Can be deployed as a web service for team use

## Workflow

### Typical OCR Labeling Workflow
1. **Load Data**: Start with PGDP export and OCR results
2. **Navigate Pages**: Use page navigation to review OCR results
3. **Filter Lines**: Use line matching filters to focus on problematic areas
4. **Edit Individual Words**:
   - Fix OCR errors by editing ground truth
   - Split incorrectly merged words
   - Merge incorrectly split words
   - Adjust bounding boxes for better training data
5. **Validate Lines**: Mark corrected lines as validated
6. **Export Training Data**: Export corrected data for model training

### Batch Operations
- **Mark Exact Matches**: Quickly validate all perfect OCR matches
- **Copy OCR to GT**: Bulk copy OCR text to ground truth for good matches
- **Page-level Refinement**: Apply bounding box improvements to entire page

## Migration Notes

If you're migrating from the Jupyter version:

1. **Same Core Functionality**: All original features are preserved
2. **Improved Interface**: Better visual organization and user experience
3. **Enhanced Editing**: More intuitive word-level editing tools
4. **Better Performance**: Faster response times and smoother interactions
5. **Easy Deployment**: Can be run on a server for team access

## Technical Details

### Architecture
- **Modular Design**: Separate classes for main labeler, page editor, and line editor
- **Event-Driven**: Reactive UI updates based on user actions
- **Callback System**: Coordinated updates between different UI components
- **State Management**: Proper handling of editing tasks and validation states

### Image Processing
- **Base64 Encoding**: Efficient image display in web browser
- **Dynamic Cropping**: Real-time image cropping for word and line views
- **Overlay Graphics**: Visual feedback for split lines and bounding boxes

### Data Integration
- **PGDP Compatibility**: Full compatibility with existing PGDP data formats
- **OCR Integration**: Seamless integration with doctr OCR pipeline
- **Export Functions**: Standard training data export formats
