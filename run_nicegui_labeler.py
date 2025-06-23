#!/usr/bin/env python3
"""
Example script to run the NiceGUI version of the OCR data labeler.

This script demonstrates how to use the NiceGuiLabeler class with project loading
from the source-pgdp-data directory structure.
"""

import pathlib
from data_labeler.nicegui_labeler import NiceGuiLabeler


def main():
    """Main function to run the NiceGUI OCR labeler"""
    
    # Set up paths - adjust these to match your data structure
    base_path = pathlib.Path(__file__).parent
    
    # Paths for project loading
    source_pgdp_data_path = base_path / "source-pgdp-data"
    labeled_ocr_path = base_path / "matched-ocr"
    training_output_path = base_path / "ml-training"
    validation_output_path = base_path / "ml-validation"
    
    # Font configuration
    monospace_font_name = "DPSansMono"
    monospace_font_path = base_path / "DPSansMono.ttf"
    
    # Create a labeler that can load projects from the directory structure
    print("Starting NiceGUI OCR Data Labeler...")
    print(f"Source data directory: {source_pgdp_data_path}")
    print("Loading available projects from source-pgdp-data/output...")
    
    # Create the labeler with project loading capability
    labeler = NiceGuiLabeler.from_project_directory(
        source_pgdp_data_path=source_pgdp_data_path,
        labeled_ocr_path=labeled_ocr_path,
        training_set_output_path=training_output_path,
        validation_set_output_path=validation_output_path,
        monospace_font_name=monospace_font_name,
        monospace_font_path=monospace_font_path,
    )
    
    # Run the labeler
    print("Starting web server at http://localhost:8080")
    print("Select a project from the dropdown and click 'Load Project' to begin.")
    labeler.run(host='localhost', port=8080)


if __name__ in {"__main__", "__mp_main__"}:
    main()
