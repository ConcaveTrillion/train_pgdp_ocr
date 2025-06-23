import pathlib
import base64
from logging import getLogger
import cv2
from shutil import copyfile

from nicegui import ui

from pd_book_tools.image_processing.cv2_processing.encoding import (
    encode_bgr_image_as_png,
)
from pd_book_tools.ocr.document import Document
from pd_book_tools.ocr.page import Page
from pd_book_tools.pgdp.pgdp_results import PGDPExport, PGDPPage
from pd_book_tools.ocr.cv2_doctr import doctr_ocr_cv2_image, get_default_doctr_predictor

from .nicegui_page_editor import NiceGuiPageEditor

# Configure logging
logger = getLogger(__name__)
ui_logger = getLogger(__name__ + ".UI")


class NiceGuiLabeler:
    """NiceGUI version of the OCR data labeler"""

    def __init__(
        self,
        pgdp_export: PGDPExport | None = None,
        labeled_ocr_path: pathlib.Path | str | None = None,
        training_set_output_path: pathlib.Path | str | None = None,
        validation_set_output_path: pathlib.Path | str | None = None,
        monospace_font_name: str = "monospace",
        monospace_font_path: pathlib.Path | str | None = None,
        start_page_name="",
        start_page_idx=0,
        doctr_predictor=None,
        source_pgdp_data_path: pathlib.Path | str | None = None,
    ):
        # Initialize attributes
        self._current_page_idx: int = 0
        self._total_pages: int = 0
        self.current_page_name = ""
        self.matched_ocr_pages = {}
        self.monospace_font_name = monospace_font_name

        # OCR and data processing
        self.doctr_predictor = doctr_predictor
        self.pgdp_export = pgdp_export

        # Project loading
        self.source_pgdp_data_path = (
            pathlib.Path(source_pgdp_data_path) if source_pgdp_data_path else None
        )
        self.available_projects = []
        self.selected_project = None
        self.project_loaded = False

        # Set up paths if provided
        if labeled_ocr_path:
            self.labeled_ocr_path: pathlib.Path = self.create_path(labeled_ocr_path)
        if training_set_output_path:
            self.training_set_output_path: pathlib.Path = self.create_path(
                training_set_output_path
            )
        if validation_set_output_path:
            self.validation_set_output_path: pathlib.Path = self.create_path(
                validation_set_output_path
            )

        # If we have a pgdp_export, initialize as before
        if self.pgdp_export:
            self._initialize_with_pgdp_export(start_page_name, start_page_idx)
        else:
            # Initialize empty state for project loading
            self.page_indexby_name = {}
            self.page_indexby_nbr = {}
            self._total_pages = 0

        # Initialize OCR predictor
        self.init_ocr_doctr_predictor()

        # UI elements (will be initialized in setup_ui)
        self.current_page_display = None
        self.page_number_input = None
        self.plain_image = None
        self.ocr_image_pgh_bounding_box = None
        self.ocr_image_lines_bounding_box = None
        self.ocr_image_words_bounding_box = None
        self.ocr_image_mismatches = None
        self.ocr_text_display = None
        self.pgdp_text_display = None
        self.project_selector = None
        self.load_project_button = None

        # Page editor for line-by-line editing
        self.page_editor: NiceGuiPageEditor | None = None

    def _initialize_with_pgdp_export(self, start_page_name="", start_page_idx=0):
        """Initialize the labeler with a PGDP export"""
        if not self.pgdp_export or not self.pgdp_export.pages:
            raise ValueError("Cannot initialize: pgdp_export is None or has no pages")

        # Create page indices
        self.page_indexby_name = {
            item.png_file: i for i, item in enumerate(self.pgdp_export.pages)
        }
        self.page_indexby_nbr = {
            i: item.png_file for i, item in enumerate(self.pgdp_export.pages)
        }
        self._total_pages = len(self.pgdp_export.pages) - 1

        # Set project_loaded first so that current_pgdp_page works
        self.project_loaded = True

        # Set starting page
        if start_page_name:
            new_idx = self.page_indexby_name.get(start_page_name, -1)
            if 0 <= new_idx < self.total_pages:
                self._current_page_idx = new_idx
                self.current_page_name = pathlib.Path(
                    self.pgdp_export.pages[self._current_page_idx].png_file
                ).stem
        elif 0 <= start_page_idx < self.total_pages:
            self._current_page_idx = start_page_idx
            self.current_page_name = pathlib.Path(
                self.pgdp_export.pages[self._current_page_idx].png_file
            ).stem

        # Make sure we have a valid current page name
        if not hasattr(self, "current_page_name") or not self.current_page_name:
            self.current_page_name = pathlib.Path(
                self.pgdp_export.pages[self._current_page_idx].png_file
            ).stem

    @classmethod
    def from_project_directory(
        cls,
        source_pgdp_data_path: pathlib.Path | str,
        labeled_ocr_path: pathlib.Path | str | None = None,
        training_set_output_path: pathlib.Path | str | None = None,
        validation_set_output_path: pathlib.Path | str | None = None,
        monospace_font_name: str = "monospace",
        monospace_font_path: pathlib.Path | str | None = None,
        doctr_predictor=None,
    ):
        """Create a labeler instance that can load projects from the source directory"""
        # Set default paths relative to source directory
        source_path = pathlib.Path(source_pgdp_data_path)

        if labeled_ocr_path is None:
            labeled_ocr_path = source_path.parent / "matched-ocr"
        if training_set_output_path is None:
            training_set_output_path = source_path.parent / "ml-training"
        if validation_set_output_path is None:
            validation_set_output_path = source_path.parent / "ml-validation"

        return cls(
            pgdp_export=None,  # Will be loaded when project is selected
            labeled_ocr_path=labeled_ocr_path,
            training_set_output_path=training_set_output_path,
            validation_set_output_path=validation_set_output_path,
            monospace_font_name=monospace_font_name,
            monospace_font_path=monospace_font_path,
            doctr_predictor=doctr_predictor,
            source_pgdp_data_path=source_pgdp_data_path,
        )

    def create_path(self, str_or_path: pathlib.Path | str) -> pathlib.Path:
        """Create directory if it doesn't exist"""
        if isinstance(str_or_path, str):
            str_or_path = pathlib.Path(str_or_path)
        if not str_or_path.exists():
            str_or_path.mkdir(parents=True, exist_ok=True)
        return str_or_path

    def init_ocr_doctr_predictor(self):
        """Initialize the OCR predictor"""
        if self.doctr_predictor:
            self.main_ocr_predictor = self.doctr_predictor
            return
        self.main_ocr_predictor = get_default_doctr_predictor()

    @property
    def current_page_idx(self):
        return self._current_page_idx

    @current_page_idx.setter
    def current_page_idx(self, value):
        if 0 <= value <= self.total_pages:
            self._current_page_idx = value
            self.current_page_name = pathlib.Path(self.current_pgdp_page.png_file).stem
            self.refresh_ui()

    @property
    def total_pages(self) -> int:
        return self._total_pages

    @property
    def current_pgdp_page(self) -> PGDPPage | None:
        if not self.pgdp_export or not self.project_loaded:
            return None
        return self.pgdp_export.pages[self.current_page_idx]

    @property
    def current_ocr_page(self) -> Page | None:
        if not self.project_loaded:
            return None
        current_idx = int(self.current_page_idx)  # Ensure it's a simple int
        if current_idx not in self.matched_ocr_pages:
            return None
        return self.matched_ocr_pages[current_idx]["page"]

    @property
    def export_prefix(self):
        if not self.pgdp_export:
            return f"unknown_project_{self.current_page_idx}"
        return f"{self.pgdp_export.project_id}_{self.current_page_idx}"

    def prev_page(self):
        """Go to previous page"""
        if not self.ensure_project_loaded():
            return
        if self.current_page_idx > 0:
            self.current_page_idx = self.current_page_idx - 1

    def next_page(self):
        """Go to next page"""
        if not self.ensure_project_loaded():
            return
        if self.current_page_idx < self.total_pages:
            self.current_page_idx = self.current_page_idx + 1

    def go_to_page(self):
        """Go to specific page number"""
        if not self.ensure_project_loaded():
            return
        if self.page_number_input and self.page_number_input.value is not None:
            page_num = int(self.page_number_input.value)
            if 0 <= page_num <= self.total_pages:
                self.current_page_idx = page_num

    def run_ocr(self, force_refresh_ocr=False):
        """Run OCR or get saved OCR document"""
        ui_logger.debug(f"Running OCR for page index: {self.current_page_idx}")

        # Import saved data if it exists
        self.import_ocr_document()

        # Check if we need to run OCR - be explicit about the comparison
        current_idx = int(self.current_page_idx)  # Ensure it's a simple int
        needs_ocr = (current_idx not in self.matched_ocr_pages) or force_refresh_ocr

        if needs_ocr:
            current_page = self.current_pgdp_page
            if not current_page:
                ui_logger.error("Cannot run OCR: no current page available")
                return

            source_image = current_page.png_full_path
            cv2_numpy_image = cv2.imread(str(source_image.resolve()))

            # Run OCR
            ocr_page: Page = doctr_ocr_cv2_image(
                image=cv2_numpy_image,
                source_image=str(source_image),
                predictor=self.main_ocr_predictor,
            )
            ocr_page.cv2_numpy_page_image = cv2.imread(str(source_image.resolve()))

            ui_logger.debug(
                f"OCR Completed. Page text length: {len(ocr_page.text)} characters."
            )

            ocr_page.reorganize_page()
            ocr_page.add_ground_truth(current_page.processed_page_text)

            self.matched_ocr_pages[current_idx] = {"page": ocr_page}

    def reset_ocr(self):
        """Reset OCR for current page"""
        self.run_ocr(force_refresh_ocr=True)
        self.refresh_ui()

    def refresh_page_images(self):
        """Refresh page images with bounding boxes"""
        ui_logger.debug(
            f"Refreshing page images for page index: {self.current_page_idx}"
        )
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        ocr_page.refresh_page_images()
        self.reload_page_images_ui()
        self.update_images()

    def reload_page_images_ui(self):
        """Reload page images for UI display"""
        logger.debug(f"Reloading page images for page index: {self.current_page_idx}")
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]

        if ocr_page.cv2_numpy_page_image is None:
            raise ValueError("Current OCR page does not have a valid image.")

        self.matched_ocr_pages[self.current_page_idx] = {
            **self.matched_ocr_pages[self.current_page_idx],
            "width": ocr_page.cv2_numpy_page_image.shape[1],
            "height": ocr_page.cv2_numpy_page_image.shape[0],
            "page_image": encode_bgr_image_as_png(ocr_page.cv2_numpy_page_image),
            "ocr_image_words_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_word_with_bboxes
            )
            if ocr_page.cv2_numpy_page_image_word_with_bboxes is not None
            else None,
            "ocr_image_mismatches": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_matched_word_with_colors
            )
            if ocr_page.cv2_numpy_page_image_matched_word_with_colors is not None
            else None,
            "ocr_image_lines_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_line_with_bboxes
            )
            if ocr_page.cv2_numpy_page_image_line_with_bboxes is not None
            else None,
            "ocr_image_pgh_bounding_box": encode_bgr_image_as_png(
                ocr_page.cv2_numpy_page_image_paragraph_with_bboxes
            )
            if ocr_page.cv2_numpy_page_image_paragraph_with_bboxes is not None
            else None,
        }

    def expand_and_refine_all_bboxes(self):
        """Expand and refine all bounding boxes"""
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]

        ui_logger.debug("Expanding all word bounding boxes to content and beyond.")
        for word in ocr_page.words:
            if ocr_page.cv2_numpy_page_image is None:
                raise ValueError(
                    "Current OCR page does not have a valid image to refine bounding boxes."
                )
            word.bounding_box = word.bounding_box.crop_bottom(
                image=ocr_page.cv2_numpy_page_image
            )
            word.bounding_box = word.bounding_box.expand_to_content(
                image=ocr_page.cv2_numpy_page_image
            )

        ocr_page.refine_bounding_boxes(padding_px=2)
        self.refresh_ui()

    def refine_all_bboxes(self):
        """Refine all bounding boxes"""
        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        ocr_page.refine_bounding_boxes(padding_px=2)
        self.refresh_ui()

    def export_training(self):
        """Export current page to training set"""
        prefix = self.export_prefix
        self.current_ocr_page.convert_to_training_set(
            output_path=self.training_set_output_path,
            prefix=prefix,
        )
        ui.notify(
            f"Exported page {self.current_page_idx} to training set", type="positive"
        )

    def export_validation(self):
        """Export current page to validation set"""
        prefix = self.export_prefix
        self.current_ocr_page.convert_to_training_set(
            output_path=self.validation_set_output_path,
            prefix=prefix,
        )
        ui.notify(
            f"Exported page {self.current_page_idx} to validation set", type="positive"
        )

    def export_ocr_document(self):
        """Export OCR document to file"""
        if self.current_page_idx not in self.matched_ocr_pages:
            ui.notify("No OCR page to save", type="negative")
            return

        ocr_page: Page = self.matched_ocr_pages[self.current_page_idx]["page"]
        ocr_image_path = self.current_pgdp_page.png_full_path

        # Copy image to labeled OCR path
        target_image_path = pathlib.Path(
            self.labeled_ocr_path, f"{self.export_prefix}.png"
        )

        ocr_document: Document = Document(
            pages=[ocr_page],
            source_lib="doctr-pgdp-labeled",
            source_path=target_image_path,
        )

        target_path = pathlib.Path(self.labeled_ocr_path, f"{self.export_prefix}.json")

        logger.info(f"Exporting OCR document to {target_path}")
        ocr_document.to_json_file(target_path)

        logger.info(f"Copying OCR image to {target_image_path}")
        copyfile(ocr_image_path, target_image_path)

        ui.notify(
            f"Saved OCR document for page {self.current_page_idx}", type="positive"
        )

    def import_ocr_document(self):
        """Import OCR document from JSON file"""
        ocr_json_path = pathlib.Path(
            self.labeled_ocr_path, f"{self.export_prefix}.json"
        )
        logger.debug(f"Importing OCR document from {ocr_json_path}")

        if not ocr_json_path.exists():
            logger.debug(f"OCR JSON file does not exist: {ocr_json_path}")
            return

        ocr_document: Document = Document.from_json_file(ocr_json_path)
        if len(ocr_document.pages) != 1:
            logger.error("OCR document must contain exactly one page.")
            return

        ocr_page: Page = ocr_document.pages[0]
        source_image = ocr_document.source_path
        if not source_image:
            logger.error("OCR document does not contain a source image path.")
            return

        ocr_page.cv2_numpy_page_image = cv2.imread(str(source_image.resolve()))
        self.matched_ocr_pages[self.current_page_idx] = {"page": ocr_page}

    def reload_ocr_from_file(self):
        """Reload OCR from file"""
        self.import_ocr_document()
        self.refresh_ui()
        ui.notify(f"Reloaded OCR for page {self.current_page_idx}", type="info")

    def update_images(self):
        """Update image displays"""
        if self.current_page_idx not in self.matched_ocr_pages:
            return

        page_data = self.matched_ocr_pages[self.current_page_idx]

        # Convert binary image data to base64 for display
        if self.plain_image is not None and page_data.get("page_image") is not None:
            img_b64 = base64.b64encode(page_data["page_image"]).decode()
            self.plain_image.set_source(f"data:image/png;base64,{img_b64}")

        if (
            self.ocr_image_pgh_bounding_box is not None
            and page_data.get("ocr_image_pgh_bounding_box") is not None
        ):
            img_b64 = base64.b64encode(page_data["ocr_image_pgh_bounding_box"]).decode()
            self.ocr_image_pgh_bounding_box.set_source(
                f"data:image/png;base64,{img_b64}"
            )

        if (
            self.ocr_image_lines_bounding_box is not None
            and page_data.get("ocr_image_lines_bounding_box") is not None
        ):
            img_b64 = base64.b64encode(
                page_data["ocr_image_lines_bounding_box"]
            ).decode()
            self.ocr_image_lines_bounding_box.set_source(
                f"data:image/png;base64,{img_b64}"
            )

        if (
            self.ocr_image_words_bounding_box is not None
            and page_data.get("ocr_image_words_bounding_box") is not None
        ):
            img_b64 = base64.b64encode(
                page_data["ocr_image_words_bounding_box"]
            ).decode()
            self.ocr_image_words_bounding_box.set_source(
                f"data:image/png;base64,{img_b64}"
            )

        if (
            self.ocr_image_mismatches is not None
            and page_data.get("ocr_image_mismatches") is not None
        ):
            img_b64 = base64.b64encode(page_data["ocr_image_mismatches"]).decode()
            self.ocr_image_mismatches.set_source(f"data:image/png;base64,{img_b64}")

    def update_text_displays(self):
        """Update text displays"""
        if self.current_page_idx not in self.matched_ocr_pages:
            return

        # Update OCR text display
        if self.ocr_text_display:
            ocr_lines = self.current_ocr_page.text.splitlines()
            ocr_html = "<table style='font-family: monospace; font-size: 12px;'>"
            for i, line in enumerate(ocr_lines):
                ocr_html += f"<tr><td>{i}</td><td>{line}</td></tr>"
            ocr_html += "</table>"
            self.ocr_text_display.content = ocr_html

        # Update PGDP text display
        if self.pgdp_text_display:
            pgdp_html = "<table style='font-family: monospace; font-size: 12px;'>"
            for line_idx, line in self.current_pgdp_page.processed_lines:
                pgdp_html += f"<tr><td>{line_idx}</td><td>{line}</td></tr>"
            pgdp_html += "</table>"
            self.pgdp_text_display.content = pgdp_html

    def refresh_ui(self):
        """Refresh the entire UI"""
        ui_logger.debug("Refreshing UI for page index: " + str(self.current_page_idx))

        # Update header
        if self.current_page_display:
            self.current_page_display.text = f"Page {self.current_page_idx} of {self.total_pages}: {self.current_page_name}"

        if self.page_number_input:
            self.page_number_input.value = self.current_page_idx

        # Only run OCR and update displays if a project is loaded
        if self.is_project_ready():
            # Run OCR and update displays
            try:
                self.run_ocr()
            except Exception as e:
                ui_logger.error(f"Error running OCR: {e}")
                ui.notify(f"Error running OCR: {e}", type="negative")
                raise e
            try:
                self.reload_page_images_ui()
            except Exception as e:
                ui_logger.error(f"Error reloading page images: {e}")
                ui.notify(f"Error reloading page images: {e}", type="negative")
                raise e
            try:
                self.update_images()
            except Exception as e:
                ui_logger.error(f"Error updating images: {e}")
                ui.notify(f"Error updating images: {e}", type="negative")
                raise e
            try:
                self.update_text_displays()
            except Exception as e:
                ui_logger.error(f"Error updating text displays: {e}")
                ui.notify(f"Error updating text displays: {e}", type="negative")
                raise e

            # Update page editor with new data
            if self.page_editor:
                if not hasattr(self, "current_pgdp_page") or not hasattr(
                    self, "current_ocr_page"
                ):
                    ui_logger.warning(
                        "Page editor update skipped: current_pgdp_page or current_ocr_page not set"
                    )
                    return
                if not self.current_pgdp_page or not self.current_ocr_page:
                    ui_logger.warning(
                        "Page editor update skipped: current_pgdp_page or current_ocr_page is None"
                    )
                    return
                current_pgdp_page = self.current_pgdp_page
                current_ocr_page = self.current_ocr_page
                try:
                    self.page_editor.update_line_matches(
                        current_pgdp_page, current_ocr_page
                    )
                except Exception as e:
                    ui_logger.error(f"Error updating page editor: {e}")
                    ui.notify(f"Error updating page editor: {e}", type="negative")
                    raise e
        else:
            # Show a message when no project is loaded
            if hasattr(self, "current_page_display") and self.current_page_display:
                self.current_page_display.text = (
                    "No project loaded - Please select and load a project"
                )

    def setup_ui(self):
        """Setup the NiceGUI interface"""
        # Add custom CSS for monospace font
        ui.add_head_html(f"""
        <style>
            .monospace {{
                font-family: '{self.monospace_font_name}', monospace !important;
                font-size: 12px !important;
            }}
        </style>
        """)

        with ui.header():
            ui.label("OCR Data Labeler").classes("text-h4")

        # Navigation controls
        with ui.row().classes("w-full gap-4 items-center p-4"):
            ui.button("Previous", on_click=self.prev_page).props("flat")
            self.current_page_display = ui.label(
                f"Page {self.current_page_idx} of {self.total_pages}: {self.current_page_name}"
            )
            ui.button("Next", on_click=self.next_page).props("flat")
            self.page_number_input = ui.number(
                "Go to page",
                value=self.current_page_idx,
                min=0,
                max=self.total_pages,
                precision=0,
            )
            ui.button("Go", on_click=self.go_to_page).props("flat")

        # Action buttons
        with ui.row().classes("w-full gap-2 p-4"):
            ui.button("Save OCR to File", on_click=self.export_ocr_document).props(
                "outlined"
            )
            ui.button("Reload OCR from File", on_click=self.reload_ocr_from_file).props(
                "outlined"
            )
            ui.button("Reset OCR", on_click=self.reset_ocr).props("outlined")
            ui.button("Export Training Set", on_click=self.export_training).props(
                "outlined color=green"
            )
            ui.button("Export Validation Set", on_click=self.export_validation).props(
                "outlined color=blue"
            )

        # Additional action buttons
        with ui.row().classes("w-full gap-2 p-4"):
            ui.button(
                "Expand & Refine All BBoxes", on_click=self.expand_and_refine_all_bboxes
            ).props("outlined")
            ui.button("Refine All BBoxes", on_click=self.refine_all_bboxes).props(
                "outlined"
            )
            ui.button("Refresh Page Images", on_click=self.refresh_page_images).props(
                "outlined"
            )

        # Project loading section (only show if source_pgdp_data_path is set)
        if self.source_pgdp_data_path:
            with ui.row().classes("w-full gap-2 p-4"):
                ui.label("Load Project from Directory:").classes("text-sm")
                self.project_selector = (
                    ui.select(
                        options=self.get_available_projects(),
                        value=self.selected_project,
                    )
                    .props("filled")
                    .classes("min-w-48")
                )
                self.load_project_button = ui.button(
                    "Load Project", on_click=self.on_load_project_clicked
                ).props("outlined")
                if self.project_loaded:
                    self.load_project_button.disable()
                    self.load_project_button.set_text("Project Loaded")

        # Main content area
        with ui.splitter(value=30).classes("w-full h-96") as splitter:
            with splitter.before:
                # Image tabs
                with ui.tabs().classes("w-full") as tabs:
                    mismatches_tab = ui.tab("Mismatches")
                    original_tab = ui.tab("Original")
                    paragraphs_tab = ui.tab("Paragraphs")
                    lines_tab = ui.tab("Lines")
                    words_tab = ui.tab("Words")

                with ui.tab_panels(tabs, value=mismatches_tab).classes("w-full"):
                    with ui.tab_panel(mismatches_tab):
                        self.ocr_image_mismatches = ui.image().classes(
                            "max-w-full max-h-96"
                        )

                    with ui.tab_panel(original_tab):
                        self.plain_image = ui.image().classes("max-w-full max-h-96")

                    with ui.tab_panel(paragraphs_tab):
                        self.ocr_image_pgh_bounding_box = ui.image().classes(
                            "max-w-full max-h-96"
                        )

                    with ui.tab_panel(lines_tab):
                        self.ocr_image_lines_bounding_box = ui.image().classes(
                            "max-w-full max-h-96"
                        )

                    with ui.tab_panel(words_tab):
                        self.ocr_image_words_bounding_box = ui.image().classes(
                            "max-w-full max-h-96"
                        )

            with splitter.after:
                # Text display tabs
                with ui.tabs().classes("w-full") as text_tabs:
                    matching_tab = ui.tab("Line Matching")
                    ocr_text_tab = ui.tab("OCR Text")
                    pgdp_text_tab = ui.tab("PGDP P3 Text")

                with ui.tab_panels(text_tabs, value=matching_tab).classes("w-full"):
                    with ui.tab_panel(matching_tab):
                        # Initialize page editor
                        page_editor_container = ui.column().classes(
                            "w-full h-full overflow-auto"
                        )

                        def page_image_change_callback():
                            """Callback when page images need to be refreshed"""
                            try:
                                self.current_ocr_page.refresh_page_images()
                                self.update_images()
                            except (KeyError, AttributeError):
                                pass

                        self.page_editor = NiceGuiPageEditor(
                            current_pgdp_page=None,  # Will be set in refresh_ui
                            current_ocr_page=None,  # Will be set in refresh_ui
                            monospace_font_name=self.monospace_font_name,
                            page_image_change_callback=page_image_change_callback,
                        )
                        self.page_editor.draw_ui(page_editor_container)

                    with ui.tab_panel(ocr_text_tab):
                        self.ocr_text_display = ui.html().classes(
                            "monospace overflow-auto h-80"
                        )

                    with ui.tab_panel(pgdp_text_tab):
                        self.pgdp_text_display = ui.html().classes(
                            "monospace overflow-auto h-80"
                        )

        # Initialize the UI with current data
        self.refresh_ui()

    def is_project_ready(self) -> bool:
        """Check if a project is loaded and ready for operations"""
        return self.project_loaded and self.pgdp_export is not None

    def ensure_project_loaded(self) -> bool:
        """Ensure a project is loaded, show notification if not"""
        if not self.is_project_ready():
            ui.notify("Please load a project first", type="warning")
            return False
        return True

    def get_available_projects(self):
        """Get list of available projects from the source directory"""
        if not self.source_pgdp_data_path:
            return []

        output_dir = self.source_pgdp_data_path / "output"
        if not output_dir.exists():
            return []

        projects = []
        for item in output_dir.iterdir():
            if item.is_dir():
                pages_json = item / "pages.json"
                if pages_json.exists():
                    projects.append(item.name)

        return sorted(projects)

    def load_project(self, project_name: str):
        """Load a project from the source directory"""
        if not self.source_pgdp_data_path:
            ui.notify("No source data path configured", type="negative")
            return False

        project_dir = self.source_pgdp_data_path / "output" / project_name
        pages_json_path = project_dir / "pages.json"

        if not pages_json_path.exists():
            ui.notify(
                f"Project {project_name} not found or missing pages.json",
                type="negative",
            )
            return False

        try:
            # Load the PGDP export from the JSON file
            self.pgdp_export = PGDPExport.from_json_file(pages_json_path)

            # Update the project ID
            self.pgdp_export.project_id = project_name

            # Initialize with the loaded export
            self._initialize_with_pgdp_export()

            ui.notify(f"Project {project_name} initialized", type="positive")

            # Update UI to reflect the loaded project
            self.selected_project = project_name
            self.refresh_ui()

            ui.notify(f"Project {project_name} loaded successfully", type="positive")
            return True

        except Exception as e:
            ui.notify(
                f"Error loading project {project_name}: {str(e)}", type="negative"
            )
            return False

    def on_load_project_clicked(self):
        """Handle the Load Project button click"""
        if not self.project_selector or not self.project_selector.value:
            ui.notify("Please select a project first", type="warning")
            return

        success = self.load_project(self.project_selector.value)
        if success and self.load_project_button:
            self.load_project_button.disable()
            self.load_project_button.set_text("Project Loaded")

    def run(self, host="localhost", port=8080):
        """Run the NiceGUI application"""
        self.setup_ui()
        ui.run(host=host, port=port, title="OCR Data Labeler")


# Example usage function
def create_nicegui_labeler(
    pgdp_export: PGDPExport,
    labeled_ocr_path: pathlib.Path | str,
    training_set_output_path: pathlib.Path | str,
    validation_set_output_path: pathlib.Path | str,
    monospace_font_name: str = "monospace",
    monospace_font_path: pathlib.Path | str | None = None,
    start_page_name="",
    start_page_idx=0,
    doctr_predictor=None,
    host="localhost",
    port=8080,
) -> NiceGuiLabeler:
    """
    Create and run a NiceGUI version of the OCR data labeler

    Args:
        pgdp_export: PGDP export data
        labeled_ocr_path: Path to save labeled OCR data
        training_set_output_path: Path to save training data
        validation_set_output_path: Path to save validation data
        monospace_font_name: Name of monospace font to use
        monospace_font_path: Path to monospace font file
        start_page_name: Starting page name
        start_page_idx: Starting page index
        doctr_predictor: OCR predictor to use
        host: Host to run server on
        port: Port to run server on

    Returns:
        NiceGuiLabeler instance
    """
    labeler = NiceGuiLabeler(
        pgdp_export=pgdp_export,
        labeled_ocr_path=labeled_ocr_path,
        training_set_output_path=training_set_output_path,
        validation_set_output_path=validation_set_output_path,
        monospace_font_name=monospace_font_name,
        monospace_font_path=monospace_font_path,
        start_page_name=start_page_name,
        start_page_idx=start_page_idx,
        doctr_predictor=doctr_predictor,
    )

    labeler.run(host=host, port=port)
    return labeler
